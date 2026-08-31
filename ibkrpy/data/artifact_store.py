# ibkrpy/data/artifact_store.py
# 訓練產物 (scaler + feature manifest) 的儲存層
#
# 檔案佈局
# --------
#   weights/training_artifacts.json
#   {
#     "version": 1,
#     "artifacts": {
#       "AAPL": {
#         "scaler":   {"Close": {"min": .., "max": ..}, ...},
#         "manifest": {"features": [...], "price_relative": [...], ...},
#         "updated":  "2026-08-31T12:00:00Z"
#       }
#     }
#   }

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ibkrpy.artifact_store")

ScalerDict = Dict[str, Dict[str, float]]
ManifestDict = Dict[str, object]

# Walk-forward 產生的暫時標的。這些產物不落盤。
WF_MARKER = "__wf"

SCALER = "scaler"
MANIFEST = "manifest"

# 舊格式的檔名後綴 -> bundle 內的鍵
_LEGACY_SUFFIX = {"_scaler.json": SCALER, "_features.json": MANIFEST}


class ArtifactStore(ABC):
    """DataPipeline 唯一依賴的介面。"""

    @abstractmethod
    def load_scaler(self, symbol: str) -> Optional[ScalerDict]: ...

    @abstractmethod
    def load_manifest(self, symbol: str) -> Optional[ManifestDict]: ...

    @abstractmethod
    def save_scaler(self, symbol: str, scaler: ScalerDict) -> None: ...

    @abstractmethod
    def save_manifest(self, symbol: str, manifest: ManifestDict) -> None: ...

    @abstractmethod
    def save_bundle(
        self,
        symbol: str,
        scaler: Optional[ScalerDict] = None,
        manifest: Optional[ManifestDict] = None,
    ) -> None:
        """一次原子寫入。scaler 與 manifest 不會出現只更新一半的狀態。"""

    @abstractmethod
    def delete(self, symbol: str) -> None: ...

    @abstractmethod
    def symbols(self) -> List[str]: ...


class NullArtifactStore(ArtifactStore):
    """不落盤。walk-forward 與測試使用。"""

    def load_scaler(self, symbol): return None
    def load_manifest(self, symbol): return None
    def save_scaler(self, symbol, scaler): pass
    def save_manifest(self, symbol, manifest): pass
    def save_bundle(self, symbol, scaler=None, manifest=None): pass
    def delete(self, symbol): pass
    def symbols(self): return []


class LegacyFileStore(ArtifactStore):
    """
    舊格式: weights/{symbol}_scaler.json 與 weights/{symbol}_features.json。
    保留完整讀寫能力，讓 scaler_settings.consolidated=false 仍可運作。
    """

    def __init__(self, weights_dir: str):
        self.weights_dir = weights_dir

    def _path(self, symbol: str, kind: str) -> str:
        suffix = "_scaler.json" if kind == SCALER else "_features.json"
        return os.path.join(self.weights_dir, f"{symbol}{suffix}")

    def _read(self, symbol: str, kind: str):
        path = self._path(symbol, kind)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[{symbol}] 舊格式 {kind} 讀取失敗 ({path}): {e}")
            return None

    def _write(self, symbol: str, kind: str, blob) -> None:
        os.makedirs(self.weights_dir, exist_ok=True)
        with open(self._path(symbol, kind), "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=4, ensure_ascii=False)

    def load_scaler(self, symbol): return self._read(symbol, SCALER)
    def load_manifest(self, symbol): return self._read(symbol, MANIFEST)
    def save_scaler(self, symbol, scaler): self._write(symbol, SCALER, scaler)
    def save_manifest(self, symbol, manifest): self._write(symbol, MANIFEST, manifest)

    def save_bundle(self, symbol, scaler=None, manifest=None):
        # 舊格式本質上做不到跨檔原子性，這正是要遷移的理由之一。
        if manifest is not None:
            self.save_manifest(symbol, manifest)
        if scaler is not None:
            self.save_scaler(symbol, scaler)

    def delete(self, symbol: str) -> None:
        for kind in (SCALER, MANIFEST):
            try:
                os.remove(self._path(symbol, kind))
            except FileNotFoundError:
                pass

    def symbols(self) -> List[str]:
        if not os.path.isdir(self.weights_dir):
            return []
        found = set()
        for name in os.listdir(self.weights_dir):
            for suffix in _LEGACY_SUFFIX:
                if name.endswith(suffix):
                    found.add(name[: -len(suffix)])
        return sorted(found)


class _FileLock:
    """
    以 O_EXCL 建立鎖檔的跨行程鎖。重訓是獨立行程 (見 retrain_client_id)，
    兩個行程同時 read-modify-write 同一個 JSON 會互相蓋掉。
    逾時就搶下陳舊的鎖 —— 寧可極小機率覆寫，也不要讓訓練永久卡死。
    """

    def __init__(self, path: str, timeout: float = 10.0):
        self.path, self.timeout, self.fd = path + ".lock", timeout, None

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    try:
                        age = time.time() - os.path.getmtime(self.path)
                        logger.warning(
                            f"訓練產物鎖等待逾時 (鎖已存在 {age:.0f} 秒)，強制取得。"
                        )
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            os.remove(self.path)
        except OSError:
            pass


class ConsolidatedArtifactStore(ArtifactStore):
    """
    新格式: 單一 weights/training_artifacts.json。

    讀取時以 mtime 判斷快取是否過期，因此重訓行程更新檔案後，
    實盤行程不必重啟就會拿到新值。
    """

    VERSION = 1

    def __init__(self, path: str, fallback: Optional[ArtifactStore] = None):
        self.path = path
        self.fallback = fallback
        self._cache: Dict[str, dict] = {}
        self._mtime: float = -1.0
        self._warned_unpaired: set = set()

    # -- 檔案 I/O --

    def _read_all(self) -> Dict[str, dict]:
        if not os.path.exists(self.path):
            return {}
        try:
            mtime = os.path.getmtime(self.path)
            if mtime == self._mtime and self._cache:
                return self._cache
            with open(self.path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            data = blob.get("artifacts", {}) if isinstance(blob, dict) else {}
            self._cache, self._mtime = data, mtime
            return data
        except Exception as e:
            logger.error(f"training_artifacts.json 讀取失敗 ({self.path}): {e}")
            return self._cache or {}

    def _write_all(self, data: Dict[str, dict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        blob = {"version": self.VERSION, "artifacts": data}
        # 原子寫入: 先寫暫存檔再 rename。中途當機不會留下半個 JSON。
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        self._cache = data
        self._mtime = os.path.getmtime(self.path)

    # -- 讀取 --

    def _get(self, symbol: str, kind: str):
        entry = self._read_all().get(symbol)
        if entry is not None and entry.get(kind) is not None:
            return entry[kind]
        if self.fallback is not None:
            legacy = (
                self.fallback.load_scaler(symbol)
                if kind == SCALER
                else self.fallback.load_manifest(symbol)
            )
            if legacy is not None:
                logger.debug(f"[{symbol}] {kind} 由舊格式載入 (尚未遷移)。")
            return legacy
        return None

    def load_scaler(self, symbol: str) -> Optional[ScalerDict]:
        return self._get(symbol, SCALER)

    def load_manifest(self, symbol: str) -> Optional[ManifestDict]:
        blob = self._get(symbol, MANIFEST)
        if blob is not None:
            self._warn_if_unpaired(symbol)
        return blob

    def _warn_if_unpaired(self, symbol: str) -> None:
        """
        manifest 有、scaler 沒有 (或反之) 代表上一次訓練寫到一半就中斷。
        用這種配對推論出來的結果不可信，必須讓人知道。
        """
        if symbol in self._warned_unpaired:
            return
        has_s = self._get(symbol, SCALER) is not None
        has_m = self._get(symbol, MANIFEST) is not None
        if has_s != has_m:
            self._warned_unpaired.add(symbol)
            missing = "scaler" if has_m else "manifest"
            logger.error(
                f"[{symbol}] 訓練產物不成對 —— 缺少 {missing}。"
                f"這代表上次訓練中途失敗，該標的的預測不可信，請重新執行 --mode train。"
            )

    # -- 寫入 --

    def save_scaler(self, symbol: str, scaler: ScalerDict) -> None:
        self.save_bundle(symbol, scaler=scaler)

    def save_manifest(self, symbol: str, manifest: ManifestDict) -> None:
        self.save_bundle(symbol, manifest=manifest)

    def save_bundle(self, symbol, scaler=None, manifest=None) -> None:
        if WF_MARKER in symbol:
            return  # walk-forward 不落盤
        if scaler is None and manifest is None:
            return
        with _FileLock(self.path):
            self._mtime = -1.0  # 強制重讀，避免蓋掉其他行程剛寫入的內容
            data = dict(self._read_all())
            entry = dict(data.get(symbol) or {})
            if scaler is not None:
                entry[SCALER] = scaler
            if manifest is not None:
                entry[MANIFEST] = manifest
            entry["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            )
            data[symbol] = entry
            self._write_all(data)
        self._warned_unpaired.discard(symbol)

    def delete(self, symbol: str) -> None:
        with _FileLock(self.path):
            self._mtime = -1.0
            data = dict(self._read_all())
            if data.pop(symbol, None) is not None:
                self._write_all(data)
        if self.fallback is not None:
            self.fallback.delete(symbol)

    def symbols(self) -> List[str]:
        return sorted(self._read_all().keys())

    # -- 稽核與遷移 --

    def audit(self) -> Dict[str, List[str]]:
        """回傳 {"paired": [...], "scaler_only": [...], "manifest_only": [...]}"""
        out = {"paired": [], "scaler_only": [], "manifest_only": []}
        for sym, entry in self._read_all().items():
            has_s = entry.get(SCALER) is not None
            has_m = entry.get(MANIFEST) is not None
            key = "paired" if (has_s and has_m) else ("scaler_only" if has_s else "manifest_only")
            out[key].append(sym)
        return {k: sorted(v) for k, v in out.items()}

    def migrate_from(self, legacy: ArtifactStore, remove_old: bool = False) -> Tuple[int, int]:
        """把舊的逐檔產物併入單一檔案。回傳 (標的數, 產物數)。可重複執行。"""
        syms = moved = 0
        with _FileLock(self.path):
            self._mtime = -1.0
            data = dict(self._read_all())
            for sym in legacy.symbols():
                if WF_MARKER in sym:
                    continue
                entry = dict(data.get(sym) or {})
                touched = False
                for kind, loader in (
                    (SCALER, legacy.load_scaler),
                    (MANIFEST, legacy.load_manifest),
                ):
                    if entry.get(kind) is not None:
                        continue  # 已遷移，不覆寫
                    blob = loader(sym)
                    if blob:
                        entry[kind] = blob
                        moved += 1
                        touched = True
                if touched:
                    entry.setdefault(
                        "updated",
                        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    )
                    data[sym] = entry
                    syms += 1
            if moved:
                self._write_all(data)
        if remove_old:
            for sym in legacy.symbols():
                legacy.delete(sym)
        return syms, moved


def build_artifact_store(weights_dir: str, config=None) -> ArtifactStore:
    """Composition Root 使用。預設合併格式，並保留舊格式的讀取退路。"""
    s = (config.get("scaler_settings") or {}) if config else {}
    legacy = LegacyFileStore(weights_dir)
    if not s.get("consolidated", True):
        return legacy
    return ConsolidatedArtifactStore(
        os.path.join(weights_dir, s.get("filename", "training_artifacts.json")),
        fallback=legacy,
    )