# ibkrpy/data/scaler_store.py
# 特徵縮放參數的儲存層

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger("ibkrpy.scaler_store")

ScalerDict = Dict[str, Dict[str, float]]

WF_MARKER = "__wf"


class ScalerStore(ABC):
    """DataPipeline 唯一依賴的介面。"""

    @abstractmethod
    def load(self, symbol: str) -> Optional[ScalerDict]:
        """讀不到回傳 None (不是拋例外，也不是空 dict —— 兩者語意不同)。"""

    @abstractmethod
    def save(self, symbol: str, scaler: ScalerDict) -> None: ...

    @abstractmethod
    def delete(self, symbol: str) -> None: ...

    @abstractmethod
    def symbols(self) -> List[str]: ...


class NullScalerStore(ScalerStore):
    """不落盤。walk-forward 與測試使用。"""

    def load(self, symbol):
        return None

    def save(self, symbol, scaler):
        pass

    def delete(self, symbol):
        pass

    def symbols(self):
        return []


class _FileLock:
    """
    以 O_EXCL 建立鎖檔的簡易跨行程鎖。夠用是因為寫入本身很短 (< 10ms)，
    而且重訓行程與實盤行程不會高頻爭用。逾時就放行並記錄 —— 寧可有極小
    機率覆寫，也不要讓訓練整個卡死。
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
                    age = time.time() - os.path.getmtime(self.path)
                    logger.warning(
                        f"scaler 鎖等待逾時 (鎖已存在 {age:.0f} 秒)，強制取得。"
                    )
                    try:
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


class ConsolidatedScalerStore(ScalerStore):
    """
    單一 weights/_training_artifacts.json

        {
          "version": 1,
          "scalers": { "AAPL": {"Close": {"min": .., "max": ..}, ...}, ... }
        }

    走讀取路徑時會查記憶體快取，並以檔案 mtime 判斷是否需要重讀 ——
    重訓行程更新檔案後，實盤行程不必重啟就會拿到新值。
    """

    VERSION = 1

    def __init__(self, path: str, fallback: Optional[ScalerStore] = None):
        self.path = path
        self.fallback = fallback
        self._cache: Dict[str, ScalerDict] = {}
        self._mtime: float = -1.0


    def _read_all(self) -> Dict[str, ScalerDict]:
        if not os.path.exists(self.path):
            return {}
        try:
            mtime = os.path.getmtime(self.path)
            if mtime == self._mtime and self._cache:
                return self._cache
            with open(self.path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            data = blob.get("scalers", {}) if isinstance(blob, dict) else {}
            self._cache, self._mtime = data, mtime
            return data
        except Exception as e:
            logger.error(f"_training_artifacts.json 讀取失敗 ({self.path}): {e}")
            return self._cache or {}

    def _write_all(self, data: Dict[str, ScalerDict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        blob = {"version": self.VERSION, "scalers": data}
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2, sort_keys=True)
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


    def load(self, symbol: str) -> Optional[ScalerDict]:
        found = self._read_all().get(symbol)
        if found is not None:
            return found
        return None

    def save(self, symbol: str, scaler: ScalerDict) -> None:
        if WF_MARKER in symbol:
            return
        with _FileLock(self.path):
            self._mtime = -1.0
            data = dict(self._read_all())
            data[symbol] = scaler
            self._write_all(data)

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


def build_scaler_store(weights_dir: str, config=None) -> ScalerStore:
    """Composition Root 使用。預設啟用合併格式，並保留舊格式的讀取退路。"""
    s = (config.get("scaler_settings") or {}) if config else {}
    return ConsolidatedScalerStore(
        os.path.join(weights_dir, s.get("filename", "_scalers.json"))
    )
