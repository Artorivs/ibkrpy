# ibkrpy/data/artifact_store.py
# Storage for training artifacts (scaler + feature manifest)

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

WF_MARKER = "__wf"

SCALER = "scaler"
MANIFEST = "manifest"

_LEGACY_SUFFIX = {"_scaler.json": SCALER, "_features.json": MANIFEST}


class ArtifactStore(ABC):
    """The only interface DataPipeline depends on."""

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
        """One atomic write, so scaler and manifest are never half-updated."""

    @abstractmethod
    def delete(self, symbol: str) -> None: ...

    @abstractmethod
    def symbols(self) -> List[str]: ...


class NullArtifactStore(ArtifactStore):
    """In-memory only. Used by walk-forward and tests."""

    def load_scaler(self, symbol):
        return None

    def load_manifest(self, symbol):
        return None

    def save_scaler(self, symbol, scaler):
        pass

    def save_manifest(self, symbol, manifest):
        pass

    def save_bundle(self, symbol, scaler=None, manifest=None):
        pass

    def delete(self, symbol):
        pass

    def symbols(self):
        return []


class LegacyFileStore(ArtifactStore):
    """
    Legacy format: weights/{symbol}_scaler.json and weights/{symbol}_features.json.
    Fully readable and writable, so scaler_settings.consolidated=false still works.
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
            logger.error(f"[{symbol}] Failed to read legacy {kind} ({path}): {e}")
            return None

    def _write(self, symbol: str, kind: str, blob) -> None:
        os.makedirs(self.weights_dir, exist_ok=True)
        with open(self._path(symbol, kind), "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=4, ensure_ascii=False)

    def load_scaler(self, symbol):
        return self._read(symbol, SCALER)

    def load_manifest(self, symbol):
        return self._read(symbol, MANIFEST)

    def save_scaler(self, symbol, scaler):
        self._write(symbol, SCALER, scaler)

    def save_manifest(self, symbol, manifest):
        self._write(symbol, MANIFEST, manifest)

    def save_bundle(self, symbol, scaler=None, manifest=None):
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


class LegacyScalersJsonStore(ArtifactStore):
    """
    Transitional format: weights/_scalers.json ({"version":1,"scalers":{...}}).
    """

    def __init__(self, path: str):
        self.path = path
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
            logger.error(f"Failed to read scalers.json ({self.path}): {e}")
            return self._cache or {}

    def load_scaler(self, symbol):
        return self._read_all().get(symbol)

    def load_manifest(self, symbol):
        return None

    def save_scaler(self, symbol, scaler):
        pass

    def save_manifest(self, symbol, manifest):
        pass

    def save_bundle(self, symbol, scaler=None, manifest=None):
        pass

    def delete(self, symbol):
        pass

    def symbols(self):
        return sorted(self._read_all().keys())


class ChainedArtifactStore(ArtifactStore):
    """
    Tries several read-only fallbacks in order; the first non-None result wins.
    """

    def __init__(self, stores: List[ArtifactStore]):
        self.stores = [s for s in stores if s is not None]

    def _first(self, method: str, symbol: str):
        for st in self.stores:
            try:
                blob = getattr(st, method)(symbol)
            except Exception:
                continue
            if blob is not None:
                return blob
        return None

    def load_scaler(self, symbol):
        return self._first("load_scaler", symbol)

    def load_manifest(self, symbol):
        return self._first("load_manifest", symbol)

    def save_scaler(self, symbol, scaler):
        pass

    def save_manifest(self, symbol, manifest):
        pass

    def save_bundle(self, symbol, scaler=None, manifest=None):
        pass

    def delete(self, symbol):
        for st in self.stores:
            try:
                st.delete(symbol)
            except Exception:
                pass

    def symbols(self):
        out = set()
        for st in self.stores:
            try:
                out.update(st.symbols())
            except Exception:
                pass
        return sorted(out)


class _FileLock:
    """
    Cross-process lock using an O_EXCL lock file. Retraining runs in its own process
    (see retrain_client_id), and two concurrent read-modify-writes of one JSON clobber
    each other. A stale lock is taken over on timeout rather than blocking training.
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
                            f"Training-artifact lock wait timed out (lock is {age:.0f}s old); taking it."
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
    Current format: a single weights/training_artifacts.json.

    The cache is invalidated by file mtime, so a live process picks up a
    retrain's new values without restarting.
    """

    VERSION = 1

    def __init__(self, path: str, fallback: Optional[ArtifactStore] = None):
        self.path = path
        self.fallback = fallback
        self._cache: Dict[str, dict] = {}
        self._mtime: float = -1.0
        self._warned_unpaired: set = set()

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
            logger.error(f"Failed to read training_artifacts.json ({self.path}): {e}")
            return self._cache or {}

    def _write_all(self, data: Dict[str, dict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        blob = {"version": self.VERSION, "artifacts": data}
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
                logger.debug(
                    f"[{symbol}] {kind} loaded from the legacy format (not migrated yet)."
                )
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
        A manifest without a scaler (or the reverse) means the last training run stopped
        halfway. Predictions built on that pairing aren't reliable, so say so.
        """
        if symbol in self._warned_unpaired:
            return
        has_s = self._get(symbol, SCALER) is not None
        has_m = self._get(symbol, MANIFEST) is not None
        if has_s != has_m:
            self._warned_unpaired.add(symbol)
            missing = "scaler" if has_m else "manifest"
            logger.error(
                f"[{symbol}] Training artifacts are unpaired; missing {missing}. "
                f"The last training run failed partway; predictions are unreliable. Run --mode train."
            )

    def save_scaler(self, symbol: str, scaler: ScalerDict) -> None:
        self.save_bundle(symbol, scaler=scaler)

    def save_manifest(self, symbol: str, manifest: ManifestDict) -> None:
        self.save_bundle(symbol, manifest=manifest)

    def save_bundle(self, symbol, scaler=None, manifest=None) -> None:
        if WF_MARKER in symbol:
            return
        if scaler is None and manifest is None:
            return
        with _FileLock(self.path):
            self._mtime = -1.0
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

    def audit(self) -> Dict[str, List[str]]:
        """Returns {"paired": [...], "scaler_only": [...], "manifest_only": [...]}"""
        out = {"paired": [], "scaler_only": [], "manifest_only": []}
        for sym, entry in self._read_all().items():
            has_s = entry.get(SCALER) is not None
            has_m = entry.get(MANIFEST) is not None
            key = (
                "paired"
                if (has_s and has_m)
                else ("scaler_only" if has_s else "manifest_only")
            )
            out[key].append(sym)
        return {k: sorted(v) for k, v in out.items()}

    def migrate_from(
        self, legacy: ArtifactStore, remove_old: bool = False
    ) -> Tuple[int, int]:
        """Merge legacy per-file artifacts into the single file. Returns (symbols, artifacts). Idempotent."""
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
                        continue
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


def _setting(config, key: str, default):
    """
    config can be a dict or a ConfigManager (dot-path get); both must work,
    or settings silently come back empty.
    """
    if config is None:
        return default
    try:
        val = config.get(f"scaler_settings.{key}")
        if val is not None:
            return val
    except Exception:
        pass
    try:
        section = config.get("scaler_settings")
        if isinstance(section, dict) and key in section:
            return section[key]
    except Exception:
        pass
    return default


def build_artifact_store(weights_dir: str, config=None) -> ArtifactStore:
    """
    Composition root. Uses the consolidated format and keeps every legacy read fallback.
    """
    legacy_files = LegacyFileStore(weights_dir)
    if not _setting(config, "consolidated", True):
        return legacy_files

    filename = _setting(config, "filename", "training_artifacts.json")
    fallback = ChainedArtifactStore(
        [
            LegacyScalersJsonStore(os.path.join(weights_dir, "scalers.json")),
            legacy_files,
        ]
    )
    return ConsolidatedArtifactStore(
        os.path.join(weights_dir, filename), fallback=fallback
    )
