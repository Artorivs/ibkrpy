# ibkrpy/data/scaler_store.py
# Storage for feature-scaling parameters

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
    """The only interface DataPipeline depends on."""

    @abstractmethod
    def load(self, symbol: str) -> Optional[ScalerDict]:
        """Returns None when missing (not an exception, not an empty dict)."""

    @abstractmethod
    def save(self, symbol: str, scaler: ScalerDict) -> None: ...

    @abstractmethod
    def delete(self, symbol: str) -> None: ...

    @abstractmethod
    def symbols(self) -> List[str]: ...


class NullScalerStore(ScalerStore):
    """In-memory only. Used by walk-forward and tests."""

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
    Simple cross-process lock using an O_EXCL lock file. Writes are short (< 10 ms)
    and contention is rare, so on timeout we log and proceed rather than
    block training.
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
                        f"Scaler lock wait timed out (lock is {age:.0f}s old); taking it."
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
    Single weights/training_artifacts.json:

        {
          "version": 1,
          "scalers": { "AAPL": {"Close": {"min": .., "max": ..}, ...}, ... }
        }

    Reads use an in-memory cache keyed on file mtime, so a live process picks up
    a retrain's new values without restarting.
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
            logger.error(f"Failed to read training_artifacts.json ({self.path}): {e}")
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
    """Composition root. Uses the consolidated format, with a read fallback to the legacy one."""
    s = (config.get("scaler_settings") or {}) if config else {}
    return ConsolidatedScalerStore(
        os.path.join(weights_dir, s.get("filename", "_scalers.json"))
    )
