# ibkrpy/strategy/prediction_calibrator.py
# Entry threshold calibration and model collapse detection.

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence

import numpy as np

logger = logging.getLogger("ibkrpy.calibrator")


@dataclass(frozen=True)
class ThresholdContext:
    """All inputs for a threshold. New fields don't break existing policies."""

    symbol: str
    sigma: float
    history: Sequence[float]
    min_edge_pct: float = 0.0005
    term: str = "long_term"


@dataclass(frozen=True)
class ThresholdDecision:
    value: float
    source: str
    detail: str = ""


@dataclass
class CollapseReport:
    """Model collapse diagnosis."""

    collapsed: bool
    samples: int
    dispersion: float
    unique_ratio: float
    detail: str = ""


class PredictionHistoryStore(ABC):
    """Per-symbol history of predicted moves, for threshold calibration and collapse detection."""

    @abstractmethod
    def append(self, symbol: str, value: float) -> None: ...

    @abstractmethod
    def get(self, symbol: str) -> List[float]: ...


class InMemoryPredictionHistory(PredictionHistoryStore):
    """Rolling window. Resets on restart; for backtests and tests."""

    def __init__(self, maxlen: int = 500):
        self._maxlen = maxlen
        self._data: Dict[str, Deque[float]] = {}

    def append(self, symbol: str, value: float) -> None:
        if value is None or not np.isfinite(value):
            return
        key = str(symbol).upper()
        if key not in self._data:
            self._data[key] = deque(maxlen=self._maxlen)
        self._data[key].append(float(value))

    def get(self, symbol: str) -> List[float]:
        return list(self._data.get(str(symbol).upper(), ()))

    def snapshot(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self._data.items()}


class JsonPredictionHistory(InMemoryPredictionHistory):
    """
    Persisted version, so a live restart doesn't have to wait for min_samples scans
    before trading again.

    Extends the in-memory store and adds throttled writes to disk.
    """

    def __init__(self, path: str, maxlen: int = 500, flush_every: int = 25):
        super().__init__(maxlen=maxlen)
        self._path = path
        self._flush_every = max(1, int(flush_every))
        self._pending = 0
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            for sym, values in raw.items():
                for v in values:
                    super().append(sym, v)
            logger.debug(f"Loaded prediction history for {len(raw)} symbols.")
        except Exception as e:
            logger.warning(f"Failed to load prediction history ({self._path}): {e}")

    def append(self, symbol: str, value: float) -> None:
        super().append(symbol, value)
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self.snapshot(), f)
            self._pending = 0
        except Exception as e:
            logger.warning(f"Failed to write prediction history: {e}")


class ThresholdPolicy(ABC):
    """
    Contract: returns a positive threshold (decimal) and never raises.
    All implementations are interchangeable, so policies can be swapped safely.
    """

    @abstractmethod
    def compute(self, ctx: ThresholdContext) -> ThresholdDecision: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class SigmaThresholdPolicy(ThresholdPolicy):
    """
    Threshold = σ × multiplier. Kept for two uses:
        1. a one-line fallback
        2. a baseline to compare other policies against

    Not recommended in production: σ is on the price scale, not the prediction's scale.
    """

    def __init__(self, multiplier: float = 1.0, static_floor: float = 0.0):
        self._mult = float(multiplier)
        self._floor = float(static_floor)

    def compute(self, ctx: ThresholdContext) -> ThresholdDecision:
        value = max(self._floor, ctx.sigma * self._mult)
        return ThresholdDecision(value, "sigma", f"σ×{self._mult:g}")


class QuantileThresholdPolicy(ThresholdPolicy):
    """
    Threshold = the q-th quantile of the symbol's past |prediction|.

    Means "enter only when this model is unusually confident", rather than "wait for a
    prediction larger than market noise", which no real next-bar predictor produces:
    its output std is about σ × √R², i.e. 0.1-0.2σ at a few percent R², so requiring
    |prediction| > 1σ means waiting for a 5-10 std outlier of its own distribution.

    With too few samples it returns source="insufficient" and the fallback chain
    decides what to use.
    """

    def __init__(self, quantile: float = 0.85, min_samples: int = 40):
        self._q = min(max(float(quantile), 0.5), 0.999)
        self._min_samples = int(min_samples)

    def compute(self, ctx: ThresholdContext) -> ThresholdDecision:
        hist = [abs(float(v)) for v in ctx.history if np.isfinite(v)]
        if len(hist) < self._min_samples:
            return ThresholdDecision(
                float("inf"),
                "insufficient",
                f"{len(hist)}/{self._min_samples} samples",
            )
        value = float(np.quantile(hist, self._q))
        if not np.isfinite(value) or value <= 0:
            return ThresholdDecision(float("inf"), "insufficient", "quantile is 0")
        return ThresholdDecision(
            value,
            "quantile",
            f"P{self._q * 100:.0f} of {len(hist)} samples",
        )


class CostFloorThresholdPolicy(ThresholdPolicy):
    """
    Wraps another policy with a trading-cost floor.

    A safety valve: without it the quantile policy would keep trading when the model
    predicts 0.06% and the round trip costs 0.05%, which just pays the spread.
    """

    def __init__(self, inner: ThresholdPolicy, absolute_floor: float):
        self._inner = inner
        self._floor = float(absolute_floor)

    def compute(self, ctx: ThresholdContext) -> ThresholdDecision:
        inner = self._inner.compute(ctx)
        floor = max(self._floor, float(ctx.min_edge_pct or 0.0))
        if inner.value >= floor:
            return inner
        return ThresholdDecision(
            floor,
            "cost_floor",
            f"{inner.source} gave {inner.value * 100:.3f}%, below the cost floor {floor * 100:.3f}%",
        )

    @property
    def name(self) -> str:
        return f"CostFloor({self._inner.name})"


class FallbackThresholdPolicy(ThresholdPolicy):
    """
    Tries policies in order; the first finite value wins.
    On cold start the quantile policy lacks samples and the σ policy takes over.
    """

    def __init__(self, policies: Sequence[ThresholdPolicy]):
        self._policies = list(policies)
        if not self._policies:
            raise ValueError("FallbackThresholdPolicy needs at least one policy")

    def compute(self, ctx: ThresholdContext) -> ThresholdDecision:
        last = None
        for policy in self._policies:
            try:
                decision = policy.compute(ctx)
            except Exception as e:
                logger.error(
                    f"[{ctx.symbol}] {policy.name} raised, breaking its contract: {e}"
                )
                continue
            last = decision
            if np.isfinite(decision.value):
                return decision
        return last or ThresholdDecision(
            float("inf"), "none", "no policy produced a threshold"
        )

    @property
    def name(self) -> str:
        return "Fallback(" + ", ".join(p.name for p in self._policies) + ")"


class CollapseDetector:
    """
    Detects model output that has collapsed to a constant.

    Regressing next-bar log returns with MSE lets a network settle on the sample mean
    (about 0) when returns are nearly unpredictable. It then outputs the same value every
    scan, with no exception or NaN, so it has to be detected explicitly.
    """

    def __init__(
        self,
        min_samples: int = 20,
        dispersion_floor: float = 1e-5,
        unique_ratio_floor: float = 0.05,
    ):
        self._min_samples = int(min_samples)
        self._dispersion_floor = float(dispersion_floor)
        self._unique_floor = float(unique_ratio_floor)

    def inspect(self, history: Sequence[float]) -> CollapseReport:
        values = [float(v) for v in history if v is not None and np.isfinite(v)]
        n = len(values)
        if n < self._min_samples:
            return CollapseReport(
                False,
                n,
                float("nan"),
                float("nan"),
                f"too few samples ({n}/{self._min_samples}); not judging yet",
            )

        dispersion = float(np.std(values))
        unique_ratio = len(set(np.round(values, 10))) / n

        if dispersion <= self._dispersion_floor:
            return CollapseReport(
                True,
                n,
                dispersion,
                unique_ratio,
                f"prediction std {dispersion:.2e} ≈ 0; output collapsed to a constant",
            )
        if unique_ratio <= self._unique_floor:
            return CollapseReport(
                True,
                n,
                dispersion,
                unique_ratio,
                f"only {int(unique_ratio * n)} distinct values in {n} predictions",
            )
        return CollapseReport(False, n, dispersion, unique_ratio, "output varies")


def build_threshold_policy(config) -> ThresholdPolicy:
    """
    Composition root. Builds the threshold policy from config.yaml.

    Default chain: CostFloor(Fallback(Quantile, Sigma))
        - enough history -> quantile of the prediction distribution
        - cold start     -> σ policy (conservative)
        - neither may go below the trading-cost floor
    """
    s = config.get("threshold_settings") or {}
    mode = str(s.get("mode", "quantile")).lower()

    sigma_policy = SigmaThresholdPolicy(
        multiplier=float(
            s.get(
                "sigma_multiplier",
                config.get("strategy_settings.entry_sigma_multiplier", 1.0),
            )
        ),
        static_floor=float(
            config.get("strategy_settings.min_prediction_threshold_pct", 0.0)
        ),
    )

    if mode == "sigma":
        inner: ThresholdPolicy = sigma_policy
    else:
        quantile_policy = QuantileThresholdPolicy(
            quantile=float(s.get("quantile", 0.85)),
            min_samples=int(s.get("min_samples", 40)),
        )
        cold_start = str(s.get("cold_start", "sigma")).lower()
        if cold_start == "block":
            inner = quantile_policy
        else:
            inner = FallbackThresholdPolicy([quantile_policy, sigma_policy])

    return CostFloorThresholdPolicy(
        inner, absolute_floor=float(s.get("cost_floor_pct", 0.0003))
    )


def build_prediction_history(config, weights_dir: str) -> PredictionHistoryStore:
    s = config.get("threshold_settings") or {}
    maxlen = int(s.get("history_length", 500))
    if s.get("persist_history", True):
        return JsonPredictionHistory(
            os.path.join(weights_dir, "_prediction_history.json"), maxlen=maxlen
        )
    return InMemoryPredictionHistory(maxlen=maxlen)
