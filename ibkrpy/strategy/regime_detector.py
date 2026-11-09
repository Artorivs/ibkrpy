# ibkrpy/strategy/regime_detector.py
# Market regime detection with hysteresis, asymmetric confirmation and top-reversal risk.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from .strategy_components import MarketRegime

logger = logging.getLogger("ibkrpy.regime")


@dataclass
class RegimeAssessment:
    """Full result of one regime assessment.

    reversal_risk lets "still a bull trend, but weakening" be expressed, logged and acted
    on, instead of waiting for the moving-average cross to show the trend has ended.
    """

    regime: MarketRegime
    reversal_risk: float = 0.0
    trend_strength: float = 0.0
    bars_in_regime: int = 0
    raw_regime: MarketRegime = MarketRegime.SIDEWAYS_QUIET
    reasons: List[str] = field(default_factory=list)

    @property
    def is_trending(self) -> bool:
        return self.regime in (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND)

    @property
    def is_turning(self) -> bool:
        """The trend is weakening but hasn't flipped yet."""
        return self.reversal_risk >= 0.5

    def describe(self) -> str:
        parts = [self.regime.name]
        if self.regime is not self.raw_regime:
            parts.append(f"(raw {self.raw_regime.name}, held by hysteresis)")
        parts.append(f"reversal risk {self.reversal_risk:.0%}")
        if self.reasons:
            parts.append("| " + " · ".join(self.reasons))
        return " ".join(parts)


def _safe_last(series: Optional[pd.Series], default: float = np.nan) -> float:
    """Last finite value of the series, or default; the caller decides what to do."""
    if series is None or len(series) == 0:
        return default
    value = pd.to_numeric(series, errors="coerce").dropna()
    if value.empty:
        return default
    return float(value.iloc[-1])


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder average true range.
    """
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def wilder_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ADX, as a single series (the first column, as before)."""
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = wilder_atr(df, period)
    alpha = 1.0 / period
    plus_di = (
        100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    )
    minus_di = (
        100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    )

    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


class ReversalRiskModel:
    """Combines independent signals into a 0-1 reversal risk score.

    Each signal is a pure function from DataFrame to 0-1 and knows nothing of the others;
    adding one means adding a method and registering its weight.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "momentum_fade": 0.22,
            "price_below_fast_ma": 0.20,
            "lower_highs": 0.20,
            "trend_exhaustion": 0.13,
            "volatility_expansion": 0.13,
            "distribution_volume": 0.12,
        }

    @staticmethod
    def momentum_fade(df: pd.DataFrame, fast: int, slow: int) -> float:
        """Fast MA slope flattening or turning down: upward momentum fading.

        Uses slope, not distance: during a top the fast/slow gap can still be positive
        while the fast MA has already flattened, which is the earliest sign. The slope is
        normalized by its own recent positive slope, so price level doesn't matter.
        """
        close = df["Close"]
        if len(close) < slow + fast:
            return 0.0

        ma_fast = close.rolling(fast).mean()
        slope = ma_fast.diff(5) / ma_fast.replace(0, np.nan)
        recent = slope.dropna()
        if len(recent) < 20:
            return 0.0

        now = float(recent.iloc[-1])
        positive = recent.tail(60)
        positive = positive[positive > 0]
        if positive.empty:
            return 1.0 if now <= 0 else 0.0

        norm = float(positive.median())
        if norm <= 0:
            return 0.0
        return _clamp01((norm - now) / (2.0 * norm))

    @staticmethod
    def price_below_fast_ma(df: pd.DataFrame, fast: int) -> float:
        """Close below the fast MA, scored by depth. The most direct sign of a weakening trend."""
        close = df["Close"]
        if len(close) < fast + 2:
            return 0.0
        ma_fast = _safe_last(close.rolling(fast).mean())
        last = float(close.iloc[-1])
        if not np.isfinite(ma_fast) or ma_fast <= 0:
            return 0.0
        gap = (ma_fast - last) / ma_fast
        return _clamp01(gap / 0.03)

    @staticmethod
    def trend_exhaustion(df: pd.DataFrame, adx: Optional[pd.Series]) -> float:
        """ADX falling from its high: the trend is exhausting."""
        if adx is None or len(adx.dropna()) < 6:
            return 0.0
        recent = adx.dropna().tail(8)
        peak = float(recent.max())
        now = float(recent.iloc[-1])
        if peak < 20.0:
            return 0.0
        return _clamp01((peak - now) / peak)

    @staticmethod
    def lower_highs(df: pd.DataFrame, lookback: int) -> float:
        """Pullback from the range high: a topping pattern."""
        high = df["High"]
        close = df["Close"]
        if len(high) < lookback + 2:
            return 0.0
        window_high = float(high.tail(lookback).max())
        last_close = float(close.iloc[-1])
        if window_high <= 0:
            return 0.0
        drawdown = (window_high - last_close) / window_high
        return _clamp01(drawdown / 0.06)

    @staticmethod
    def volatility_expansion(df: pd.DataFrame, atr: Optional[pd.Series]) -> float:
        """Volatility expanding vs. its own baseline: disagreement, common late in a bull run."""
        if atr is None or len(atr.dropna()) < 25:
            return 0.0
        series = atr.dropna()
        baseline = float(series.tail(25).median())
        now = float(series.iloc[-1])
        if baseline <= 0:
            return 0.0
        return _clamp01((now / baseline - 1.0) / 0.6)

    @staticmethod
    def distribution_volume(df: pd.DataFrame, lookback: int) -> float:
        """Down-bar volume exceeding up-bar volume: distribution."""
        if "Volume" not in df.columns or len(df) < lookback + 1:
            return 0.0
        recent = df.tail(lookback)
        change = recent["Close"].diff()
        volume = pd.to_numeric(recent["Volume"], errors="coerce")
        down = volume[change < 0].sum()
        up = volume[change > 0].sum()
        if up <= 0:
            return 1.0 if down > 0 else 0.0
        return _clamp01((down / up - 1.0) / 0.8)

    def score(
        self,
        df: pd.DataFrame,
        adx: Optional[pd.Series],
        atr: Optional[pd.Series],
        fast: int,
        slow: int,
        lookback: int,
    ) -> tuple[float, List[str]]:
        signals = {
            "momentum_fade": self.momentum_fade(df, fast, slow),
            "price_below_fast_ma": self.price_below_fast_ma(df, fast),
            "lower_highs": self.lower_highs(df, lookback),
            "trend_exhaustion": self.trend_exhaustion(df, adx),
            "volatility_expansion": self.volatility_expansion(df, atr),
            "distribution_volume": self.distribution_volume(df, lookback),
        }
        total_weight = sum(self.weights.get(k, 0.0) for k in signals)
        if total_weight <= 0:
            return 0.0, []

        score = (
            sum(v * self.weights.get(k, 0.0) for k, v in signals.items()) / total_weight
        )
        reasons = [
            f"{name} {value:.0%}"
            for name, value in sorted(signals.items(), key=lambda kv: -kv[1])
            if value >= 0.4
        ]
        return _clamp01(score), reasons


class MarketRegimeDetector:
    """Classifies the regime from ADX, moving averages and ATR, with hysteresis and reversal risk.

    Key choices:

    1. Longer default periods (20/50). 5/10-period MAs get crossed back and forth by the
          unfinished current bar, which flipped the regime up to 8 times a day.
    2. Hysteresis: a new regime must hold for several bars before it switches, and the
          confirmation is asymmetric: turning bearish needs fewer bars than turning bullish,
          since markets fall faster than they rise.
    3. Reversal risk rises before BEAR_TREND is confirmed, covering the gap between
          "bull is over" and "bear is confirmed".
    """

    def __init__(self, config: Dict[str, object] = None):
        cfg = config or {}

        self.adx_period = int(cfg.get("regime_adx_period", 14))
        self.ma_short = int(cfg.get("regime_ma_short", 20))
        self.ma_long = int(cfg.get("regime_ma_long", 50))
        self.atr_period = int(cfg.get("regime_atr_period", 14))
        self.adx_threshold = float(cfg.get("regime_adx_trend_threshold", 20.0))
        self.vol_threshold = float(cfg.get("regime_volatility_threshold_pct", 0.02))

        self.bull_confirm_bars = int(cfg.get("regime_bull_confirm_bars", 3))
        self.bear_confirm_bars = int(cfg.get("regime_bear_confirm_bars", 1))

        self.use_closed_bars_only = bool(cfg.get("regime_use_closed_bars_only", True))

        self.reversal_lookback = int(cfg.get("regime_reversal_lookback", 20))
        self.reversal_model = ReversalRiskModel(cfg.get("regime_reversal_weights"))

        self._state: Dict[str, Dict[str, object]] = {}

    def _indicators(self, df: pd.DataFrame) -> Dict[str, object]:
        """Compute indicators. On failure, log why and return None for the caller to handle."""
        result: Dict[str, object] = {"adx": None, "atr": None}
        try:
            result["adx"] = wilder_adx(df, self.adx_period)
            result["atr"] = wilder_atr(df, self.atr_period)
        except Exception as exc:
            logger.warning(
                f"Regime indicator calculation failed; using a neutral regime: {exc}"
            )
        return result

    def _classify(
        self, df: pd.DataFrame, adx_value: float, atr_pct: float
    ) -> MarketRegime:
        close = df["Close"]
        ma_short = _safe_last(close.rolling(self.ma_short).mean())
        ma_long = _safe_last(close.rolling(self.ma_long).mean())

        if not np.isfinite(ma_short) or not np.isfinite(ma_long):
            return MarketRegime.SIDEWAYS_QUIET

        is_trending = np.isfinite(adx_value) and adx_value > self.adx_threshold
        if is_trending:
            return (
                MarketRegime.BULL_TREND
                if ma_short > ma_long
                else MarketRegime.BEAR_TREND
            )

        is_volatile = np.isfinite(atr_pct) and atr_pct > self.vol_threshold
        return (
            MarketRegime.SIDEWAYS_VOLATILE
            if is_volatile
            else MarketRegime.SIDEWAYS_QUIET
        )

    def _confirm_bars(self, candidate: MarketRegime) -> int:
        """Turning bearish needs fewer confirmation bars."""
        if candidate is MarketRegime.BEAR_TREND:
            return self.bear_confirm_bars
        if candidate is MarketRegime.BULL_TREND:
            return self.bull_confirm_bars
        return max(self.bear_confirm_bars, 1)

    def _apply_hysteresis(
        self, key: str, candidate: MarketRegime
    ) -> tuple[MarketRegime, int]:
        state = self._state.setdefault(
            key,
            {"regime": candidate, "pending": candidate, "streak": 0, "bars": 0},
        )
        state["bars"] = int(state["bars"]) + 1

        if candidate is state["regime"]:
            state["pending"] = candidate
            state["streak"] = 0
            return state["regime"], int(state["bars"])

        if candidate is state["pending"]:
            state["streak"] = int(state["streak"]) + 1
        else:
            state["pending"] = candidate
            state["streak"] = 1

        if int(state["streak"]) >= self._confirm_bars(candidate):
            state["regime"] = candidate
            state["streak"] = 0
            state["bars"] = 1

        return state["regime"], int(state["bars"])

    def assess(self, df: pd.DataFrame, key: str = None) -> RegimeAssessment:
        """Full assessment. key keeps each symbol's hysteresis state separate."""
        minimum = max(self.ma_long, self.adx_period, self.atr_period) + 2
        if df is None or len(df) < minimum:
            return RegimeAssessment(
                regime=MarketRegime.SIDEWAYS_QUIET,
                reasons=[
                    f"not enough data ({0 if df is None else len(df)}/{minimum} bars)"
                ],
            )

        frame = df.iloc[:-1] if self.use_closed_bars_only and len(df) > minimum else df

        indicators = self._indicators(frame)
        adx_series = indicators["adx"]
        atr_series = indicators["atr"]

        adx_value = _safe_last(adx_series, default=np.nan)
        atr_value = _safe_last(atr_series, default=np.nan)
        last_close = float(frame["Close"].iloc[-1])
        atr_pct = (
            atr_value / last_close
            if last_close > 0 and np.isfinite(atr_value)
            else np.nan
        )

        raw = self._classify(frame, adx_value, atr_pct)
        regime, bars = self._apply_hysteresis(key or "__global__", raw)

        reversal, reasons = self.reversal_model.score(
            frame,
            adx_series,
            atr_series,
            fast=self.ma_short,
            slow=self.ma_long,
            lookback=self.reversal_lookback,
        )

        if regime is not MarketRegime.BULL_TREND:
            reversal *= 0.5

        trend_strength = _clamp01(adx_value / 50.0) if np.isfinite(adx_value) else 0.0

        return RegimeAssessment(
            regime=regime,
            reversal_risk=reversal,
            trend_strength=trend_strength,
            bars_in_regime=bars,
            raw_regime=raw,
            reasons=reasons,
        )

    def detect(self, df: pd.DataFrame, key: str = None) -> MarketRegime:
        """Simplified interface for existing callers."""
        return self.assess(df, key).regime
