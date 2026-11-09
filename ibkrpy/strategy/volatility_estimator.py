# ibkrpy/strategy/volatility_estimator.py
# Per-bar realized volatility from the same bars, used to check or replace the model's σ.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ibkrpy.volatility")

_TRADING_DAYS = 252.0

_ABS_FLOOR_PER_BAR = 0.0005

_BAND_LO = 1.0 / 3.0
_BAND_HI = 3.0


@dataclass(frozen=True)
class VolatilityEstimate:
    """Full result of a volatility estimate, with source and diagnostics."""

    value: float
    source: str
    realized: float
    model: Optional[float]
    ratio: Optional[float]
    detail: str = ""

    @property
    def annualized(self) -> float:
        """Display only; the caller supplies bars_per_year."""
        return self.value * math.sqrt(_TRADING_DAYS)


def _close_to_close(
    close: pd.Series, lookback: int, halflife: Optional[float]
) -> float:
    """Std of log returns; EWMA when halflife is set (weights recent bars more)."""
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < 5:
        return float("nan")
    rets = rets.iloc[-lookback:]
    if halflife:
        weights = 0.5 ** (np.arange(len(rets))[::-1] / float(halflife))
        weights /= weights.sum()
        mean = float(np.sum(weights * rets.to_numpy()))
        var = float(np.sum(weights * (rets.to_numpy() - mean) ** 2))
        return math.sqrt(max(var, 0.0))
    return float(rets.std(ddof=1))


def _yang_zhang(df: pd.DataFrame, lookback: int) -> float:
    """
    Yang-Zhang estimator. Captures overnight gaps and intraday range, and is about 5-7x
    as efficient as close-to-close for the same sample, which matters with 60 bars.

    Needs Open/High/Low/Close; returns nan when columns or data are missing.
    """
    need = ("Open", "High", "Low", "Close")
    if not all(c in df.columns for c in need):
        return float("nan")

    win = df.iloc[-(lookback + 1) :]
    o = pd.to_numeric(win["Open"], errors="coerce")
    h = pd.to_numeric(win["High"], errors="coerce")
    l = pd.to_numeric(win["Low"], errors="coerce")
    c = pd.to_numeric(win["Close"], errors="coerce")

    prev_c = c.shift(1)
    frame = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "pc": prev_c}).dropna()
    frame = frame[(frame > 0).all(axis=1)]
    n = len(frame)
    if n < 5:
        return float("nan")

    overnight = np.log(frame["o"] / frame["pc"])
    open_close = np.log(frame["c"] / frame["o"])
    rs = np.log(frame["h"] / frame["c"]) * np.log(frame["h"] / frame["o"]) + np.log(
        frame["l"] / frame["c"]
    ) * np.log(frame["l"] / frame["o"])

    var_o = float(overnight.var(ddof=1))
    var_c = float(open_close.var(ddof=1))
    var_rs = float(rs.mean())

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = var_o + k * var_c + (1 - k) * var_rs
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return math.sqrt(var)


def realized_volatility(
    df: pd.DataFrame, lookback: int = 60, halflife: Optional[float] = 20.0
) -> float:
    """
    Realized volatility per bar (decimal), in the units of the bars given; never annualized.

    Yang-Zhang first; EWMA close-to-close when OHLC is missing or degenerate.
    Returns nan if both fail; VolatilityEstimator decides what to do.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return float("nan")

    yz = _yang_zhang(df, lookback)
    if np.isfinite(yz) and yz > 0:
        return yz

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    ctc = _close_to_close(close, lookback, halflife)
    return ctc if np.isfinite(ctc) and ctc > 0 else float("nan")


class VolatilityEstimator:
    """
    Chooses the σ handed to the strategy, using realized volatility as the reference.

    Contract: estimate() always returns a finite positive value and never raises.
    σ feeds the threshold policy, the exit geometry's noise floor and risk cap, regime
    detection and risk-parity sizing; a nan in any of them fails silently.
    """

    def __init__(
        self,
        lookback: int = 60,
        halflife: float = 20.0,
        band_lo: float = _BAND_LO,
        band_hi: float = _BAND_HI,
        abs_floor: float = _ABS_FLOOR_PER_BAR,
        trust_model: bool = True,
        distrust_symbols=None,
    ):
        self.lookback = int(lookback)
        self.halflife = float(halflife)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.abs_floor = float(abs_floor)
        self.trust_model = bool(trust_model)
        self.distrust = {str(x).upper() for x in (distrust_symbols or [])}
        self._warned: set = set()

    def estimate(
        self,
        symbol: str,
        df: pd.DataFrame,
        model_vol_per_bar: Optional[float] = None,
    ) -> VolatilityEstimate:
        """
        :param df:                the bars the strategy uses this tick (same units as σ)
        :param model_vol_per_bar: the model's per-bar volatility. Convert annual values
                                                            before passing; this method doesn't guess conventions.
        """
        realized = realized_volatility(df, self.lookback, self.halflife)

        if not np.isfinite(realized) or realized <= 0:
            fallback = max(self.abs_floor, 0.01)
            return VolatilityEstimate(
                fallback,
                "floor",
                float("nan"),
                model_vol_per_bar,
                None,
                "Too few bars for realized volatility; using the conservative floor",
            )

        realized = max(realized, self.abs_floor)

        if symbol.upper() in self.distrust:
            if symbol not in self._warned:
                self._warned.add(symbol)
                logger.info(
                    f"[{symbol}] In distrust_symbols; using realized volatility "
                    f"{realized * 100:.3f}%."
                )
            return VolatilityEstimate(
                realized,
                "realized",
                realized,
                model_vol_per_bar,
                None,
                "symbol is in distrust_symbols",
            )

        if not self.trust_model or model_vol_per_bar is None:
            return VolatilityEstimate(
                realized,
                "realized",
                realized,
                model_vol_per_bar,
                None,
                "using realized volatility",
            )

        model = float(model_vol_per_bar)
        if not np.isfinite(model) or model <= 0:
            return VolatilityEstimate(
                realized,
                "realized",
                realized,
                model,
                None,
                "model value not finite or not positive",
            )

        ratio = model / realized
        if self.band_lo <= ratio <= self.band_hi:
            return VolatilityEstimate(
                model,
                "model",
                realized,
                model,
                ratio,
                f"model value within band (×{ratio:.2f})",
            )

        if symbol not in self._warned:
            self._warned.add(symbol)
            logger.error(
                f"🌡️ [{symbol}] Model volatility {model * 100:.3f}% differs from realized "
                f"{realized * 100:.3f}% by {ratio:.2f}x; using realized."
                f"The GARCH weights for this symbol are probably mis-scaled; retrain."
            )
        return VolatilityEstimate(
            realized,
            "realized",
            realized,
            model,
            ratio,
            f"model value off by {ratio:.2f}x, outside [{self.band_lo:.2f}, {self.band_hi:.2f}]",
        )


def build_volatility_estimator(config) -> VolatilityEstimator:
    """Composition root."""
    s = (config.get("volatility_settings") or {}) if config else {}
    return VolatilityEstimator(
        lookback=int(s.get("lookback", 60)),
        halflife=float(s.get("ewma_halflife", 20.0)),
        band_lo=float(s.get("model_band_lo", _BAND_LO)),
        band_hi=float(s.get("model_band_hi", _BAND_HI)),
        abs_floor=float(s.get("abs_floor_per_bar", _ABS_FLOOR_PER_BAR)),
        trust_model=bool(s.get("trust_model", True)),
        distrust_symbols=s.get("distrust_symbols") or [],
    )
