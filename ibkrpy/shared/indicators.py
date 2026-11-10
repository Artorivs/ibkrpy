# ibkrpy/data/indicators.py
# Technical indicators (replaces pandas_ta 0.3.14b)

from typing import Optional

import numpy as np
import pandas as pd

_EPSILON = np.finfo(float).eps


def _non_zero_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """high - low, plus epsilon everywhere if any range is 0 (as pandas_ta does)."""
    diff = high - low
    if diff.eq(0).any():
        diff = diff + _EPSILON
    return diff


def rma(close: pd.Series, length: int) -> pd.Series:
    """Wilder moving average. pandas_ta 0.3.14b uses adjust=True (the pandas default)."""
    return close.ewm(alpha=1.0 / length, min_periods=length).mean()


def sma(close: pd.Series, length: int) -> pd.Series:
    return close.rolling(length, min_periods=length).mean().rename(f"SMA_{length}")


def ema(close: pd.Series, length: int) -> pd.Series:
    """EMA seeded with the SMA of the first `length` bars, then span=length, adjust=False."""
    if len(close) < length:
        # pandas_ta returns None here; all-NaN lets the later dropna remove it
        return pd.Series(np.nan, index=close.index, name=f"EMA_{length}")
    seeded = close.astype(float).copy()
    seeded.iloc[: length - 1] = np.nan
    seeded.iloc[length - 1] = close.iloc[:length].mean()
    return seeded.ewm(span=length, adjust=False).mean().rename(f"EMA_{length}")


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    change = close.diff(1)
    gain = change.clip(lower=0)
    loss = change.clip(upper=0)
    gain_avg = rma(gain, length)
    loss_avg = rma(loss, length)
    return (100.0 * gain_avg / (gain_avg + loss_avg.abs())).rename(f"RSI_{length}")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [_non_zero_range(high, low), high - prev_close, prev_close - low], axis=1
    )
    tr = ranges.abs().max(axis=1)
    tr.iloc[:1] = np.nan
    return tr


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    return rma(true_range(high, low, close), length).rename(f"ATRr_{length}")


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    if slow < fast:
        fast, slow = slow, fast
    line = ema(close, fast) - ema(close, slow)
    first = line.first_valid_index()
    signal_line = (
        ema(line.loc[first:], signal).reindex(line.index)
        if first is not None
        else pd.Series(np.nan, index=line.index)
    )
    suffix = f"{fast}_{slow}_{signal}"
    return pd.DataFrame(
        {
            f"MACD_{suffix}": line,
            f"MACDh_{suffix}": line - signal_line,
            f"MACDs_{suffix}": signal_line,
        },
        index=close.index,
    )


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands with population std (ddof=0), as in pandas_ta."""
    std = float(std)
    mid = close.rolling(length, min_periods=length).mean()
    deviation = std * close.rolling(length, min_periods=length).std(ddof=0)
    lower = mid - deviation
    upper = mid + deviation
    width = _non_zero_range(upper, lower)
    suffix = f"{length}_{std}"
    return pd.DataFrame(
        {
            f"BBL_{suffix}": lower,
            f"BBM_{suffix}": mid,
            f"BBU_{suffix}": upper,
            f"BBB_{suffix}": 100 * width / mid,
            f"BBP_{suffix}": _non_zero_range(close, lower) / width,
        },
        index=close.index,
    )


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.DataFrame:
    """Returns ADX_n, DMP_n, DMN_n (ADX first), as in pandas_ta."""
    atr_ = atr(high, low, close, length)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    plus_dm = plus_dm.mask(plus_dm.abs() < _EPSILON, 0.0)
    minus_dm = minus_dm.mask(minus_dm.abs() < _EPSILON, 0.0)

    k = 100.0 / atr_
    dmp = k * rma(plus_dm, length)
    dmn = k * rma(minus_dm, length)
    dx = 100.0 * (dmp - dmn).abs() / (dmp + dmn)
    return pd.DataFrame(
        {f"ADX_{length}": rma(dx, length), f"DMP_{length}": dmp, f"DMN_{length}": dmn},
        index=close.index,
    )


def _enough(series: pd.Series, length: int) -> bool:
    """pandas_ta returns None when there are fewer than `length` rows."""
    return series is not None and len(series) >= length


def add_default_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the default model indicators, with column names matching pandas_ta 0.3.14b
    df.ta.<indicator>(append=True):

      SMA_10, EMA_20, RSI_14, ATRr_14,
      MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9,
      BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    """
    out = df.copy()
    high, low, close = out["High"], out["Low"], out["Close"]

    parts: list[Optional[pd.Series | pd.DataFrame]] = []
    if _enough(close, 10):
        parts.append(sma(close, 10))
    if _enough(close, 20):
        parts.append(ema(close, 20))
    if _enough(close, 14):
        parts.append(rsi(close, 14))
        parts.append(atr(high, low, close, 14))
    if _enough(close, 26):
        parts.append(macd(close, 12, 26, 9))
    if _enough(close, 20):
        parts.append(bbands(close, 20, 2.0))

    for part in parts:
        if isinstance(part, pd.Series):
            out[part.name] = part
        else:
            for col in part.columns:
                out[col] = part[col]
    return out
