# ibkrpy/data/bar_sync.py
# Pure helpers for incremental bar downloads: request sizing, gap and split repair

import math
from typing import Optional

import numpy as np
import pandas as pd

NY = "America/New_York"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Two or more weekdays missing in a row is a download hole, not a holiday.
GAP_MISSING_WEEKDAYS = 2
# A bar-to-bar close ratio within SPLIT_TOLERANCE of a common split factor looks like an
# unadjusted split. Matching factors, not just size, keeps genuine big moves from
# triggering a refetch on every sync.
SPLIT_FACTORS = (1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0)
SPLIT_TOLERANCE = 0.05
SPLIT_LOOKBACK_BARS = 5
# Stored vs refetched closes differing by more than this means the provider re-adjusted.
ADJUSTMENT_TOLERANCE = 0.15


def duration_str(days: int, bar_size: str) -> str:
    """IB duration covering at least `days` calendar days (rounded up, never short)."""
    days = max(int(days), 1)
    if days >= 365:
        return f"{math.ceil(days / 365)} Y"
    if days >= 30 and bar_size != "1 day":
        return f"{math.ceil(days / 30)} M"
    return f"{days} D"


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """UTC index, unique sorted timestamps, capitalised OHLCV columns."""
    out = df.rename(columns={c.lower(): c for c in OHLCV}).copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.loc[:, [c for c in OHLCV if c in out.columns]]


def session_dates(index: pd.Index) -> pd.Index:
    """Trading date of each bar. Daily bars are stamped 00:00 UTC on their own date;
    intraday bars belong to their New York calendar date."""
    ts = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    if len(ts) and not daily_rows_off_midnight(ts).any():
        return pd.Index(ts.date)
    return pd.Index(ts.tz_convert(NY).date)


def find_gap_start(index: pd.Index, since: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Last bar before the earliest hole of ≥2 missing weekdays after `since`, or None."""
    if len(index) < 2:
        return None
    ts = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    days = pd.Series(session_dates(ts), index=ts)
    last_bar = days.groupby(days.values).apply(lambda s: s.index[-1])
    dates = np.array(sorted(last_bar.index), dtype="datetime64[D]")
    missing = np.busday_count(dates[:-1], dates[1:]) - 1
    for i in np.flatnonzero(missing >= GAP_MISSING_WEEKDAYS):
        last_before = last_bar.iloc[i]
        if last_before >= since:
            return last_before
    return None


def _is_split_factor(ratio: float) -> bool:
    r = ratio if ratio >= 1 else 1.0 / ratio
    return any(abs(r / f - 1.0) <= SPLIT_TOLERANCE for f in SPLIT_FACTORS)


def find_split_jump(close: pd.Series, since: pd.Timestamp) -> Optional[pd.Timestamp]:
    """A bar a few bars before the earliest split-like close jump after `since`, or None."""
    c = pd.to_numeric(close, errors="coerce")
    ratio = (c / c.shift(1)).dropna()
    ratio = ratio[(ratio.index >= since) & (ratio > 0)]
    hits = [ts for ts, r in ratio.items() if _is_split_factor(float(r))]
    if not hits:
        return None
    # Start a few bars early so the refetch overlaps enough pre-jump bars to measure
    # the adjustment ratio (adjustment_ratio uses the oldest shared bars).
    pos = c.index.get_loc(hits[0])
    return c.index[max(pos - SPLIT_LOOKBACK_BARS, 0)]


def adjustment_ratio(
    stored: pd.Series, fetched: pd.Series, n: int = 5
) -> Optional[float]:
    """Median stored/fetched close over the oldest `n` shared timestamps, or None."""
    shared = stored.index.intersection(fetched.index)[:n]
    if len(shared) == 0:
        return None
    r = (stored.loc[shared] / fetched.loc[shared]).replace([np.inf, -np.inf], np.nan)
    r = r.dropna()
    if r.empty:
        return None
    return float(r.median())


def needs_rescale(ratio: Optional[float]) -> bool:
    return ratio is not None and ratio > 0 and abs(ratio - 1.0) > ADJUSTMENT_TOLERANCE


def rescale_bars(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Bring stored bars onto the provider's current adjustment (prices ÷ ratio, volume × ratio)."""
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col] / ratio
    if "Volume" in out.columns:
        out["Volume"] = out["Volume"] * ratio
    return out


def daily_rows_off_midnight(index: pd.Index) -> np.ndarray:
    """Mask of rows in a "1 day" series whose UTC time is not 00:00 (intraday bars)."""
    ts = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return np.asarray((ts.hour != 0) | (ts.minute != 0) | (ts.second != 0), dtype=bool)
