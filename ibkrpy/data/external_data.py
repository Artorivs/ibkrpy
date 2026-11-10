# ibkrpy/data/external_data.py
# External data fetchers: FMP (fundamentals / sector) and FRED (macro)

import aiohttp
import asyncio
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from fredapi import Fred

from ibkrpy.shared.session_clock import NY

DEFAULT_VIX = 20.0


def bar_trading_days(index) -> pd.DatetimeIndex:
    """
    Trading day for each bar in the index (naive, midnight).

    The DB and IBKR (formatDate=2) use UTC. Daily bars are stored at UTC midnight of the
    trading day, so the UTC date is correct; intraday bars are converted to New York time
    first, or bars after 20:00 ET would land on the next day.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.normalize()
    utc = idx.tz_convert("UTC").tz_localize(None)
    local = idx.tz_convert(NY).tz_localize(None)
    is_daily_bar = utc == utc.normalize()
    return pd.DatetimeIndex(np.where(is_daily_bar, utc, local)).normalize()


def align_daily_series(
    series: Optional[pd.Series], index, default: float = DEFAULT_VIX
) -> pd.Series:
    """
    Align a daily series (e.g. FRED VIXCLS) to the bar index, using the previous trading day's value.

    Previous day only: the same-day close doesn't exist during the session (lookahead),
    and FRED publishes a day late, so live trading only ever sees the previous day.
    """
    if series is None or series.empty:
        return pd.Series(default, index=index, dtype=float)

    daily = series.dropna().copy()
    daily.index = pd.DatetimeIndex(daily.index)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily.index = daily.index.normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    if daily.empty:
        return pd.Series(default, index=index, dtype=float)

    days = bar_trading_days(index)
    pos = daily.index.searchsorted(days, side="left") - 1
    values = np.where(
        pos >= 0, daily.to_numpy(dtype=float)[np.clip(pos, 0, None)], np.nan
    )
    aligned = pd.Series(values, index=index, dtype=float).bfill()
    return aligned.fillna(default)


class ExternalDataFetcher:
    """Fetches data from external APIs (FMP, FRED)."""

    def __init__(self, fmp_api_key: str = None, fred_api_key: str = None):
        self.fmp_api_key = fmp_api_key
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"

    async def fetch_fmp_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch an FMP company profile (fundamentals / sector)."""
        if not self.fmp_api_key:
            print("FMP API key not configured.")
            return None

        url = f"{self.fmp_base_url}/profile/{symbol}?apikey={self.fmp_api_key}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, list) and len(data) > 0:
                            return data[0]
                    return None
        except Exception as e:
            print(f"Failed to fetch FMP data ({symbol}): {e}")
            return None

    async def fetch_fred_series(self, series_id: str) -> pd.Series:
        """Fetch a FRED series (e.g. VIX)."""
        if not self.fred:
            print("FRED API key not configured.")
            return pd.Series(dtype=float)

        try:
            series = await asyncio.to_thread(self.fred.get_series, series_id)
            series.index = pd.to_datetime(series.index)
            return series
        except Exception as e:
            print(f"Failed to fetch FRED data ({series_id}): {e}")
            return pd.Series(dtype=float)
