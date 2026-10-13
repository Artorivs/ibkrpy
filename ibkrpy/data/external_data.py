# ibkrpy/data/external_data.py
# 外部數據獲取器，整合 FMP (基本面/產業) 與 FRED (宏觀經濟) API 的輕量級模組

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
    K 線索引對應的交易日 (naive, 午夜)。

    資料庫與 IBKR (formatDate=2) 的時間都是 UTC。日 K 存成「交易日的 UTC 午夜」，
    直接取 UTC 日期即可；盤中 K 要換成紐約時間再取日期，否則 20:00 ET 之後的
    K 線會被算到隔天。
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
    把日頻序列 (例如 FRED VIXCLS) 對齊到 K 線索引，每根 K 線取「前一個交易日」的值。

    [修正] 舊版直接以帶時區的 K 線索引去 map 無時區的 FRED 日期，
    永遠對不上，整欄 NaN 後退回常數 20.0 —— 訓練與實盤看到的 VIX 從來不是真的。

    只取前一日的值有兩個理由: 當日收盤的 VIX 在盤中尚未產生 (避免前視偏差)，
    而 FRED 也要隔日才公布，實盤本來就只拿得到前一日 —— 訓練與實盤因此一致。
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
    """統合外部 API (FMP, FRED) 的數據獲取"""

    def __init__(self, fmp_api_key: str = None, fred_api_key: str = None):
        self.fmp_api_key = fmp_api_key
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"

    async def fetch_fmp_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """獲取 FMP 公司基本面/產業配置"""
        if not self.fmp_api_key:
            print("未配置 FMP API Key。")
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
            print(f"獲取 FMP 數據 ({symbol}) 失敗: {e}")
            return None

    async def fetch_fred_series(self, series_id: str) -> pd.Series:
        """獲取 FRED 宏觀經濟指標 (例如 VIX)"""
        if not self.fred:
            print("未配置 FRED API Key。")
            return pd.Series(dtype=float)

        try:
            series = await asyncio.to_thread(self.fred.get_series, series_id)
            series.index = pd.to_datetime(series.index)
            return series
        except Exception as e:
            print(f"獲取 FRED 數據 ({series_id}) 失敗: {e}")
            return pd.Series(dtype=float)
