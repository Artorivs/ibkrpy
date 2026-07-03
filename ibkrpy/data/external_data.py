# ibkrpy/data/external_data.py
# 外部數據獲取器，整合 FMP (基本面/產業) 與 FRED (宏觀經濟) API 的輕量級模組

import aiohttp
import asyncio
import pandas as pd
from typing import Optional, Dict, Any
from fredapi import Fred

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
            # fredapi 本身是同步的，使用 to_thread 避免阻塞
            series = await asyncio.to_thread(self.fred.get_series, series_id)
            series.index = pd.to_datetime(series.index)
            return series
        except Exception as e:
            print(f"獲取 FRED 數據 ({series_id}) 失敗: {e}")
            return pd.Series(dtype=float)