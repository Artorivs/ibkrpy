# ibkrpy/data/ibkr_data_manager.py
# 將底層連接、歷史數據、實時訂閱與帳戶資訊統合於一處，去除過度碎片化的繼承。

import asyncio
import time
import pandas as pd
from typing import Callable, Dict, Any, Optional
from ib_insync import IB, Contract, util

class IBKRDataManager:
    """統一處理所有與 IBKR 互動的數據請求 (具備機構級 API 頻率防護)"""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # 實時訂閱水位監控
        self._active_subscriptions = {}
        self.max_subscriptions = 55  # 刻意設定在 60 以下，預留安全緩衝
        
        # 歷史數據頻率限制器 (防護 IBKR Pacing Violation: 10 分鐘內最多 60 次)
        self._hist_req_timestamps = []
        self._hist_req_lock = asyncio.Lock()
        
    async def connect(self):
        """建立或恢復與 IBKR 的連線"""
        if not self.ib.isConnected():
            try:
                await self.ib.connectAsync(self.host, self.port, self.client_id)
                print(f"IBKR 連線成功: {self.host}:{self.port} (Client ID: {self.client_id})")
            except Exception as e:
                print(f"IBKR 連線失敗: {e}")
                
    async def get_net_liquidation(self, currency: str = "USD") -> float:
        """獲取帳戶淨值"""
        await self.connect()
        try:
            summary = await self.ib.accountSummaryAsync()
            for item in summary:
                if item.tag == 'NetLiquidationByCurrency' and item.currency == currency:
                    return float(item.value)
        except Exception as e:
            print(f"獲取帳戶淨值失敗: {e}")
        return 0.0

    async def _rate_limit_historical(self):
        """非阻塞速率限制器：動態調節請求頻率"""
        async with self._hist_req_lock:
            now = time.time()
            # 清除超過 10 分鐘 (600秒) 的歷史請求紀錄
            self._hist_req_timestamps = [t for t in self._hist_req_timestamps if now - t < 600]
            
            # 若 10 分鐘內請求次數已達 55 次，觸發主動休眠機制
            if len(self._hist_req_timestamps) >= 55:
                # 計算需要等待多久才能釋放最舊的一個請求額度
                sleep_time = 600 - (now - self._hist_req_timestamps[0]) + 1
                print(f"🛡️ [API 防護] 歷史數據請求逼近券商上限，系統自動進入降溫休眠 {sleep_time:.1f} 秒...")
                await asyncio.sleep(sleep_time)
                
            self._hist_req_timestamps.append(time.time())

    async def fetch_historical_data(
        self, 
        contract: Contract, 
        end_datetime: str = '', 
        duration: str = '1 Y', 
        bar_size: str = '1 day',
        what_to_show: str = 'TRADES'
    ) -> pd.DataFrame:
        """獲取歷史 K 線數據 (受速率限制器保護)"""
        await self.connect()
        await self._rate_limit_historical()  # 執行 API 前先經過限速器檢查
        
        if contract.conId:
            query_contract = Contract(conId=contract.conId, exchange=contract.exchange or 'SMART')
        else:
            query_contract = contract

        try:
            bars = await self.ib.reqHistoricalDataAsync(
                query_contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=2
            )
            if not bars:
                return pd.DataFrame()
                
            df = util.df(bars)
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            print(f"獲取 {contract.symbol if contract.symbol else contract.conId} 歷史數據失敗: {e}")
            return pd.DataFrame()

    async def subscribe_realtime_bars(
        self, 
        contract: Contract, 
        bar_size: int = 5, 
        callback: Callable = None
    ):
        """訂閱實時 K 線數據 (受水位限制器保護)"""
        await self.connect()
        sub_key = f"{contract.symbol}_{bar_size}"
        
        if sub_key in self._active_subscriptions:
            print(f"已存在 {sub_key} 的訂閱。")
            return
            
        # 嚴格監控訂閱水位
        if len(self._active_subscriptions) >= self.max_subscriptions:
            print(f"❌ [API 防護] 拒絕訂閱 {contract.symbol}：已達實時訂閱安全上限 ({self.max_subscriptions} 檔)。")
            return
            
        try:
            query_contract = Contract(conId=contract.conId, exchange=contract.exchange or 'SMART') if contract.conId else contract
            bars = self.ib.reqRealTimeBars(query_contract, bar_size, 'TRADES', False)
            if callback:
                bars.updateEvent += callback
                
            self._active_subscriptions[sub_key] = {"bars": bars, "callback": callback}
            print(f"成功訂閱 {contract.symbol} 的實時數據 (目前水位: {len(self._active_subscriptions)}/{self.max_subscriptions})。")
        except Exception as e:
            print(f"訂閱實時數據失敗: {e}")

    def cancel_realtime_subscription(self, contract: Contract, bar_size: int = 5):
        """取消實時數據訂閱並釋放額度"""
        sub_key = f"{contract.symbol}_{bar_size}"
        sub_info = self._active_subscriptions.pop(sub_key, None)
        if sub_info:
            bars = sub_info["bars"]
            callback = sub_info["callback"]
            if callback:
                bars.updateEvent -= callback
            self.ib.cancelRealTimeBars(bars)
            print(f"已取消 {sub_key} 的訂閱，釋放 API 額度。")