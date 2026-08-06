# ibkrpy/data/ibkr_data_manager.py
# 將底層連接、歷史數據、實時訂閱與帳戶資訊統合於一處。

import asyncio
import time
import logging
from typing import Callable, Dict, Any, Optional, Tuple

import pandas as pd
from ib_insync import IB, Contract, Stock, util

logger = logging.getLogger("ibkrpy")

_BAR_SECONDS = {
    "1 secs": 1, "5 secs": 5, "10 secs": 10, "15 secs": 15, "30 secs": 30,
    "1 min": 60, "1 mins": 60, "2 mins": 120, "3 mins": 180, "5 mins": 300,
    "10 mins": 600, "15 mins": 900, "20 mins": 1200, "30 mins": 1800,
    "1 hour": 3600, "1 hours": 3600, "2 hours": 7200, "3 hours": 10800,
    "4 hours": 14400, "8 hours": 28800,
    "1 day": 86400, "1 week": 604800, "1 month": 2592000,
}


def bar_seconds(bar_size: str) -> int:
    return _BAR_SECONDS.get(str(bar_size).strip(), 300)


class IBKRDataManager:
    """統一處理所有與 IBKR 互動的數據請求 (具備機構級 API 頻率防護)"""

    # IBKR 官方 pacing 規則：
    #   - 10 分鐘內不得超過 60 次歷史請求
    #   - 15 秒內不得送出「完全相同」的歷史請求
    #   - 2 秒內同一 Contract+Exchange+TickType 不得超過 6 次
    HIST_WINDOW_SECONDS = 600
    HIST_MAX_IN_WINDOW = 50          # 對 60 留 10 次緩衝給臨時補資料
    IDENTICAL_COOLDOWN = 16.0        # 官方 15s，取 16s 留餘裕
    SAME_CONTRACT_WINDOW = 2.0
    SAME_CONTRACT_MAX = 4            # 官方 6 次，取 4 次
    MIN_REQUEST_GAP = 0.25           # 全域最小間隔，避免觸發「軟節流」

    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

        # 實時訂閱水位監控
        self._active_subscriptions: Dict[str, Dict[str, Any]] = {}
        self.max_subscriptions = 55

        # ---- 歷史數據速率控制 ----
        self._hist_req_timestamps: list = []            # 全域 10 分鐘窗
        self._identical_last: Dict[str, float] = {}     # 請求簽章 -> 上次送出時間
        self._contract_hits: Dict[str, list] = {}       # 合約鍵 -> 近 2 秒的送出時間
        self._last_request_at = 0.0
        self._hist_req_lock = asyncio.Lock()

        # ---- 結果快取 (請求簽章 -> (取得時間, DataFrame)) ----
        self._hist_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # ---- 合約快取：qualify 過的完整合約，絕不重建 ----
        self._contracts: Dict[str, Contract] = {}
        self._unresolvable: set = set()

        self._error_hook_installed = False

    # ------------------------------------------------------------------
    # 連線
    # ------------------------------------------------------------------

    async def connect(self):
        if self.ib.isConnected():
            return
        try:
            await self.ib.connectAsync(self.host, self.port, self.client_id)
            logger.info(f"IBKR 連線成功: {self.host}:{self.port} (Client ID: {self.client_id})")
            self._install_error_hook()
        except Exception as e:
            logger.warning(f"IBKR 連線失敗: {e}")
            raise ConnectionError(f"無法連線至 IBKR {self.host}:{self.port}: {e}") from e

    async def _ensure_connected(self) -> bool:
        """資料讀取路徑專用：連線失敗時降級為 False，不中斷整輪掃描"""
        try:
            await self.connect()
            return True
        except ConnectionError:
            return False

    def _install_error_hook(self):
        """
        ib_insync 的 reqHistoricalDataAsync 在 Error 162 時「不會拋例外」，
        而是回傳空 list 並把錯誤丟到 errorEvent。舊版因此只看得到 ib_insync
        印出的裸合約，完全不知道是哪檔標的、哪個週期、哪一段區間失敗。
        """
        if self._error_hook_installed:
            return
        self.ib.errorEvent += self._on_ib_error
        self._error_hook_installed = True

    def _symbol_of(self, contract) -> str:
        if contract is None:
            return "?"
        if getattr(contract, "symbol", ""):
            return contract.symbol
        con_id = getattr(contract, "conId", 0)
        for sym, c in self._contracts.items():
            if c.conId == con_id:
                return f"{sym}(conId={con_id})"
        return f"conId={con_id}"

    def _on_ib_error(self, reqId, errorCode, errorString, contract=None):
        # 2104/2106/2158 是連線正常的通知，2107/2119 是資料農場休眠，皆非錯誤
        if errorCode in (2104, 2106, 2107, 2108, 2119, 2158, 2100, 2150):
            logger.debug(f"[IBKR] ({errorCode}) {errorString}")
            return

        sym = self._symbol_of(contract)

        if errorCode == 162:
            msg = str(errorString)
            if "pacing violation" in msg.lower():
                logger.error(
                    f"[IBKR][{sym}] 🚦 觸發歷史資料 pacing violation。"
                    f"代表速率控制仍不夠保守，或 TWS 上有其他 client 共用同一組配額。原文: {msg}"
                )
            elif "no market data permissions" in msg.lower():
                logger.error(
                    f"[IBKR][{sym}] 🔒 缺少市場資料訂閱權限。請到 Account Management 開通對應交易所，"
                    f"或改用 whatToShow='ADJUSTED_LAST' / 延遲報價。原文: {msg}"
                )
            elif "unknown contract" in msg.lower():
                logger.error(
                    f"[IBKR][{sym}] ❓ 合約無法解析。多半是送出的合約欄位不完整 "
                    f"(缺 symbol / currency / primaryExchange)，或該 conId 已下市 / 更名。原文: {msg}"
                )
            elif "no historical market data" in msg.lower():
                logger.warning(f"[IBKR][{sym}] 該區間沒有歷史資料 (可能早於上市日)。原文: {msg}")
            else:
                logger.error(f"[IBKR][{sym}] 歷史資料服務錯誤 ({errorCode}): {msg}")
            return

        if errorCode in (200, 300, 321, 322, 366):
            logger.error(f"[IBKR][{sym}] 合約/請求錯誤 ({errorCode}): {errorString}")
            return

        if errorCode in (1100, 1101, 1102, 504):
            logger.warning(f"[IBKR] 連線狀態變化 ({errorCode}): {errorString}")
            return

        logger.warning(f"[IBKR][{sym}] ({errorCode}) {errorString}")

    # ------------------------------------------------------------------
    # 合約解析 (Error 162 的核心修正)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        """IBKR 的多股別代碼用空白分隔 (BRK B / BF B)，不是點或減號。"""
        return str(symbol).strip().upper().replace(".", " ").replace("-", " ")

    async def qualify(self, symbol: str, primary_exchange: str = None) -> Optional[Contract]:
        """
        取得「完整」合約並快取。回傳 None 代表該代碼無法解析，呼叫端應直接跳過。

        重點：qualifyContractsAsync 回填的是 conId + primaryExchange + tradingClass +
        currency 等一整組欄位，這一整組必須原封不動地拿去請求歷史資料。
        """
        key = self._normalise_symbol(symbol)
        if key in self._contracts:
            return self._contracts[key]
        if key in self._unresolvable:
            return None
        if not await self._ensure_connected():
            return None

        contract = Stock(key, "SMART", "USD")
        if primary_exchange:
            contract.primaryExchange = primary_exchange

        try:
            resolved = await self.ib.qualifyContractsAsync(contract)
        except Exception as e:
            logger.error(f"[{key}] 合約解析請求失敗: {e}")
            return None

        # qualifyContractsAsync 解析失敗時回傳空 list，且「不會」拋例外。
        if not resolved:
            details = []
            try:
                details = await self.ib.reqContractDetailsAsync(Stock(key, "", "USD"))
            except Exception:
                pass

            if not details:
                logger.error(
                    f"[{key}] ❌ IBKR 找不到此股票代碼。請確認：(a) 拼寫正確且未更名/下市；"
                    f"(b) 多股別代碼要用空白，例如 BRK B 而非 BRK.B；"
                    f"(c) 非美股需要另外指定交易所與幣別。此代碼本輪之後將被跳過。"
                )
                self._unresolvable.add(key)
                return None

            # 有多筆結果時，取美國主要上市所的那一筆
            preferred = ("NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "ISLAND")
            picked = None
            for d in details:
                if getattr(d.contract, "primaryExchange", "") in preferred:
                    picked = d.contract
                    break
            picked = picked or details[0].contract
            picked.exchange = "SMART"
            contract = picked
            logger.warning(
                f"[{key}] SMART 直接解析失敗，改由 contractDetails 選定 "
                f"conId={contract.conId} / primary={contract.primaryExchange}。"
            )

        if not contract.conId:
            logger.error(f"[{key}] ❌ 解析後仍無 conId，跳過此標的。")
            self._unresolvable.add(key)
            return None

        self._contracts[key] = contract
        logger.debug(
            f"[{key}] 合約已解析: conId={contract.conId}, "
            f"primary={contract.primaryExchange}, currency={contract.currency}"
        )
        return contract

    def get_cached_contract(self, symbol: str) -> Optional[Contract]:
        return self._contracts.get(self._normalise_symbol(symbol))

    # ------------------------------------------------------------------
    # 帳戶
    # ------------------------------------------------------------------

    async def get_net_liquidation(self, currency: str = "USD") -> float:
        if not await self._ensure_connected():
            return 0.0
        try:
            summary = await self.ib.accountSummaryAsync()
            for item in summary:
                if item.tag == 'NetLiquidationByCurrency' and item.currency == currency:
                    return float(item.value)
        except Exception as e:
            logger.warning(f"獲取帳戶淨值失敗: {e}")
        return 0.0

    # ------------------------------------------------------------------
    # 速率控制
    # ------------------------------------------------------------------

    @staticmethod
    def _signature(contract: Contract, end_datetime: str, duration: str,
                   bar_size: str, what_to_show: str, use_rth: bool) -> str:
        return "|".join([
            str(getattr(contract, "conId", 0) or getattr(contract, "symbol", "?")),
            end_datetime or "NOW",
            duration, bar_size, what_to_show, str(use_rth),
        ])

    @staticmethod
    def _contract_key(contract: Contract, what_to_show: str) -> str:
        return f"{getattr(contract, 'conId', 0)}|{getattr(contract, 'exchange', '')}|{what_to_show}"

    def _cache_ttl(self, bar_size: str, end_datetime: str) -> float:
        """
        指定了 endDateTime 的歷史區間是不變的，可以長期快取。
        滾動請求 (endDateTime='') 的有效期綁在 bar size 上：
        一根 1 小時 K 沒收完之前，重抓也只會拿到同一組已收 K 線。
        """
        if end_datetime:
            return 24 * 3600.0
        return min(max(bar_seconds(bar_size) * 0.5, 30.0), 3600.0)

    async def _acquire_hist_slot(self, signature: str, contract_key: str) -> float:
        """
        回傳需要等待的秒數 (0 代表可立即送出)。在鎖內完成所有帳務，
        呼叫端在鎖外 sleep，避免長時間持鎖擋住其他協程。
        """
        async with self._hist_req_lock:
            now = time.time()
            wait = 0.0

            # 規則 1：15 秒內不得重送相同請求
            last_same = self._identical_last.get(signature)
            if last_same is not None:
                gap = now - last_same
                if gap < self.IDENTICAL_COOLDOWN:
                    wait = max(wait, self.IDENTICAL_COOLDOWN - gap)

            # 規則 2：2 秒內同合約不得超過 N 次
            hits = [t for t in self._contract_hits.get(contract_key, [])
                    if now - t < self.SAME_CONTRACT_WINDOW]
            self._contract_hits[contract_key] = hits
            if len(hits) >= self.SAME_CONTRACT_MAX:
                wait = max(wait, self.SAME_CONTRACT_WINDOW - (now - hits[0]) + 0.05)

            # 規則 3：10 分鐘內不得超過 N 次
            self._hist_req_timestamps = [
                t for t in self._hist_req_timestamps if now - t < self.HIST_WINDOW_SECONDS
            ]
            if len(self._hist_req_timestamps) >= self.HIST_MAX_IN_WINDOW:
                oldest = self._hist_req_timestamps[0]
                wait = max(wait, self.HIST_WINDOW_SECONDS - (now - oldest) + 1.0)
                logger.warning(
                    f"🛡️ [API 防護] 10 分鐘窗內已用掉 {len(self._hist_req_timestamps)}/"
                    f"{self.HIST_MAX_IN_WINDOW} 次歷史請求，降溫 {wait:.1f} 秒。"
                    f"（快取命中率 {self.cache_hit_rate():.0%}，持續觸發代表輪詢頻率仍高於資料更新頻率）"
                )

            # 全域最小間隔
            gap_since_last = now - self._last_request_at
            if gap_since_last < self.MIN_REQUEST_GAP:
                wait = max(wait, self.MIN_REQUEST_GAP - gap_since_last)

            stamp = now + wait
            self._identical_last[signature] = stamp
            self._contract_hits.setdefault(contract_key, []).append(stamp)
            self._hist_req_timestamps.append(stamp)
            self._last_request_at = stamp
            return wait

    def cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return (self._cache_hits / total) if total else 0.0

    def pacing_stats(self) -> Dict[str, Any]:
        now = time.time()
        live = [t for t in self._hist_req_timestamps if now - t < self.HIST_WINDOW_SECONDS]
        return {
            "requests_in_window": len(live),
            "window_limit": self.HIST_MAX_IN_WINDOW,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self.cache_hit_rate(),
            "subscriptions": len(self._active_subscriptions),
        }

    # ------------------------------------------------------------------
    # 歷史資料
    # ------------------------------------------------------------------

    async def fetch_historical_data(
        self,
        contract: Contract,
        end_datetime: str = '',
        duration: str = '1 Y',
        bar_size: str = '1 day',
        what_to_show: str = 'TRADES',
        use_rth: bool = True,
        allow_cache: bool = True,
    ) -> pd.DataFrame:
        """
        獲取歷史 K 線 (受快取 + 三條 pacing 規則保護)。
        """
        if contract is None or not getattr(contract, "conId", 0):
            logger.error(
                f"[{getattr(contract, 'symbol', '?')}] 合約尚未解析 (conId 為空)，"
                f"拒絕送出歷史請求。請先呼叫 qualify()。"
            )
            return pd.DataFrame()

        if not await self._ensure_connected():
            return pd.DataFrame()

        signature = self._signature(contract, end_datetime, duration, bar_size, what_to_show, use_rth)
        ttl = self._cache_ttl(bar_size, end_datetime)

        if allow_cache:
            cached = self._hist_cache.get(signature)
            if cached and (time.time() - cached[0]) < ttl:
                self._cache_hits += 1
                logger.debug(
                    f"[{contract.symbol}] 快取命中 ({bar_size}/{duration})，"
                    f"未消耗 API 配額 (剩餘有效 {ttl - (time.time() - cached[0]):.0f}s)。"
                )
                return cached[1].copy()

        self._cache_misses += 1
        contract_key = self._contract_key(contract, what_to_show)
        wait = await self._acquire_hist_slot(signature, contract_key)
        if wait > 0:
            logger.debug(f"[{contract.symbol}] 速率控制等待 {wait:.2f}s ({bar_size}/{duration})")
            await asyncio.sleep(wait)

        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=2,
            )
        except Exception as e:
            logger.warning(
                f"[{contract.symbol}] 歷史請求發生例外 "
                f"({bar_size}/{duration}/end={end_datetime or 'NOW'}): {e}"
            )
            return pd.DataFrame()

        # ib_insync 在 Error 162 時回傳空 list 而非拋例外 —— 這裡明確記錄請求參數，
        # 讓日誌能直接對上 errorEvent 印出的那筆錯誤。
        if not bars:
            logger.warning(
                f"[{contract.symbol}] 歷史請求無資料返回 "
                f"(conId={contract.conId}, {bar_size}, {duration}, "
                f"end={end_datetime or 'NOW'}, {what_to_show}, RTH={use_rth})。"
                f"請對照上方 IBKR errorEvent 訊息判斷原因。"
            )
            return pd.DataFrame()

        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()

        df.set_index('date', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                           'close': 'Close', 'volume': 'Volume'}, inplace=True)

        if allow_cache:
            self._hist_cache[signature] = (time.time(), df.copy())
            if len(self._hist_cache) > 4000:
                oldest = sorted(self._hist_cache.items(), key=lambda kv: kv[1][0])[:1000]
                for k, _ in oldest:
                    self._hist_cache.pop(k, None)

        return df

    async def fetch_historical_by_symbol(self, symbol: str, **kwargs) -> pd.DataFrame:
        """便利入口：自動解析合約後抓資料。無法解析時回傳空 DataFrame。"""
        contract = await self.qualify(symbol)
        if contract is None:
            return pd.DataFrame()
        return await self.fetch_historical_data(contract, **kwargs)

    # ------------------------------------------------------------------
    # 實時訂閱
    # ------------------------------------------------------------------

    async def subscribe_realtime_bars(self, contract: Contract, bar_size: int = 5,
                                      callback: Callable = None):
        """訂閱實時 K 線 (5 秒 bar)。實時訂閱不佔用歷史請求配額。"""
        if not await self._ensure_connected():
            return
        if contract is None or not getattr(contract, "conId", 0):
            logger.warning("拒絕訂閱：合約尚未解析。")
            return

        sub_key = f"{contract.symbol}_{bar_size}"
        if sub_key in self._active_subscriptions:
            logger.debug(f"已存在 {sub_key} 的訂閱。")
            return

        if len(self._active_subscriptions) >= self.max_subscriptions:
            logger.warning(
                f"❌ [API 防護] 拒絕訂閱 {contract.symbol}："
                f"已達實時訂閱安全上限 ({self.max_subscriptions} 檔)。"
            )
            return

        try:
            bars = self.ib.reqRealTimeBars(contract, bar_size, 'TRADES', False)
            if callback:
                bars.updateEvent += callback
            self._active_subscriptions[sub_key] = {"bars": bars, "callback": callback}
            logger.debug(
                f"成功訂閱 {contract.symbol} 實時數據 "
                f"(水位: {len(self._active_subscriptions)}/{self.max_subscriptions})。"
            )
        except Exception as e:
            logger.warning(f"訂閱實時數據失敗: {e}")

    def cancel_realtime_subscription(self, contract: Contract, bar_size: int = 5):
        sub_key = f"{contract.symbol}_{bar_size}"
        sub_info = self._active_subscriptions.pop(sub_key, None)
        if sub_info:
            bars = sub_info["bars"]
            callback = sub_info["callback"]
            if callback:
                bars.updateEvent -= callback
            self.ib.cancelRealTimeBars(bars)
            logger.info(f"已取消 {sub_key} 的訂閱，釋放 API 額度。")