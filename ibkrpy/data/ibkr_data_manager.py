# ibkrpy/data/ibkr_data_manager.py
# Connection, historical data, real-time subscriptions and account info in one place.

import asyncio
import time
import logging
from typing import Callable, Dict, Any, Optional, Tuple

import pandas as pd
from ib_insync import IB, Contract, Stock, util

logger = logging.getLogger("ibkrpy")

_BAR_SECONDS = {
    "1 secs": 1,
    "5 secs": 5,
    "10 secs": 10,
    "15 secs": 15,
    "30 secs": 30,
    "1 min": 60,
    "1 mins": 60,
    "2 mins": 120,
    "3 mins": 180,
    "5 mins": 300,
    "10 mins": 600,
    "15 mins": 900,
    "20 mins": 1200,
    "30 mins": 1800,
    "1 hour": 3600,
    "1 hours": 3600,
    "2 hours": 7200,
    "3 hours": 10800,
    "4 hours": 14400,
    "8 hours": 28800,
    "1 day": 86400,
    "1 week": 604800,
    "1 month": 2592000,
}


def bar_seconds(bar_size: str) -> int:
    return _BAR_SECONDS.get(str(bar_size).strip(), 300)


class IBKRDataManager:
    """All IBKR data requests, with API rate-limit protection."""

    HIST_WINDOW_SECONDS = 600
    HIST_MAX_IN_WINDOW = 50
    IDENTICAL_COOLDOWN = 16.0
    SAME_CONTRACT_WINDOW = 2.0
    SAME_CONTRACT_MAX = 4
    MIN_REQUEST_GAP = 0.25

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

        self._active_subscriptions: Dict[str, Dict[str, Any]] = {}
        self.max_subscriptions = 55

        self._hist_req_timestamps: list = []
        self._identical_last: Dict[str, float] = {}
        self._contract_hits: Dict[str, list] = {}
        self._last_request_at = 0.0
        self._hist_req_lock = asyncio.Lock()

        self._hist_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self._contracts: Dict[str, Contract] = {}
        self._unresolvable: set = set()

        self._error_hook_installed = False

    async def connect(self):
        if self.ib.isConnected():
            return
        try:
            await self.ib.connectAsync(self.host, self.port, self.client_id)
            logger.info(
                f"IBKR connected: {self.host}:{self.port} (Client ID: {self.client_id})"
            )
            self._install_error_hook()
        except Exception as e:
            logger.warning(f"IBKR connection failed: {e}")
            raise ConnectionError(
                f"Cannot connect to IBKR {self.host}:{self.port}: {e}"
            ) from e

    async def _ensure_connected(self) -> bool:
        """For data reads: returns False on connection failure instead of aborting the scan."""
        try:
            await self.connect()
            return True
        except ConnectionError:
            return False

    def _install_error_hook(self):
        """
        On Error 162, reqHistoricalDataAsync does not raise; it returns an empty list and
        reports through errorEvent. This hook logs which symbol, bar size and range failed.
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
        if errorCode in (2104, 2106, 2107, 2108, 2119, 2158, 2100, 2150):
            logger.debug(f"[IBKR] ({errorCode}) {errorString}")
            return

        sym = self._symbol_of(contract)

        if errorCode == 162:
            msg = str(errorString)
            if "pacing violation" in msg.lower():
                logger.error(
                    f"[IBKR][{sym}] 🚦 Historical data pacing violation. "
                    f"Rate limiting is not conservative enough, or another client shares the quota. Original: {msg}"
                )
            elif "no market data permissions" in msg.lower():
                logger.error(
                    f"[IBKR][{sym}] 🔒 Missing market data permissions. Subscribe to the exchange in Account Management, "
                    f"or use whatToShow='ADJUSTED_LAST' / delayed data. Original: {msg}"
                )
            elif "unknown contract" in msg.lower():
                logger.error(
                    f"[IBKR][{sym}] ❓ Contract cannot be resolved. Usually incomplete contract fields "
                    f"(symbol / currency / primaryExchange), or the conId was delisted / renamed. Original: {msg}"
                )
            elif "no historical market data" in msg.lower():
                logger.warning(
                    f"[IBKR][{sym}] No historical data for this range (possibly before listing). Original: {msg}"
                )
            else:
                logger.error(
                    f"[IBKR][{sym}] Historical data service error ({errorCode}): {msg}"
                )
            return

        if errorCode in (200, 300, 321, 322, 366):
            logger.error(
                f"[IBKR][{sym}] Contract/request error ({errorCode}): {errorString}"
            )
            return

        if errorCode in (1100, 1101, 1102, 504):
            logger.warning(
                f"[IBKR] Connection state change ({errorCode}): {errorString}"
            )
            return

        logger.warning(f"[IBKR][{sym}] ({errorCode}) {errorString}")

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        """IBKR separates share classes with a space (BRK B / BF B), not a dot or dash."""
        return str(symbol).strip().upper().replace(".", " ").replace("-", " ")

    async def qualify(
        self, symbol: str, primary_exchange: str = None
    ) -> Optional[Contract]:
        """
        Resolve the full contract and cache it. Returns None if the symbol cannot be resolved.

        qualifyContractsAsync fills conId, primaryExchange, tradingClass, currency etc.; that
        whole set must be passed unchanged to historical data requests.
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
            logger.error(f"[{key}] Contract resolution request failed: {e}")
            return None

        if not resolved:
            details = []
            try:
                details = await self.ib.reqContractDetailsAsync(Stock(key, "", "USD"))
            except Exception:
                pass

            if not details:
                logger.error(
                    f"[{key}] ❌ IBKR does not know this symbol. Check that (a) it is spelled right and not renamed/delisted; "
                    f"(b) share classes use a space, e.g. BRK B not BRK.B; "
                    f"(c) non-US stocks specify exchange and currency. Skipping this symbol from now on."
                )
                self._unresolvable.add(key)
                return None

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
                f"[{key}] SMART resolution failed; contractDetails picked "
                f"conId={contract.conId} / primary={contract.primaryExchange}."
            )

        if not contract.conId:
            logger.error(f"[{key}] ❌ Still no conId after resolution; skipping.")
            self._unresolvable.add(key)
            return None

        self._contracts[key] = contract
        logger.debug(
            f"[{key}] Contract resolved: conId={contract.conId}, "
            f"primary={contract.primaryExchange}, currency={contract.currency}"
        )
        return contract

    def get_cached_contract(self, symbol: str) -> Optional[Contract]:
        return self._contracts.get(self._normalise_symbol(symbol))

    async def get_net_liquidation(self, currency: str = "USD") -> float:
        if not await self._ensure_connected():
            return 0.0
        try:
            summary = await self.ib.accountSummaryAsync()
            for item in summary:
                if item.tag == "NetLiquidationByCurrency" and item.currency == currency:
                    return float(item.value)
        except Exception as e:
            logger.warning(f"Failed to get account net liquidation: {e}")
        return 0.0

    @staticmethod
    def _signature(
        contract: Contract,
        end_datetime: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ) -> str:
        return "|".join(
            [
                str(getattr(contract, "conId", 0) or getattr(contract, "symbol", "?")),
                end_datetime or "NOW",
                duration,
                bar_size,
                what_to_show,
                str(use_rth),
            ]
        )

    @staticmethod
    def _contract_key(contract: Contract, what_to_show: str) -> str:
        return f"{getattr(contract, 'conId', 0)}|{getattr(contract, 'exchange', '')}|{what_to_show}"

    def _cache_ttl(self, bar_size: str, end_datetime: str) -> float:
        """
        A range with an explicit endDateTime never changes and can be cached long. A rolling
        request (endDateTime='') expires with the bar size: until a bar closes, refetching
        returns the same closed bars.
        """
        if end_datetime:
            return 24 * 3600.0
        return min(max(bar_seconds(bar_size) * 0.5, 30.0), 3600.0)

    async def _acquire_hist_slot(self, signature: str, contract_key: str) -> float:
        """
        Seconds to wait (0 = send now). All bookkeeping happens under the lock; the caller
        sleeps outside it so other coroutines are not blocked.
        """
        async with self._hist_req_lock:
            now = time.time()
            wait = 0.0

            last_same = self._identical_last.get(signature)
            if last_same is not None:
                gap = now - last_same
                if gap < self.IDENTICAL_COOLDOWN:
                    wait = max(wait, self.IDENTICAL_COOLDOWN - gap)

            hits = [
                t
                for t in self._contract_hits.get(contract_key, [])
                if now - t < self.SAME_CONTRACT_WINDOW
            ]
            self._contract_hits[contract_key] = hits
            if len(hits) >= self.SAME_CONTRACT_MAX:
                wait = max(wait, self.SAME_CONTRACT_WINDOW - (now - hits[0]) + 0.05)

            self._hist_req_timestamps = [
                t
                for t in self._hist_req_timestamps
                if now - t < self.HIST_WINDOW_SECONDS
            ]
            if len(self._hist_req_timestamps) >= self.HIST_MAX_IN_WINDOW:
                oldest = self._hist_req_timestamps[0]
                wait = max(wait, self.HIST_WINDOW_SECONDS - (now - oldest) + 1.0)
                logger.warning(
                    f"🛡️ [API guard] {len(self._hist_req_timestamps)}/{self.HIST_MAX_IN_WINDOW} "
                    f"historical requests used in the 10-minute window; cooling down {wait:.1f}s "
                    f"(cache hit rate {self.cache_hit_rate():.0%}; repeated triggers mean polling is faster than data updates)"
                )

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
        live = [
            t for t in self._hist_req_timestamps if now - t < self.HIST_WINDOW_SECONDS
        ]
        return {
            "requests_in_window": len(live),
            "window_limit": self.HIST_MAX_IN_WINDOW,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self.cache_hit_rate(),
            "subscriptions": len(self._active_subscriptions),
        }

    async def fetch_historical_data(
        self,
        contract: Contract,
        end_datetime: str = "",
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        allow_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical bars (protected by the cache and three pacing rules).
        """
        if contract is None or not getattr(contract, "conId", 0):
            logger.error(
                f"[{getattr(contract, 'symbol', '?')}] Contract not resolved (empty conId); "
                f"refusing the historical request. Call qualify() first."
            )
            return pd.DataFrame()

        if not await self._ensure_connected():
            return pd.DataFrame()

        signature = self._signature(
            contract, end_datetime, duration, bar_size, what_to_show, use_rth
        )
        ttl = self._cache_ttl(bar_size, end_datetime)

        if allow_cache:
            cached = self._hist_cache.get(signature)
            if cached and (time.time() - cached[0]) < ttl:
                self._cache_hits += 1
                logger.debug(
                    f"[{contract.symbol}] Cache hit ({bar_size}/{duration}); "
                    f"no API quota used ({ttl - (time.time() - cached[0]):.0f}s left)."
                )
                return cached[1].copy()

        self._cache_misses += 1
        contract_key = self._contract_key(contract, what_to_show)
        wait = await self._acquire_hist_slot(signature, contract_key)
        if wait > 0:
            logger.debug(
                f"[{contract.symbol}] Rate limit wait {wait:.2f}s ({bar_size}/{duration})"
            )
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
                f"[{contract.symbol}] Historical request raised "
                f"({bar_size}/{duration}/end={end_datetime or 'NOW'}): {e}"
            )
            return pd.DataFrame()

        if not bars:
            logger.warning(
                f"[{contract.symbol}] Historical request returned no data "
                f"(conId={contract.conId}, {bar_size}, {duration}, "
                f"end={end_datetime or 'NOW'}, {what_to_show}, RTH={use_rth}). "
                f"See the IBKR errorEvent message above for the cause."
            )
            return pd.DataFrame()

        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.set_index("date").rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "average": "Average",
            },
        )

        if allow_cache:
            self._hist_cache[signature] = (time.time(), df.copy())
            if len(self._hist_cache) > 4000:
                oldest = sorted(self._hist_cache.items(), key=lambda kv: kv[1][0])[
                    :1000
                ]
                for k, _ in oldest:
                    self._hist_cache.pop(k, None)

        return df

    async def fetch_historical_by_symbol(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Convenience entry: resolve the contract, then fetch. Returns an empty DataFrame if unresolved."""
        contract = await self.qualify(symbol)
        if contract is None:
            return pd.DataFrame()
        return await self.fetch_historical_data(contract, **kwargs)

    async def subscribe_realtime_bars(
        self, contract: Contract, bar_size: int = 5, callback: Callable = None
    ):
        """Subscribe to 5-second real-time bars. These do not use the historical request quota."""
        if not await self._ensure_connected():
            return
        if contract is None or not getattr(contract, "conId", 0):
            logger.warning("Subscription refused: contract not resolved.")
            return

        sub_key = f"{contract.symbol}_{bar_size}"
        if sub_key in self._active_subscriptions:
            logger.debug(f"Subscription {sub_key} already exists.")
            return

        if len(self._active_subscriptions) >= self.max_subscriptions:
            logger.warning(
                f"❌ [API guard] Subscription to {contract.symbol} refused: "
                f"real-time subscription limit reached ({self.max_subscriptions})."
            )
            return

        try:
            bars = self.ib.reqRealTimeBars(contract, bar_size, "TRADES", False)
            if callback:
                bars.updateEvent += callback
            self._active_subscriptions[sub_key] = {"bars": bars, "callback": callback}
            logger.debug(
                f"Subscribed to {contract.symbol} real-time data "
                f"(usage: {len(self._active_subscriptions)}/{self.max_subscriptions})."
            )
        except Exception as e:
            logger.warning(f"Real-time subscription failed: {e}")

    def cancel_realtime_subscription(self, contract: Contract, bar_size: int = 5):
        sub_key = f"{contract.symbol}_{bar_size}"
        sub_info = self._active_subscriptions.pop(sub_key, None)
        if sub_info:
            bars = sub_info["bars"]
            callback = sub_info["callback"]
            if callback:
                bars.updateEvent -= callback
            self.ib.cancelRealTimeBars(bars)
            logger.info(f"Cancelled subscription {sub_key}, releasing API quota.")
