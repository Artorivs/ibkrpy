# ibkrpy/core/system_daemon.py
# 24/7 daemon: trades during regular hours and retrains when the market is closed

import asyncio
import datetime

from ibkrpy.shared.system_log import global_logger
from ibkrpy.shared.session_clock import NY, Session
from ibkrpy.manager.trading_engine import TradingEngine
from ibkrpy.data.ibkr_data_manager import IBKRDataManager
from ibkrpy.core.retrain_scheduler import RetrainScheduler

OFF_MARKET_POLL_SECONDS = 300
RETRAIN_BUSY_POLL_SECONDS = 60


class SystemDaemon:
    """Runs the day/night cycle: trade in regular hours, hand off to RetrainScheduler when closed."""

    def __init__(
        self,
        ib_manager: IBKRDataManager,
        trading_engine: TradingEngine,
        symbols: list,
        retrain_scheduler: RetrainScheduler = None,
    ):
        self.ib_manager = ib_manager
        self.engine = trading_engine
        self.symbols = list(symbols)
        self.logger = global_logger
        self.tick_interval_minutes = 5

        self.retrain = retrain_scheduler or RetrainScheduler(
            trading_engine.config, on_success=self._on_retrained
        )

        self._scan_offset = 0
        self._reconnect_failures = 0

    def _on_retrained(self):
        self.engine.models.invalidate()
        self.logger.info("Model cache cleared; new weights load on the next loop.")

    def _is_market_open(self, now_ny: datetime.datetime) -> bool:
        """Regular trading hours. Holidays and early closes come from SessionClock."""
        return self.engine.session_clock.classify(now_ny) is Session.RTH

    async def _handle_reconnect(self):
        """Recover from IBKR disconnects during 24/7 operation."""
        if self.ib_manager.ib.isConnected():
            return
        self.logger.warning("IBKR disconnected; reconnecting...")
        try:
            await self.ib_manager.connect()
            if self._reconnect_failures:
                self.logger.info(
                    f"IBKR reconnected (after {self._reconnect_failures} failed attempts)."
                )
            else:
                self.logger.info("IBKR reconnected.")
            self._reconnect_failures = 0
        except Exception as e:
            self._reconnect_failures += 1
            msg = f"Reconnect failed (attempt {self._reconnect_failures}): {e}; retrying next loop."
            if self._reconnect_failures >= 3:
                self.logger.error(msg)
            else:
                self.logger.warning(msg)

    def _scan_order(self):
        """Rotate the scan starting point."""
        if not self.symbols:
            return []
        n = len(self.symbols)
        self._scan_offset = (self._scan_offset + 1) % n
        return self.symbols[self._scan_offset :] + self.symbols[: self._scan_offset]

    async def _run_market_cycle(self, now_ny: datetime.datetime):
        self.logger.info("=" * 40)
        self.logger.info(
            f"📈 Market open (NY {now_ny:%H:%M}): running scan and decisions"
        )
        self.logger.info("=" * 40)

        if self.ib_manager.ib.isConnected():
            await self.engine.update_system_state()
            for symbol in self._scan_order():
                await self.engine.run_tick(symbol)
                await asyncio.sleep(2)

        self.logger.info(
            f"✅ Scan done. Waiting for the next {self.tick_interval_minutes}-minute bar..."
        )
        await asyncio.sleep(self.tick_interval_minutes * 60)

    async def run_24_7(self):
        """24-hour main loop."""
        self.logger.info("🚀 Starting 24/7 system daemon...")

        try:
            while True:
                await self._handle_reconnect()

                now_ny = datetime.datetime.now(NY)
                self.retrain.poll(now_ny)

                if self._is_market_open(now_ny):
                    if self.retrain.running:
                        self.logger.warning(
                            "⚠️ Retrain still running; staying connected without trading this loop."
                        )
                        await asyncio.sleep(RETRAIN_BUSY_POLL_SECONDS)
                        continue
                    await self._run_market_cycle(now_ny)
                    continue

                self.logger.info(
                    f"🌙 After hours (NY {now_ny:%H:%M}): idle / maintenance mode."
                )
                self.retrain.maybe_start(now_ny)
                await asyncio.sleep(OFF_MARKET_POLL_SECONDS)

        except asyncio.CancelledError:
            await self.retrain.stop()
            self.logger.info("Stop signal received; daemon shut down cleanly.")
