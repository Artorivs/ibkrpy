# ibkrpy/core/system_daemon.py
# 全天候守護進程，負責調度實盤交易與自動模型重訓 (24/7 運行)

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
    """控制整個量化系統的日夜節律：只在常規時段交易，休市時交給 RetrainScheduler 重訓。"""

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
        self.logger.info("已清除模型快取，下一輪將載入新權重。")

    def _is_market_open(self, now_ny: datetime.datetime) -> bool:
        """常規時段 (RTH)。日曆、假日與提早收市由 SessionClock 統一處理。"""
        return self.engine.session_clock.classify(now_ny) is Session.RTH

    async def _handle_reconnect(self):
        """處理 IBKR API 在 24 小時運行中可能出現的斷線問題"""
        if self.ib_manager.ib.isConnected():
            return
        self.logger.warning("檢測到 IBKR 斷線，嘗試重新連接...")
        try:
            await self.ib_manager.connect()
            if self._reconnect_failures:
                self.logger.info(
                    f"IBKR 重新連接成功（先前失敗 {self._reconnect_failures} 次）。"
                )
            else:
                self.logger.info("IBKR 重新連接成功！")
            self._reconnect_failures = 0
        except Exception as e:
            self._reconnect_failures += 1
            msg = (
                f"重連失敗 (第 {self._reconnect_failures} 次): {e}，將在下個迴圈重試。"
            )
            if self._reconnect_failures >= 3:
                self.logger.error(msg)
            else:
                self.logger.warning(msg)

    def _scan_order(self):
        """輪替掃描起點"""
        if not self.symbols:
            return []
        n = len(self.symbols)
        self._scan_offset = (self._scan_offset + 1) % n
        return self.symbols[self._scan_offset :] + self.symbols[: self._scan_offset]

    async def _run_market_cycle(self, now_ny: datetime.datetime):
        self.logger.info("=" * 40)
        self.logger.info(f"📈 盤中時間 (NY: {now_ny:%H:%M}) - 執行 AI 掃描與決策")
        self.logger.info("=" * 40)

        if self.ib_manager.ib.isConnected():
            await self.engine.update_system_state()
            for symbol in self._scan_order():
                await self.engine.run_tick(symbol)
                await asyncio.sleep(2)

        self.logger.info(
            f"✅ 掃描完成。等待下一個 {self.tick_interval_minutes} 分鐘 K 線..."
        )
        await asyncio.sleep(self.tick_interval_minutes * 60)

    async def run_24_7(self):
        """24 小時主迴圈"""
        self.logger.info("🚀 啟動 24/7 系統守護進程 (System Daemon)...")

        try:
            while True:
                await self._handle_reconnect()

                now_ny = datetime.datetime.now(NY)
                self.retrain.poll(now_ny)

                if self._is_market_open(now_ny):
                    if self.retrain.running:
                        self.logger.warning(
                            "⚠️ 重訓仍在進行中，本輪僅維持連線，不進行交易決策。"
                        )
                        await asyncio.sleep(RETRAIN_BUSY_POLL_SECONDS)
                        continue
                    await self._run_market_cycle(now_ny)
                    continue

                self.logger.info(
                    f"🌙 盤後時間 (NY: {now_ny:%H:%M}) - 系統進入休眠/維護模式。"
                )
                self.retrain.maybe_start(now_ny)
                await asyncio.sleep(OFF_MARKET_POLL_SECONDS)

        except asyncio.CancelledError:
            await self.retrain.stop()
            self.logger.info("接收到停止信號，守護進程安全關閉。")
