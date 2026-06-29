# ibkrpy/core/system_daemon.py
# 全天候守護進程，負責調度實盤交易與自動模型重訓 (24/7 運行)

import asyncio
import datetime
from zoneinfo import ZoneInfo
from ibkrpy.shared.system_log import global_logger
from ibkrpy.manager.trading_engine import TradingEngine
from ibkrpy.manager.pipeline_manager import PipelineManager
from ibkrpy.data.ibkr_data_manager import IBKRDataManager

class SystemDaemon:
    """控制整個量化系統的日夜節律"""
    
    def __init__(self, ib_manager: IBKRDataManager, trading_engine: TradingEngine, pipeline_manager: PipelineManager, symbols: list):
        self.ib_manager = ib_manager
        self.engine = trading_engine
        self.pipeline = pipeline_manager
        self.symbols = symbols
        self.logger = global_logger
        self.tick_interval_minutes = 5
        self.last_retrain_date = None

    def _is_market_open(self, now_ny: datetime.datetime) -> bool:
        """
        精確的市場時間判斷 (採用 America/New_York 時區，自動處理夏令時間)
        """
        # 週末不開盤 (0=週一, ..., 5=週六, 6=週日)
        if now_ny.weekday() >= 5: 
            return False
            
        # 美股常規交易時間：美東 09:30 到 16:00
        market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_ny <= market_close

    async def _handle_reconnect(self):
        """處理 IBKR API 在 24 小時運行中可能出現的斷線問題"""
        if not self.ib_manager.ib.isConnected():
            self.logger.warning("檢測到 IBKR 斷線，嘗試重新連接...")
            try:
                await self.ib_manager.connect()
                self.logger.info("IBKR 重新連接成功！")
            except Exception as e:
                self.logger.error(f"重連失敗: {e}，將在下個迴圈重試。")

    async def run_24_7(self):
        """24 小時主迴圈"""
        self.logger.info("🚀 啟動 24/7 系統守護進程 (System Daemon)...")
        
        # 建立紐約時區物件，徹底隔離伺服器所在地 (如日本 JST) 的時差干擾
        ny_tz = ZoneInfo("America/New_York")
        
        try:
            while True:
                await self._handle_reconnect()
                
                # 獲取當下精準的紐約時間
                now_ny = datetime.datetime.now(ny_tz)
                is_open = self._is_market_open(now_ny)
                
                if is_open:
                    # ===== 盤中：執行高頻實盤交易 =====
                    self.logger.info("=" * 40)
                    self.logger.info(f" 📈 盤中時間 (NY: {now_ny.strftime('%H:%M')}) - 執行 AI 掃描與決策")
                    self.logger.info("=" * 40)
                    
                    if self.ib_manager.ib.isConnected():
                        await self.engine.update_system_state()
                        for symbol in self.symbols:
                            await self.engine.run_tick(symbol)
                            await asyncio.sleep(2) # 避免 API 風險
                            
                    self.logger.info(f"✅ 掃描完成。進入休眠，等待下一個 {self.tick_interval_minutes} 分鐘 K 線...")
                    await asyncio.sleep(self.tick_interval_minutes * 60)
                    
                else:
                    # ===== 盤後/週末：維護與模型重訓 =====
                    self.logger.info(f"🌙 盤後時間 (NY: {now_ny.strftime('%H:%M')}) - 系統進入休眠/維護模式。")
                    
                    # 判斷是否需要進行週末大保養 (紐約時間每週六執行一次模型重訓)
                    if now_ny.weekday() == 5 and now_ny.date() != self.last_retrain_date:
                        self.logger.info("🛠️ 觸發週末定期保養：更新歷史數據與模型重訓！")
                        await self.pipeline.run_autopilot()
                        self.last_retrain_date = now_ny.date()
                        self.logger.info("✅ 週末保養完畢。")
                        
                    # 將盤後的檢查間距從 1 小時縮短為 5 分鐘
                    # 這樣隔天開盤 (09:30) 時，系統最多只會延遲幾分鐘就會瞬間甦醒
                    await asyncio.sleep(300)
                    
        except asyncio.CancelledError:
            self.logger.info("接收到停止信號，守護進程安全關閉。")