# ibkrpy/core/system_daemon.py
# 全天候守護進程，負責調度實盤交易與自動模型重訓 (24/7 運行)

import asyncio
import datetime
import json
import os
import sys
from zoneinfo import ZoneInfo

from ibkrpy.shared.system_log import global_logger
from ibkrpy.manager.trading_engine import TradingEngine
from ibkrpy.manager.pipeline_manager import PipelineManager
from ibkrpy.data.ibkr_data_manager import IBKRDataManager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_PATH = os.path.join(PROJECT_ROOT, "data", "daemon_state.json")

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None


class SystemDaemon:
    """控制整個量化系統的日夜節律"""

    def __init__(
        self,
        ib_manager: IBKRDataManager,
        trading_engine: TradingEngine,
        pipeline_manager: PipelineManager,
        symbols: list,
        retrain_client_id: int = 101,
    ):
        self.ib_manager = ib_manager
        self.engine = trading_engine
        self.pipeline = pipeline_manager
        self.symbols = list(symbols)
        self.logger = global_logger
        self.tick_interval_minutes = 5

        # 重訓以獨立行程執行，必須使用不同的 client_id，否則會與 daemon 的連線衝突
        self.retrain_client_id = retrain_client_id
        self._retrain_task = None

        self._scan_offset = 0
        self._reconnect_failures = 0
        self._session_cache = {}

        self.last_retrain_date = self._load_state().get("last_retrain_date")

        if mcal is not None:
            self._calendar = mcal.get_calendar("NYSE")
        else:
            self._calendar = None
            self.logger.warning(
                "未安裝 pandas_market_calendars，退回「週一至週五 09:30-16:00」的粗略判斷。"
                " 休市日與提早收市日將無法辨識，建議執行 poetry install 補齊相依。"
            )

    # ------------------------------------------------------------------
    # 狀態持久化
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"last_retrain_date": self.last_retrain_date}, f)
        except Exception as e:
            self.logger.error(f"寫入 daemon 狀態失敗: {e}")

    # ------------------------------------------------------------------
    # 交易日曆
    # ------------------------------------------------------------------

    def _get_session(self, day: datetime.date):
        """
        回傳該日的 (開盤時刻, 收盤時刻)，休市日回傳 None。
        pandas_market_calendars 的 schedule 已包含提早收市 (例如感恩節隔日 13:00)。
        """
        if day in self._session_cache:
            return self._session_cache[day]

        session = None
        if self._calendar is not None:
            try:
                sched = self._calendar.schedule(start_date=day, end_date=day)
                if not sched.empty:
                    session = (sched.iloc[0]["market_open"], sched.iloc[0]["market_close"])
            except Exception as e:
                self.logger.error(f"查詢交易日曆失敗 ({day}): {e}")

        # 快取只保留近期日期，避免長期運行時無限成長
        if len(self._session_cache) > 10:
            self._session_cache.clear()
        self._session_cache[day] = session
        return session

    def _is_market_open(self, now_ny: datetime.datetime) -> bool:
        session = self._get_session(now_ny.date())

        if self._calendar is not None:
            if session is None:
                return False   # 週末或假日
            market_open, market_close = session
            return market_open <= now_ny <= market_close

        # 沒有日曆套件時的降級判斷 (無法辨識假日與早收)
        if now_ny.weekday() >= 5:
            return False
        market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_ny <= market_close

    # ------------------------------------------------------------------
    # 連線維護
    # ------------------------------------------------------------------

    async def _handle_reconnect(self):
        """處理 IBKR API 在 24 小時運行中可能出現的斷線問題"""
        if self.ib_manager.ib.isConnected():
            return
        self.logger.warning("檢測到 IBKR 斷線，嘗試重新連接...")
        try:
            await self.ib_manager.connect()
            if self._reconnect_failures:
                self.logger.info(f"IBKR 重新連接成功（先前失敗 {self._reconnect_failures} 次）。")
            else:
                self.logger.info("IBKR 重新連接成功！")
            self._reconnect_failures = 0
        except Exception as e:
            self._reconnect_failures += 1
            msg = f"重連失敗 (第 {self._reconnect_failures} 次): {e}，將在下個迴圈重試。"
            if self._reconnect_failures >= 3:
                self.logger.error(msg)
            else:
                self.logger.warning(msg)

    # ------------------------------------------------------------------
    # 重訓 (獨立行程)
    # ------------------------------------------------------------------

    async def _run_retrain_subprocess(self):
        """
        以獨立 subprocess 執行完整重訓。

        用獨立行程而非 asyncio.to_thread 的理由：
          1. TensorFlow 會在行程層級固定執行緒池與記憶體，訓練結束後不會歸還；
             獨立行程結束即完全釋放。
          2. 訓練中的例外或 OOM 不會拖垮 24/7 的交易主行程。
          3. GIL 完全隔離，主迴圈的心跳絕對不會被拖慢。
        """
        cmd = [
            sys.executable, "-m", "ibkrpy.core.main",
            "--mode", "autopilot",
            "--client-id", str(self.retrain_client_id),
        ]
        self.logger.info(f"🛠️ 啟動重訓子行程 (client_id={self.retrain_client_id}): {' '.join(cmd)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=PROJECT_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # 逐行轉發子行程輸出，避免 PIPE 緩衝區填滿造成死鎖
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self.logger.info(f"[重訓] {line}")

            rc = await proc.wait()
            if rc == 0:
                self.logger.info("✅ 重訓子行程正常結束。")
                # 權重已更新：清掉記憶體中的模型與 scaler，讓下一輪載入新版本
                self.engine.models.invalidate()
                self.logger.info("已清除模型快取，下一輪將載入新權重。")
                return True

            self.logger.error(f"❌ 重訓子行程異常結束 (exit code {rc})，本週保養未完成。")
            return False

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"啟動重訓子行程失敗: {e}")
            return False

    def _should_retrain(self, now_ny: datetime.datetime) -> bool:
        """
        週末保養判斷。

        [修正] 舊版只認週六 (weekday == 5)，週日啟動的行程整週都不會重訓。
        改為「週末任一天，且本週尚未重訓過」。
        """
        if now_ny.weekday() < 5:
            return False
        if self._retrain_task is not None:
            return False

        year, week, _ = now_ny.isocalendar()
        current_week = f"{year}-W{week:02d}"
        return self.last_retrain_date != current_week

    def _mark_retrained(self, now_ny: datetime.datetime):
        year, week, _ = now_ny.isocalendar()
        self.last_retrain_date = f"{year}-W{week:02d}"
        self._save_state()

    # ------------------------------------------------------------------
    # 主迴圈
    # ------------------------------------------------------------------

    def _scan_order(self):
        """
        [修正] 輪替掃描起點。

        資金有限時，固定順序會讓 config.yaml 中排在後面的標的長期拿不到資金
        (前面的標的先把現金用完，後面的因低於 min_trade_usd 被跳過)。
        每輪把起點往後移一格，長期下來機會均等。
        """
        if not self.symbols:
            return []
        n = len(self.symbols)
        self._scan_offset = (self._scan_offset + 1) % n
        return self.symbols[self._scan_offset:] + self.symbols[:self._scan_offset]

    async def run_24_7(self):
        """24 小時主迴圈"""
        self.logger.info("🚀 啟動 24/7 系統守護進程 (System Daemon)...")
        ny_tz = ZoneInfo("America/New_York")

        try:
            while True:
                await self._handle_reconnect()

                now_ny = datetime.datetime.now(ny_tz)
                is_open = self._is_market_open(now_ny)

                # 先回收已完成的重訓 task，主迴圈不會停下來等它
                if self._retrain_task is not None and self._retrain_task.done():
                    try:
                        if self._retrain_task.result():
                            self._mark_retrained(now_ny)
                    except Exception as e:
                        self.logger.error(f"重訓 task 異常: {e}")
                    finally:
                        self._retrain_task = None

                if is_open:
                    # ===== 盤中：執行高頻實盤交易 =====
                    if self._retrain_task is not None:
                        self.logger.warning("⚠️ 重訓仍在進行中，本輪僅維持連線，不進行交易決策。")
                        await asyncio.sleep(60)
                        continue

                    self.logger.info("=" * 40)
                    self.logger.info(f" 📈 盤中時間 (NY: {now_ny.strftime('%H:%M')}) - 執行 AI 掃描與決策")
                    self.logger.info("=" * 40)

                    if self.ib_manager.ib.isConnected():
                        await self.engine.update_system_state()
                        for symbol in self._scan_order():
                            await self.engine.run_tick(symbol)
                            await asyncio.sleep(2)   # 避免 API 風險

                    self.logger.info(f"✅ 掃描完成。等待下一個 {self.tick_interval_minutes} 分鐘 K 線...")
                    await asyncio.sleep(self.tick_interval_minutes * 60)

                else:
                    # ===== 盤後/週末：維護與模型重訓 =====
                    self.logger.info(f"🌙 盤後時間 (NY: {now_ny.strftime('%H:%M')}) - 系統進入休眠/維護模式。")

                    if self._should_retrain(now_ny):
                        self.logger.info("🛠️ 觸發週末定期保養：以獨立行程更新歷史數據與模型重訓。")
                        self._retrain_task = asyncio.create_task(self._run_retrain_subprocess())

                    # 盤後每 5 分鐘檢查一次，隔天開盤最多延遲數分鐘就會甦醒
                    await asyncio.sleep(300)

        except asyncio.CancelledError:
            if self._retrain_task is not None:
                self._retrain_task.cancel()
            self.logger.info("接收到停止信號，守護進程安全關閉。")