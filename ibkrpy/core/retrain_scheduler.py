# ibkrpy/core/retrain_scheduler.py
# 休市重訓排程：決定「何時」以獨立行程重訓，SystemDaemon 與 live 迴圈共用

import asyncio
import datetime
import json
import os
import sys
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from ibkrpy.shared.system_log import global_logger

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
STATE_PATH = os.path.join(PROJECT_ROOT, "data", "daemon_state.json")

NY = ZoneInfo("America/New_York")

_DAY_NAMES = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
DEFAULT_RETRAIN_DAYS = ("sat", "sun")
DEFAULT_CLIENT_ID = 101
DEFAULT_RETRY_MINUTES = 60
RUNNING_RECHECK_SECONDS = 300

# 子行程輸出逐行轉印。TensorFlow / Optuna 偶有超過 asyncio 預設 64 KiB 的單行，
# 預設上限下 readline 會拋 ValueError，重訓就被誤判為「啟動失敗」。
_PIPE_LIMIT = 16 * 1024 * 1024


class RetrainScheduler:
    """
    休市時段的模型重訓排程。

    呼叫端只負責回答「現在是否休市」，其餘交給本類別：
    哪幾天可以重訓、多久一次、凍結期、狀態持久化、失敗後何時重試。

    設定 (config.yaml 的 training_settings):
        auto_retrain:   false 則完全停用自動重訓 (預設 true)
        retrain_days:   允許重訓的星期，預設 [sat, sun]
        retrain_every:  week (每 ISO 週一次，預設) | day (每個允許的日子一次)
        retry_minutes:  重訓失敗後的重試間隔，預設 60
        retrain_client_id: 重訓子行程的 IBKR client_id，預設 101
        freeze_until:   YYYY-MM-DD，當日 (含) 以前不重訓
    """

    def __init__(
        self,
        config,
        on_success: Optional[Callable[[], None]] = None,
        state_path: str = STATE_PATH,
        command: Optional[list] = None,
    ):
        self.config = config
        self.on_success = on_success
        self.state_path = state_path
        self.logger = global_logger

        self.client_id = int(
            self._setting("retrain_client_id", DEFAULT_CLIENT_ID) or DEFAULT_CLIENT_ID
        )
        self.command = command or [
            sys.executable,
            "-m",
            "ibkrpy.core.main",
            "--mode",
            "autopilot",
            "--client-id",
            str(self.client_id),
        ]

        self._task: Optional[asyncio.Task] = None
        self._task_period: Optional[str] = None
        self._retry_at: Optional[datetime.datetime] = None
        self._freeze_logged = None

        self.last_period = self._load_state().get("last_retrain_date")

    # ------------------------------------------------------------------
    # 設定
    # ------------------------------------------------------------------
    def _setting(self, key: str, default=None):
        try:
            value = self.config.get(f"training_settings.{key}", default)
        except Exception:
            return default
        return default if value is None else value

    @property
    def enabled(self) -> bool:
        value = self._setting("auto_retrain", True)
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off")
        return bool(value)

    def _retrain_days(self) -> set:
        raw = self._setting("retrain_days", list(DEFAULT_RETRAIN_DAYS))
        if isinstance(raw, str):
            raw = raw.split(",")
        days = set()
        for item in raw or []:
            key = str(item).strip().lower()[:3]
            if key in _DAY_NAMES:
                days.add(_DAY_NAMES[key])
            else:
                self.logger.error(
                    f"training_settings.retrain_days 含無法辨識的值 ({item!r})，已忽略。"
                    f" 可用值: {', '.join(_DAY_NAMES)}"
                )
        return days

    def _period_of(self, now: datetime.datetime) -> str:
        if str(self._setting("retrain_every", "week")).strip().lower() == "day":
            return now.date().isoformat()
        year, week, _ = now.isocalendar()
        return f"{year}-W{week:02d}"

    def _retry_delay(self) -> datetime.timedelta:
        try:
            minutes = float(self._setting("retry_minutes", DEFAULT_RETRY_MINUTES))
        except (TypeError, ValueError):
            minutes = DEFAULT_RETRY_MINUTES
        return datetime.timedelta(minutes=max(minutes, 1.0))

    def _freeze_until(self) -> Optional[datetime.date]:
        """讀取凍結到期日。格式錯誤時「不」凍結，但明確報錯 —— 靜默凍結更危險。"""
        raw = self._setting("freeze_until")
        if not raw:
            return None
        if isinstance(raw, datetime.date):
            return raw
        try:
            return datetime.date.fromisoformat(str(raw).strip())
        except (ValueError, TypeError):
            self.logger.error(
                f"training_settings.freeze_until 格式錯誤 ({raw!r})，"
                f"應為 YYYY-MM-DD。本次不套用凍結。"
            )
            return None

    # ------------------------------------------------------------------
    # 狀態
    # ------------------------------------------------------------------
    def _load_state(self) -> dict:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({"last_retrain_date": self.last_period}, f)
        except Exception as e:
            self.logger.error(f"寫入重訓狀態失敗: {e}")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # 排程
    # ------------------------------------------------------------------
    def is_due(self, now: datetime.datetime) -> bool:
        """假設呼叫端已確認目前休市。"""
        if not self.enabled or self._task is not None:
            return False
        if now.weekday() not in self._retrain_days():
            return False
        if self._retry_at is not None and now < self._retry_at:
            return False

        frozen_until = self._freeze_until()
        if frozen_until is not None and now.date() <= frozen_until:
            if self._freeze_logged != frozen_until:
                self._freeze_logged = frozen_until
                remaining = (frozen_until - now.date()).days
                self.logger.info(
                    f"❄️ 模型凍結中，跳過重訓 (至 {frozen_until}，尚餘 {remaining} 天)。"
                    f" 解除方式: 清空 training_settings.freeze_until。"
                )
            return False

        return self.last_period != self._period_of(now)

    def maybe_start(self, now: datetime.datetime) -> bool:
        """休市時呼叫。到期就啟動重訓子行程，回傳是否有啟動。"""
        self.poll(now)
        if not self.is_due(now):
            return False
        self._task_period = self._period_of(now)
        self.logger.info(
            f"🛠️ 觸發休市定期保養 ({self._task_period})：以獨立行程更新歷史數據與模型重訓。"
        )
        self._task = asyncio.create_task(self._run_subprocess())
        return True

    def poll(self, now: datetime.datetime) -> None:
        """收割已結束的重訓。每輪迴圈開頭呼叫。"""
        if self._task is None or not self._task.done():
            return
        task, period = self._task, self._task_period
        self._task = self._task_period = None

        try:
            ok = bool(task.result())
        except asyncio.CancelledError:
            ok = False
        except Exception as e:
            self.logger.error(f"重訓 task 異常: {e}")
            ok = False

        if not ok:
            self._retry_at = now + self._retry_delay()
            self.logger.warning(
                f"重訓未完成，將於 {self._retry_at:%m-%d %H:%M} (NY) 之後的休市時段重試。"
            )
            return

        self._retry_at = None
        self.last_period = period
        self._save_state()
        if self.on_success is not None:
            try:
                self.on_success()
            except Exception as e:
                self.logger.error(f"重訓後的回呼失敗: {e}")

    def seconds_until_recheck(self, now: datetime.datetime) -> float:
        """
        休市時最多睡多久。重訓進行中要及時收割；否則睡到紐約午夜，
        讓「進入允許重訓的日子」不會被長時間的休眠錯過。
        """
        if self._task is not None:
            return float(RUNNING_RECHECK_SECONDS)
        now = now.astimezone(NY)
        midnight = datetime.datetime.combine(
            now.date() + datetime.timedelta(days=1), datetime.time(0, 0), tzinfo=NY
        )
        candidates = [(midnight - now).total_seconds() + 1.0]
        if self._retry_at is not None:
            candidates.append((self._retry_at - now).total_seconds() + 1.0)
        return max(min(candidates), 1.0)

    async def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_subprocess(self) -> bool:
        """以獨立 subprocess 執行完整重訓 (下載 + 訓練)。"""
        self.logger.info(
            f"🛠️ 啟動重訓子行程 (client_id={self.client_id}): {' '.join(self.command)}"
        )
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=PROJECT_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=_PIPE_LIMIT,
            )

            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self.logger.info(f"[重訓] {line}")

            rc = await proc.wait()
            if rc == 0:
                self.logger.info("✅ 重訓子行程正常結束。")
                return True

            self.logger.error(
                f"❌ 重訓子行程異常結束 (exit code {rc})，本次保養未完成。"
            )
            return False

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"重訓子行程執行失敗: {e}")
            return False
        finally:
            # 主行程關閉或讀取失敗時，別留下孤兒重訓行程繼續佔用 client_id。
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
