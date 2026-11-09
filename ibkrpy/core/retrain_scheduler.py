# ibkrpy/core/retrain_scheduler.py
# Off-market retrain scheduling, shared by SystemDaemon and the live loop

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

# Child output is relayed line by line. TensorFlow / Optuna sometimes print lines over
# asyncio's default 64 KiB limit, which makes readline raise and the run look failed.
_PIPE_LIMIT = 16 * 1024 * 1024


class RetrainScheduler:
    """
    Schedules model retraining while the market is closed.

    The caller only says whether the market is closed; this class handles which days
    are allowed, how often, the freeze date, persisted state and retry timing.

    Settings (training_settings in config.yaml):
            auto_retrain:      false disables automatic retraining (default true)
            retrain_days:      weekdays allowed, default [sat, sun]
            retrain_every:     week (once per ISO week, default) | day (once per allowed day)
            retry_minutes:     wait after a failed retrain, default 60
            retrain_client_id: IBKR client_id for the retrain subprocess, default 101
            freeze_until:      YYYY-MM-DD; no retraining on or before this date
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
    # Settings
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
                    f"training_settings.retrain_days has an unrecognized value ({item!r}); ignored."
                    f" Valid values: {', '.join(_DAY_NAMES)}"
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
        """Read the freeze date. A malformed value does not freeze, but is reported loudly."""
        raw = self._setting("freeze_until")
        if not raw:
            return None
        if isinstance(raw, datetime.date):
            return raw
        try:
            return datetime.date.fromisoformat(str(raw).strip())
        except (ValueError, TypeError):
            self.logger.error(
                f"training_settings.freeze_until is malformed ({raw!r}); "
                f"expected YYYY-MM-DD. Not freezing this time."
            )
            return None

    # ------------------------------------------------------------------
    # State
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
            self.logger.error(f"Failed to write retrain state: {e}")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def is_due(self, now: datetime.datetime) -> bool:
        """Assumes the caller has checked that the market is closed."""
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
                    f"❄️ Models frozen; skipping retrain (until {frozen_until}, {remaining} days left)."
                    f" To lift it, clear training_settings.freeze_until."
                )
            return False

        return self.last_period != self._period_of(now)

    def maybe_start(self, now: datetime.datetime) -> bool:
        """Call while the market is closed. Starts the retrain subprocess when due; returns whether it did."""
        self.poll(now)
        if not self.is_due(now):
            return False
        self._task_period = self._period_of(now)
        self.logger.info(
            f"🛠️ Off-market maintenance ({self._task_period}): updating data and retraining in a subprocess."
        )
        self._task = asyncio.create_task(self._run_subprocess())
        return True

    def poll(self, now: datetime.datetime) -> None:
        """Reap a finished retrain. Call at the start of each loop."""
        if self._task is None or not self._task.done():
            return
        task, period = self._task, self._task_period
        self._task = self._task_period = None

        try:
            ok = bool(task.result())
        except asyncio.CancelledError:
            ok = False
        except Exception as e:
            self.logger.error(f"Retrain task error: {e}")
            ok = False

        if not ok:
            self._retry_at = now + self._retry_delay()
            self.logger.warning(
                f"Retrain did not finish; retrying in the first closed session after {self._retry_at:%m-%d %H:%M} (NY)."
            )
            return

        self._retry_at = None
        self.last_period = period
        self._save_state()
        if self.on_success is not None:
            try:
                self.on_success()
            except Exception as e:
                self.logger.error(f"Post-retrain callback failed: {e}")

    def seconds_until_recheck(self, now: datetime.datetime) -> float:
        """
        Longest sleep while closed: short while a retrain runs so it's reaped promptly,
        otherwise until New York midnight so a newly allowed day isn't slept through.
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
        """Run a full retrain (download + train) in a separate subprocess."""
        self.logger.info(
            f"🛠️ Starting retrain subprocess (client_id={self.client_id}): {' '.join(self.command)}"
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
                    self.logger.info(f"[retrain] {line}")

            rc = await proc.wait()
            if rc == 0:
                self.logger.info("✅ Retrain subprocess finished.")
                return True

            self.logger.error(
                f"❌ Retrain subprocess exited abnormally (exit code {rc}); maintenance incomplete."
            )
            return False

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"Retrain subprocess failed: {e}")
            return False
        finally:
            # On shutdown or read failure, don't leave an orphaned retrain holding the client_id.
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
