# ibkrpy/shared/session_clock.py
# 交易時段管理：負責時間的切換與檢查

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("ibkrpy.session_clock")

NY = ZoneInfo("America/New_York")

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None


class Session(str, Enum):
    """
    美股的時段劃分。時間皆為紐約時間。

    OVERNIGHT 指 IBKR 的 Overnight Trading (20:00 - 03:50)，僅部分標的可交易，
    且流動性與價差和日盤完全不同 —— 不可以套用同一組成本假設。
    """

    CLOSED = "closed"
    PRE = "pre"
    RTH = "rth"
    POST = "post"
    OVERNIGHT = "overnight"

    @property
    def is_tradable(self) -> bool:
        return self is not Session.CLOSED

    @property
    def is_extended(self) -> bool:
        """非常規時段。流動性較差，成本假設必須另計。"""
        return self in (Session.PRE, Session.POST, Session.OVERNIGHT)


_PRE_OPEN = dt.time(4, 0)
_RTH_OPEN = dt.time(9, 30)
_RTH_CLOSE = dt.time(16, 0)
_POST_CLOSE = dt.time(20, 0)
_OVERNIGHT_CLOSE = dt.time(3, 50)


@dataclass(frozen=True)
class SessionState:
    session: Session
    now: dt.datetime
    next_change: dt.datetime
    next_session: Session

    @property
    def seconds_until_change(self) -> float:
        return max((self.next_change - self.now).total_seconds(), 0.0)


class SessionClock:
    """
    以 NYSE 日曆判斷時段。沒有 pandas_market_calendars 時退回粗略判斷，
    但會明確警告 —— 因為假日與提早收市在降級模式下無法辨識。

    契約: classify() 與 next_transition() 永不拋例外。日曆查詢失敗時
    退回粗略判斷並記錄，寧可多醒幾次也不要讓迴圈整個停住。
    """

    def __init__(self, calendar_name: str = "NYSE", enable_overnight: bool = False):
        self.enable_overnight = bool(enable_overnight)
        self._cache: dict = {}
        self._calendar = None
        if mcal is not None:
            try:
                self._calendar = mcal.get_calendar(calendar_name)
            except Exception as e:
                logger.error(f"無法載入 {calendar_name} 日曆: {e}")
        if self._calendar is None:
            logger.warning(
                "未安裝 pandas_market_calendars，退回「週一至週五 09:30-16:00」的粗略判斷。"
                " 休市日與提早收市日將無法辨識，建議執行 poetry install 補齊相依。"
            )


    def _rth_bounds(self, day: dt.date) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        """該日的 (開盤, 收盤)，紐約時間。休市日回傳 None。"""
        if day in self._cache:
            return self._cache[day]

        bounds = None
        if self._calendar is not None:
            try:
                sched = self._calendar.schedule(start_date=day, end_date=day)
                if not sched.empty:
                    bounds = (
                        sched.iloc[0]["market_open"].tz_convert(NY).to_pydatetime(),
                        sched.iloc[0]["market_close"].tz_convert(NY).to_pydatetime(),
                    )
            except Exception as e:
                logger.error(f"查詢交易日曆失敗 ({day}): {e}")
                bounds = self._crude_bounds(day)
        else:
            bounds = self._crude_bounds(day)

        if len(self._cache) > 400:
            self._cache.clear()
        self._cache[day] = bounds
        return bounds

    @staticmethod
    def _crude_bounds(day: dt.date) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        if day.weekday() >= 5:
            return None
        return (
            dt.datetime.combine(day, _RTH_OPEN, tzinfo=NY),
            dt.datetime.combine(day, _RTH_CLOSE, tzinfo=NY),
        )

    def is_trading_day(self, day: dt.date) -> bool:
        return self._rth_bounds(day) is not None

    def _next_trading_day(self, day: dt.date, limit: int = 10) -> Optional[dt.date]:
        """夜盤判斷「不」使用這個函式 (見 classify)，但排程與回填會用到。"""
        for i in range(1, limit + 1):
            cand = day + dt.timedelta(days=i)
            if self.is_trading_day(cand):
                return cand
        return None


    def classify(self, now: Optional[dt.datetime] = None) -> Session:
        now = self._to_ny(now)
        t = now.time()

        if self.enable_overnight:
            if t >= _POST_CLOSE:
                if self.is_trading_day(now.date() + dt.timedelta(days=1)):
                    return Session.OVERNIGHT
            elif t < _OVERNIGHT_CLOSE:
                if self.is_trading_day(now.date()):
                    return Session.OVERNIGHT

        bounds = self._rth_bounds(now.date())
        if bounds is None:
            return Session.CLOSED

        rth_open, rth_close = bounds
        if now < rth_open:
            return Session.PRE if t >= _PRE_OPEN else Session.CLOSED
        if now <= rth_close:
            return Session.RTH
        return Session.POST if t < _POST_CLOSE else Session.CLOSED


    def next_transition(
        self, now: Optional[dt.datetime] = None
    ) -> Tuple[dt.datetime, Session]:
        """
        回傳 (下一次時段改變的時刻, 改變後的時段)。

        作法是「往前掃描候選邊界，取第一個時段真的不同的」。比起手寫
        每個時段的下一步，這個作法不會在假日、提早收市、夜盤跨日這些
        邊角情況上漏掉分支 —— 邊界由日曆提供，判斷交給 classify()。
        """
        now = self._to_ny(now)
        current = self.classify(now)

        for cand in self._candidate_boundaries(now):
            if cand <= now:
                continue
            nxt = self.classify(cand)
            if nxt is not current:
                return cand, nxt

        fallback = now + dt.timedelta(hours=1)
        return fallback, self.classify(fallback)

    def _candidate_boundaries(self, now: dt.datetime):
        """未來 10 天內所有可能的時段邊界，已排序。"""
        out = []
        for offset in range(0, 11):
            day = now.date() + dt.timedelta(days=offset)
            bounds = self._rth_bounds(day)
            for tod in (_OVERNIGHT_CLOSE, _PRE_OPEN, _POST_CLOSE):
                out.append(dt.datetime.combine(day, tod, tzinfo=NY))
            if bounds is not None:
                out.extend(bounds)
            else:
                out.append(dt.datetime.combine(day, _RTH_OPEN, tzinfo=NY))
        return sorted(c.replace(second=0, microsecond=0) for c in set(out))

    def state(self, now: Optional[dt.datetime] = None) -> SessionState:
        now = self._to_ny(now)
        current = self.classify(now)
        when, nxt = self.next_transition(now)
        return SessionState(current, now, when, nxt)


    @staticmethod
    def _to_ny(now: Optional[dt.datetime]) -> dt.datetime:
        if now is None:
            return dt.datetime.now(NY)
        if now.tzinfo is None:
            return now.replace(tzinfo=NY)
        return now.astimezone(NY)


def build_session_clock(config) -> SessionClock:
    """Composition Root 使用。"""
    s = (config.get("session_settings") or {}) if config else {}
    return SessionClock(
        calendar_name=s.get("calendar", "NYSE"),
        enable_overnight=bool(s.get("enable_overnight", False)),
    )
