# ibkrpy/shared/session_clock.py
# Trading session clock: classifies the current session and finds transitions

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
    US equity sessions, all in New York time.

    OVERNIGHT is IBKR Overnight Trading (20:00-03:50). Only some symbols trade, and
    liquidity and spreads differ from the day session, so cost assumptions differ too.
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
        """Outside regular hours: thinner liquidity, separate cost assumptions."""
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
    Classifies sessions with the NYSE calendar. Without pandas_market_calendars it falls back
    to a rough weekday schedule and warns, since holidays and early closes are then missed.

    Contract: classify() and next_transition() never raise. A calendar failure falls back to
    the rough schedule and is logged; waking too often beats stopping the loop.
    """

    def __init__(self, calendar_name: str = "NYSE", enable_overnight: bool = False):
        self.enable_overnight = bool(enable_overnight)
        self._cache: dict = {}
        self._calendar = None
        if mcal is not None:
            try:
                self._calendar = mcal.get_calendar(calendar_name)
            except Exception as e:
                logger.error(f"Could not load the {calendar_name} calendar: {e}")
        if self._calendar is None:
            logger.warning(
                "pandas_market_calendars is not installed; falling back to Mon-Fri 09:30-16:00."
                " Holidays and early closes will be missed; run poetry install."
            )

    def _rth_bounds(self, day: dt.date) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        """(open, close) for the day in New York time, or None on a holiday."""
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
                logger.error(f"Trading calendar lookup failed ({day}): {e}")
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
        """Not used for the overnight check (see classify), but used by scheduling and backfill."""
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
        Returns (time of the next session change, the session after it).

        Scans candidate boundaries and returns the first where the session actually changes,
        so holidays, early closes and the overnight day roll can't be missed; the calendar
        supplies the boundaries and classify() decides.
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
        """All candidate session boundaries in the next 10 days, sorted."""
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
    """Composition root."""
    s = (config.get("session_settings") or {}) if config else {}
    return SessionClock(
        calendar_name=s.get("calendar", "NYSE"),
        enable_overnight=bool(s.get("enable_overnight", False)),
    )
