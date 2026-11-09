# ibkrpy/strategy/strategy_components.py
# Shared strategy types: market regime, risk rules and position sizing

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("ibkrpy.strategy")


class MarketRegime(Enum):
    """Market regime. The single definition used across the system."""

    BULL_TREND = auto()
    BEAR_TREND = auto()
    SIDEWAYS_VOLATILE = auto()
    SIDEWAYS_QUIET = auto()


class IRiskRule(ABC):
    """Risk rule.

    Contract: assess returns (are new entries allowed, reason).

    Rules gate new exposure only. They must never block reducing or closing:
    taking away the ability to cut risk when risk is highest defeats the purpose.
    """

    @abstractmethod
    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class VIXHaltRule(IRiskRule):
    """Block new entries when VIX is above the threshold."""

    def __init__(self, threshold: float = 30.0):
        self.threshold = float(threshold)

    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        vix_series = context.get("vix_series")
        if vix_series is None or len(vix_series) == 0:
            return True, "No VIX data; not blocking"

        latest = float(vix_series.iloc[-1])
        if latest > self.threshold:
            return False, f"VIX {latest:.2f} above threshold {self.threshold:.0f}."
        return True, f"VIX {latest:.2f} normal"


class ReversalRiskRule(IRiskRule):
    """Block new entries when top-reversal risk is too high.

    Separate from CoreStrategy's reversal_block_level: the strategy judges one symbol's
    signal, this rule is the portfolio-level entry gate. Both must pass.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = float(threshold)

    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        risk = context.get("reversal_risk")
        if risk is None:
            return True, "No reversal risk data"
        if float(risk) >= self.threshold:
            return (
                False,
                f"Top-reversal risk {float(risk):.0%} above threshold {self.threshold:.0%}.",
            )
        return True, f"Reversal risk {float(risk):.0%} acceptable"


class EquityWatermark:
    """Day-start and peak NLV, persisted so a restart doesn't reset them.

    Deposits and withdrawals distort both values; delete the state file after one.
    """

    def __init__(self, path: str = None):
        self.path = path
        self.day: str = None
        self.day_start: float = None
        self.peak: float = None
        self.halted_day: str = None
        self._load()

    def _load(self):
        if not self.path:
            return
        try:
            import json
            import os

            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.day = data.get("day")
                self.day_start = data.get("day_start")
                self.peak = data.get("peak")
                self.halted_day = data.get("halted_day")
        except Exception as exc:
            logger.error(
                f"Could not read risk state {self.path}: {exc}; starting fresh."
            )

    def _save(self):
        if not self.path:
            return
        try:
            import json
            import os

            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "day": self.day,
                        "day_start": self.day_start,
                        "peak": self.peak,
                        "halted_day": self.halted_day,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp, self.path)
        except Exception as exc:
            logger.error(f"Could not write risk state {self.path}: {exc}")

    def update(self, nlv: float, day: str) -> None:
        """Call once per loop with the latest NLV and the New York date (YYYY-MM-DD)."""
        if nlv is None or not math.isfinite(nlv) or nlv <= 0:
            return
        changed = False
        if self.day != day:
            self.day, self.day_start = day, float(nlv)
            changed = True
        if self.peak is None or nlv > self.peak:
            self.peak = float(nlv)
            changed = True
        if changed:
            self._save()

    def latch_halt(self) -> None:
        if self.halted_day != self.day:
            self.halted_day = self.day
            self._save()

    def snapshot(self, nlv: float) -> Dict[str, Any]:
        return {
            "net_liquidation": nlv,
            "nlv_day_start": self.day_start,
            "nlv_peak": self.peak,
            "halted_today": self.halted_day is not None and self.halted_day == self.day,
            "watermark": self,
        }


class SessionEntryRule(IRiskRule):
    """Opens new positions only in regular hours; extended hours are close-only.

    Pre-market, after-hours and overnight have wide spreads, the Adaptive entry order
    only works in regular hours, and the bracket's stop is not set to trigger outside
    them, so a position opened then would sit unprotected until the next open.
    """

    def __init__(self, allow_extended_entries: bool = False):
        self.allow_extended_entries = bool(allow_extended_entries)

    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        session = context.get("session")
        if session is None or self.allow_extended_entries:
            return True, "Session allows entries"
        if getattr(session, "is_extended", False):
            name = getattr(session, "value", str(session))
            return False, f"Outside regular hours ({name}); close-only."
        return True, "Regular hours"


class DrawdownHaltRule(IRiskRule):
    """Blocks new entries on a daily loss (latched for the day) or a drawdown from peak."""

    def __init__(
        self, max_daily_loss_pct: float = 0.02, max_drawdown_pct: float = 0.10
    ):
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)

    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        nlv = context.get("net_liquidation")
        if not nlv or nlv <= 0:
            return True, "No NLV data; not blocking"

        if context.get("halted_today"):
            return False, "Daily loss limit hit earlier today; no new entries."

        day_start = context.get("nlv_day_start")
        if day_start and day_start > 0 and self.max_daily_loss_pct > 0:
            day_loss = 1.0 - nlv / day_start
            if day_loss >= self.max_daily_loss_pct:
                watermark = context.get("watermark")
                if watermark is not None:
                    watermark.latch_halt()
                return (
                    False,
                    f"Daily loss {day_loss:.2%} reached the "
                    f"{self.max_daily_loss_pct:.2%} limit; no new entries today.",
                )

        peak = context.get("nlv_peak")
        if peak and peak > 0 and self.max_drawdown_pct > 0:
            drawdown = 1.0 - nlv / peak
            if drawdown >= self.max_drawdown_pct:
                return (
                    False,
                    f"Drawdown {drawdown:.2%} from peak {peak:,.0f} reached the "
                    f"{self.max_drawdown_pct:.2%} limit; no new entries.",
                )

        return True, "Drawdown within limits"


class RiskController:
    """Applies all risk rules in order; any veto blocks new entries."""

    def __init__(self, rules: List[IRiskRule]):
        self.rules = list(rules)

    def check_entries_allowed(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        for rule in self.rules:
            try:
                allowed, reason = rule.assess(context)
            except Exception as exc:
                logger.error(f"Risk rule {rule.name} raised; treating as a veto: {exc}")
                return False, f"{rule.name} failed"
            if not allowed:
                return False, reason
        return True, "Risk checks passed"

    def check_trade_allowed(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Legacy alias.

        The name suggests all trades; use check_entries_allowed in new code.
        """
        return self.check_entries_allowed(context)


class PositionSizer:
    """Converts a risk budget into a share count.

    Uses the risk budget when risk per share is given, otherwise a fraction of capital.
    """

    def __init__(
        self,
        max_equity_pct: float = 0.2,
        min_trade_usd: float = 10.0,
        risk_budget_pct: float = 0.01,
    ):
        self.max_equity_pct = float(max_equity_pct)
        self.min_trade_usd = float(min_trade_usd)
        self.risk_budget_pct = float(risk_budget_pct)

    def calculate_size(
        self, capital: float, price: float, risk_per_share: float = None
    ) -> int:
        if price <= 0 or capital <= 0:
            return 0

        if risk_per_share and risk_per_share > 0:
            quantity = math.floor((capital * self.risk_budget_pct) / risk_per_share)
        else:
            quantity = math.floor((capital * self.max_equity_pct) / price)

        if quantity <= 0 or quantity * price < self.min_trade_usd:
            return 0
        return quantity
