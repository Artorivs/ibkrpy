from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ibkrpy.execution_costs")

_SEC_FEE_RATE = 0.0000278
_FINRA_TAF_PER_SHARE = 0.000166
_FINRA_TAF_CAP = 8.30


class CommissionModel(ABC):
    """Commission for one side (one order)."""

    @abstractmethod
    def per_order(self, shares: float, price: float) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class ZeroCommission(CommissionModel):
    """IBKR Lite (US clients, US stocks only)."""

    name = "zero"

    def per_order(self, shares: float, price: float) -> float:
        return 0.0


class _PerShareWithMinimum(CommissionModel):
    """
    Per share + per-order minimum + notional cap: the general shape of IBKR stock commissions.
    """

    def __init__(self, per_share: float, minimum: float, max_pct: float = 0.01):
        self.per_share = float(per_share)
        self.minimum = float(minimum)
        self.max_pct = float(max_pct)

    def per_order(self, shares: float, price: float) -> float:
        shares = abs(float(shares))
        if shares <= 0 or price <= 0:
            return 0.0
        notional = shares * price
        raw = self.per_share * shares
        return min(max(raw, self.minimum), self.max_pct * notional)


class IBKRFixedCommission(_PerShareWithMinimum):
    name = "ibkr_fixed"

    def __init__(self, per_share=0.005, minimum=1.00, max_pct=0.01):
        super().__init__(per_share, minimum, max_pct)


class IBKRTieredCommission(_PerShareWithMinimum):
    name = "ibkr_tiered"

    def __init__(self, per_share=0.0035, minimum=0.35, max_pct=0.01):
        super().__init__(per_share, minimum, max_pct)


_COMMISSION_MODELS = {
    "ibkr_fixed": IBKRFixedCommission,
    "ibkr_tiered": IBKRTieredCommission,
    "zero": ZeroCommission,
}


@dataclass(frozen=True)
class RoundTripCost:
    """Round-trip cost breakdown. Amounts in USD, ratios as decimals."""

    notional: float
    commission: float
    spread: float
    slippage: float
    regulatory: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.slippage + self.regulatory

    @property
    def pct(self) -> float:
        """Cost as a fraction of notional: the edge has to exceed this."""
        return self.total / self.notional if self.notional > 0 else float("inf")

    def explain(self) -> str:
        if self.notional <= 0:
            return "zero notional"
        p = lambda x: f"{x / self.notional * 100:.3f}%"
        return (
            f"round trip ${self.total:.2f} ({p(self.total)}) = "
            f"commission ${self.commission:.2f}({p(self.commission)}) + "
            f"spread ${self.spread:.2f}({p(self.spread)}) + "
            f"slippage ${self.slippage:.2f}({p(self.slippage)}) + "
            f"fees ${self.regulatory:.2f}"
        )


class ExecutionCostModel:
    def __init__(
        self,
        commission: CommissionModel,
        default_spread_bps: float = 2.0,
        slippage_bps: float = 1.0,
        extended_spread_multiplier: float = 4.0,
        include_regulatory: bool = True,
    ):
        self.commission = commission
        self.default_spread_bps = float(default_spread_bps)
        self.slippage_bps = float(slippage_bps)
        self.extended_spread_multiplier = float(extended_spread_multiplier)
        self.include_regulatory = bool(include_regulatory)

    def estimate(
        self,
        price: float,
        shares: float,
        spread_bps: Optional[float] = None,
        extended_session: bool = False,
    ) -> RoundTripCost:
        try:
            price = float(price)
            shares = abs(float(shares))
        except (TypeError, ValueError):
            return RoundTripCost(0.0, 0.0, 0.0, 0.0, 0.0)

        if price <= 0 or shares <= 0 or not math.isfinite(price * shares):
            return RoundTripCost(0.0, 0.0, 0.0, 0.0, 0.0)

        notional = shares * price
        commission = 2.0 * self.commission.per_order(shares, price)

        bps = self.default_spread_bps if spread_bps is None else float(spread_bps)
        if extended_session:
            bps *= self.extended_spread_multiplier
        spread = notional * (bps / 10000.0)
        slippage = 2.0 * notional * (self.slippage_bps / 10000.0)

        regulatory = 0.0
        if self.include_regulatory:
            regulatory = notional * _SEC_FEE_RATE + min(
                shares * _FINRA_TAF_PER_SHARE, _FINRA_TAF_CAP
            )

        return RoundTripCost(notional, commission, spread, slippage, regulatory)

    def min_notional_for_cost(
        self,
        target_pct: float,
        price: float,
        spread_bps: Optional[float] = None,
        extended_session: bool = False,
        max_notional: float = 5_000_000.0,
    ) -> Optional[float]:
        if target_pct <= 0 or price <= 0:
            return None

        def cost_at(notional: float) -> float:
            shares = max(notional / price, 1e-9)
            return self.estimate(price, shares, spread_bps, extended_session).pct

        if cost_at(max_notional) > target_pct:
            return None

        lo, hi = price, max_notional
        if cost_at(lo) <= target_pct:
            return lo
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if cost_at(mid) > target_pct:
                lo = mid
            else:
                hi = mid
        return hi

    def min_shares_for_edge(
        self,
        edge_pct: float,
        price: float,
        spread_bps: Optional[float] = None,
        extended_session: bool = False,
    ) -> Optional[int]:
        """Minimum shares for this edge to be profitable, or None if no size works."""
        notional = self.min_notional_for_cost(
            edge_pct, price, spread_bps, extended_session
        )
        if notional is None or price <= 0:
            return None
        return max(int(math.ceil(notional / price)), 1)

    def is_viable(
        self,
        edge_pct: float,
        price: float,
        shares: float,
        spread_bps: Optional[float] = None,
        extended_session: bool = False,
        safety_multiple: float = 1.5,
    ) -> tuple:
        """
        Does this order's predicted edge cover its cost?

        safety_multiple requires the edge to be a multiple of the cost: a break-even trade
        has zero expected value, and negative after forecast error. Default 1.5.

        Returns (viable, RoundTripCost, explanation)
        """
        cost = self.estimate(price, shares, spread_bps, extended_session)
        if cost.notional <= 0:
            return False, cost, "zero notional; cannot assess"

        required = cost.pct * safety_multiple
        edge = abs(float(edge_pct))
        if edge >= required:
            return (
                True,
                cost,
                (
                    f"edge {edge * 100:.3f}% ≥ required {required * 100:.3f}% "
                    f"(cost {cost.pct * 100:.3f}% × {safety_multiple:g})"
                ),
            )

        need = self.min_shares_for_edge(
            edge / safety_multiple, price, spread_bps, extended_session
        )
        hint = (
            f"this edge needs ≥ {need} shares (about ${need * price:,.0f})"
            if need
            else "no position size covers the cost"
        )
        return (
            False,
            cost,
            (
                f"edge {edge * 100:.3f}% < required {required * 100:.3f}%. "
                f"{cost.explain()}. {hint}"
            ),
        )


def build_execution_cost_model(config=None) -> ExecutionCostModel:
    """Composition root."""
    s = {}
    if config is not None:
        try:
            s = config.get("execution_cost_settings") or {}
        except Exception:
            s = {}
        if not isinstance(s, dict):
            s = {}

    name = str(s.get("commission_model", "ibkr_tiered")).lower()
    cls = _COMMISSION_MODELS.get(name)
    if cls is None:
        logger.error(
            f"Unknown commission model {name!r}; falling back to ibkr_tiered (the lower, more conservative estimate)."
        )
        cls = IBKRTieredCommission

    if cls is ZeroCommission:
        commission = ZeroCommission()
    else:
        kwargs = {}
        if s.get("per_share") is not None:
            kwargs["per_share"] = float(s["per_share"])
        if s.get("minimum_per_order") is not None:
            kwargs["minimum"] = float(s["minimum_per_order"])
        commission = cls(**kwargs)

    return ExecutionCostModel(
        commission=commission,
        default_spread_bps=float(s.get("default_spread_bps", 2.0)),
        slippage_bps=float(s.get("slippage_bps", 1.0)),
        extended_spread_multiplier=float(s.get("extended_spread_multiplier", 4.0)),
        include_regulatory=bool(s.get("include_regulatory_fees", True)),
    )
