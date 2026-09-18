
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
    """單邊 (一次下單) 的佣金。"""

    @abstractmethod
    def per_order(self, shares: float, price: float) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class ZeroCommission(CommissionModel):
    """IBKR Lite (僅限美國客戶的美股)。"""

    name = "zero"

    def per_order(self, shares: float, price: float) -> float:
        return 0.0


class _PerShareWithMinimum(CommissionModel):
    """
    每股計價 + 每筆最低 + 名目上限。IBKR 股票佣金的通用形狀。
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
    """一次完整進出的成本拆解。金額單位為美元，比例為小數。"""

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
        """佔名目的比例 (小數)。這就是預測優勢必須超過的門檻。"""
        return self.total / self.notional if self.notional > 0 else float("inf")

    def explain(self) -> str:
        if self.notional <= 0:
            return "名目為零"
        p = lambda x: f"{x / self.notional * 100:.3f}%"
        return (
            f"來回成本 ${self.total:.2f} ({p(self.total)}) = "
            f"佣金 ${self.commission:.2f}({p(self.commission)}) + "
            f"價差 ${self.spread:.2f}({p(self.spread)}) + "
            f"滑價 ${self.slippage:.2f}({p(self.slippage)}) + "
            f"規費 ${self.regulatory:.2f}"
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
        """給定預測優勢，至少要幾股才可能有淨利。回傳 None 表示無解。"""
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
        這筆單的預測優勢是否足以覆蓋成本?

        safety_multiple 要求優勢是成本的數倍而非勉強打平 —— 剛好打平的交易
        期望值為零，扣掉預測誤差就是負的。預設 1.5 倍。

        回傳 (是否可行, RoundTripCost, 說明)
        """
        cost = self.estimate(price, shares, spread_bps, extended_session)
        if cost.notional <= 0:
            return False, cost, "名目為零，無法評估"

        required = cost.pct * safety_multiple
        edge = abs(float(edge_pct))
        if edge >= required:
            return (
                True,
                cost,
                (
                    f"優勢 {edge * 100:.3f}% ≥ 需求 {required * 100:.3f}% "
                    f"(成本 {cost.pct * 100:.3f}% × {safety_multiple:g})"
                ),
            )

        need = self.min_shares_for_edge(
            edge / safety_multiple, price, spread_bps, extended_session
        )
        hint = (
            f"若要讓這個優勢划算，需 ≥ {need} 股 (約 ${need * price:,.0f})"
            if need
            else "在任何部位大小下都無法覆蓋成本"
        )
        return (
            False,
            cost,
            (
                f"優勢 {edge * 100:.3f}% < 需求 {required * 100:.3f}%。"
                f"{cost.explain()}。{hint}"
            ),
        )


def build_execution_cost_model(config=None) -> ExecutionCostModel:
    """Composition Root 使用。"""
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
        logger.error(f"未知的佣金模型 {name!r}，退回 ibkr_tiered (較保守的較低估計)。")
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
