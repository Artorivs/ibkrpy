# ibkrpy/strategy/strategy_components.py
# 將原本分散的組件集中，降低複雜度，提高內聚性

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("ibkrpy.strategy")


class MarketRegime(Enum):
    """市場情境。四種狀態，全系統唯一定義處。"""

    BULL_TREND = auto()
    BEAR_TREND = auto()
    SIDEWAYS_VOLATILE = auto()
    SIDEWAYS_QUIET = auto()


class IRiskRule(ABC):
    """風險規則。

    契約：assess 回傳 (是否允許「新增曝險」, 理由)。

    請注意「新增曝險」這個限定詞。風險規則不得、也不應該阻止減碼或平倉：
    風控的目的是限制風險敞口，而在風險最高的時刻剝奪處置風險的能力，
    與這個目的正好相反。
    """

    @abstractmethod
    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class VIXHaltRule(IRiskRule):
    """VIX 高於門檻時停止新增曝險。"""

    def __init__(self, threshold: float = 30.0):
        self.threshold = float(threshold)

    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        vix_series = context.get("vix_series")
        if vix_series is None or len(vix_series) == 0:
            return True, "VIX 資料缺席，不阻擋"

        latest = float(vix_series.iloc[-1])
        if latest > self.threshold:
            return False, f"VIX {latest:.2f} 高於門檻 {self.threshold:.0f}。"
        return True, f"VIX {latest:.2f} 正常"


class ReversalRiskRule(IRiskRule):
    """頂部反轉風險過高時停止新增曝險。

    與 CoreStrategy 內的 reversal_block_level 是兩道獨立的閘門：
    策略層看的是單一標的的訊號品質，這裡看的是投資組合層級的進場許可。
    兩者都通過才會開新倉。
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = float(threshold)

    def assess(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        risk = context.get("reversal_risk")
        if risk is None:
            return True, "無反轉風險資料"
        if float(risk) >= self.threshold:
            return (
                False,
                f"頂部反轉風險 {float(risk):.0%} 高於門檻 {self.threshold:.0%}。",
            )
        return True, f"反轉風險 {float(risk):.0%} 可接受"


class RiskController:
    """依序套用所有風險規則。任一規則否決即停止新增曝險。"""

    def __init__(self, rules: List[IRiskRule]):
        self.rules = list(rules)

    def check_entries_allowed(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        for rule in self.rules:
            try:
                allowed, reason = rule.assess(context)
            except Exception as exc:
                logger.error(f"風險規則 {rule.name} 拋出例外，保守起見視為否決: {exc}")
                return False, f"{rule.name} 評估失敗"
            if not allowed:
                return False, reason
        return True, "風險檢查通過"

    def check_trade_allowed(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """相容舊呼叫端的別名。

        名稱容易被誤讀為「所有交易」，新程式碼請改用 check_entries_allowed。
        """
        return self.check_entries_allowed(context)


class PositionSizer:
    """由風險預算換算股數。

    兩種模式擇一：給定每股風險時以風險預算計算，否則以資金比例計算。
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
