# ibkrpy/strategy/order_intent.py
# Pure order decision shared by the live engine, the backtest and the tuner

from dataclasses import dataclass
from typing import Optional

MAX_POSITION_WEIGHT = 0.35
CASH_BUFFER = 0.95


@dataclass(frozen=True)
class OrderSnapshot:
    """Everything the order decision reads, captured at one point in time.

    Prices and account values are plain numbers, so the same snapshot always gives the
    same intent: no broker calls, clocks or config lookups happen inside decide_order.
    """

    action: str
    current_pos: float
    price: float
    net_liquidation: float
    available_funds: float
    gross_exposure: float = 0.0
    take_profit_price: Optional[float] = None
    entries_allowed: bool = True
    allow_shorting: bool = False
    target_weight: float = 0.08
    conviction: float = 1.0
    max_gross_exposure: float = 1.0
    min_trade_usd: float = 0.0
    extended_session: bool = False
    safety_multiple: float = 1.5


@dataclass(frozen=True)
class Intent:
    """What to do with one signal. quantity is 0 when reason_code is a skip code."""

    action: str
    quantity: int
    closing: bool
    weight: float
    reason_code: str
    message: str = ""

    @property
    def is_order(self) -> bool:
        return self.quantity > 0 and self.reason_code.startswith("ORDER_")


def is_reducing(action: str, current_pos: float) -> bool:
    return (current_pos > 0 and action == "SELL") or (
        current_pos < 0 and action == "BUY"
    )


def final_weight(target_weight: float, conviction: float) -> float:
    return min(target_weight * conviction, MAX_POSITION_WEIGHT)


def _skip(snap: OrderSnapshot, weight: float, code: str, message: str) -> Intent:
    return Intent(snap.action, 0, False, weight, code, message)


def _affordable_qty(snap: OrderSnapshot, weight: float) -> tuple:
    """(shares, why) for a new position; why is set when the gross cap stops it."""
    budget = min(snap.net_liquidation * weight, snap.available_funds * CASH_BUFFER)
    if snap.net_liquidation > 0:
        room = snap.max_gross_exposure * snap.net_liquidation - snap.gross_exposure
        if room <= 0:
            return 0, (
                f"Portfolio gross exposure cap reached "
                f"({snap.max_gross_exposure * 100:.0f}%); no new position."
            )
        budget = min(budget, room)
    qty = int(budget / snap.price) if snap.price > 0 else 0
    return qty, ""


def decide_order(snap: OrderSnapshot, cost_model=None) -> Intent:
    """
    Turn a strategy signal into an order intent.

    Rules, in order:
      - with entries blocked, only an order that reduces the position goes through;
      - an opposite signal closes the whole position and never flips it;
      - a signal in the direction already held does not add;
      - a new short needs allow_shorting;
      - a new position is sized at min(weight × net liquidation, 95% of available funds,
        room left under the gross exposure cap);
      - a new position must clear the execution cost check and min_trade_usd.
    Closing orders skip the cost and minimum-size checks: an exit is never blocked.

    cost_model is an ExecutionCostModel or None (no cost check).
    """
    action = snap.action
    pos = snap.current_pos
    weight = final_weight(snap.target_weight, snap.conviction)

    if not snap.entries_allowed and not is_reducing(action, pos):
        return _skip(
            snap,
            weight,
            "RISK_ENTRIES_BLOCKED",
            "Signal valid but risk gate closed; skipping new exposure.",
        )

    if action == "BUY" and pos > 0:
        return _skip(
            snap,
            weight,
            "ALREADY_LONG",
            f"Already long {pos:+.0f} shares; BUY signal does not add.",
        )
    if action == "SELL" and pos < 0:
        return _skip(
            snap,
            weight,
            "ALREADY_SHORT",
            f"Already short {pos:+.0f} shares; SELL signal does not add.",
        )

    if is_reducing(action, pos):
        return Intent(
            action, int(abs(pos)), True, weight, f"ORDER_{action}", "Close position"
        )

    if action == "SELL" and not snap.allow_shorting:
        return _skip(
            snap,
            weight,
            "SHORTING_DISABLED",
            "SELL signal with no position and allow_shorting=False; short blocked.",
        )

    qty, why = _affordable_qty(snap, weight)
    if qty <= 0:
        if why:
            return _skip(snap, weight, "GROSS_EXPOSURE_CAP", why)
        budget = min(snap.net_liquidation * weight, snap.available_funds * CASH_BUFFER)
        return _skip(
            snap,
            weight,
            "NO_BUYING_POWER",
            f"{action} signal but 0 shares affordable. "
            f"Target weight {weight * 100:.1f}% / net liq ${snap.net_liquidation:,.0f} / "
            f"available ${snap.available_funds:,.0f} / budget ${budget:,.0f} / "
            f"price {snap.price:.2f}.",
        )

    if cost_model is not None:
        try:
            gross_edge = abs(float(snap.take_profit_price) - snap.price) / snap.price
        except (TypeError, ValueError, ZeroDivisionError):
            gross_edge = 0.0
        viable, _, why = cost_model.is_viable(
            edge_pct=gross_edge,
            price=snap.price,
            shares=qty,
            extended_session=snap.extended_session,
            safety_multiple=snap.safety_multiple,
        )
        if not viable:
            return _skip(
                snap,
                weight,
                "COST_NOT_VIABLE",
                f"{action} signal cancelled: costs exceed the edge. {why}",
            )

    value = qty * snap.price
    if value < snap.min_trade_usd:
        return _skip(
            snap,
            weight,
            "BELOW_MIN_TRADE",
            f"{action} signal cancelled: trade value ${value:.2f} is below the "
            f"minimum ${snap.min_trade_usd:.2f}, to avoid fee drag.",
        )

    return Intent(action, qty, False, weight, f"ORDER_{action}", "Open position")


def regime_size_scale(
    regime, reversal_risk: float, reducing: bool, config=None
) -> float:
    """Scale position size by regime and reversal risk.

    Reducing and closing always return 1.0: shrinking an exit leaves a tail that cannot
    be handled.

    A confirmed bull regime scales up, so the system is not too conservative in bull
    markets; rising reversal risk scales down linearly, so positions exit while a top
    forms instead of waiting for the regime to turn bearish.
    """
    if reducing:
        return 1.0

    from ibkrpy.strategy.strategy_components import MarketRegime

    def get(key, default):
        return config.get(key, default) if config is not None else default

    scales = {
        MarketRegime.BULL_TREND: get("strategy_settings.size_scale_bull", 1.25),
        MarketRegime.BEAR_TREND: get("strategy_settings.size_scale_bear", 0.60),
        MarketRegime.SIDEWAYS_VOLATILE: get(
            "strategy_settings.size_scale_volatile", 0.70
        ),
        MarketRegime.SIDEWAYS_QUIET: get("strategy_settings.size_scale_quiet", 0.90),
    }
    scale = float(scales.get(regime, 1.0))
    scale *= max(1.0 - reversal_risk * 0.75, 0.25)
    return scale


def cost_floor_pct(
    cost_model, price: float, net_liquidation: float, base_pct: float, safety: float
) -> float:
    """
    Round-trip cost as a fraction of the notional a base-size position would trade.

    Share count is unknown at the threshold stage, so base_pct × net liquidation stands in
    for notional. This is a coarse screen; decide_order runs the real check on the actual
    share count. It still matters: exit geometry uses it as the take-profit floor, so a
    target is never set inside the cost band.
    """
    notional = max(float(net_liquidation) * base_pct, float(price))
    shares = max(notional / float(price), 1.0)
    return cost_model.estimate(price, shares).pct * safety
