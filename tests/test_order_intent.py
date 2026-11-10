"""The order decision is one pure function shared by live, the backtest and the tuner."""

import asyncio
import dataclasses
import types
import unittest

import numpy as np
import pandas as pd

from ibkrpy.evaluation import backtest_engine
from ibkrpy.evaluation.backtest_engine import BacktestEngine
from ibkrpy.manager import trading_engine
from ibkrpy.manager.trading_engine import TradingEngine
from ibkrpy.strategy.execution_costs import build_execution_cost_model
from ibkrpy.strategy.order_intent import (
    Intent,
    OrderSnapshot,
    decide_order,
    regime_size_scale,
)
from ibkrpy.strategy.strategy_components import MarketRegime


def snap(**kw):
    base = dict(
        action="BUY",
        current_pos=0.0,
        price=100.0,
        net_liquidation=100_000.0,
        available_funds=100_000.0,
        target_weight=0.10,
        allow_shorting=True,
    )
    base.update(kw)
    return OrderSnapshot(**base)


class NeverViable:
    def is_viable(self, **kw):
        return False, None, "too expensive"


class DecideOrderTests(unittest.TestCase):
    def test_same_snapshot_gives_same_intent(self):
        s = snap()
        self.assertEqual(decide_order(s), decide_order(s))

    def test_snapshot_is_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snap().price = 1.0

    def test_new_position_sized_at_weight_of_net_liquidation(self):
        intent = decide_order(snap())
        self.assertEqual((intent.reason_code, intent.quantity), ("ORDER_BUY", 100))
        self.assertFalse(intent.closing)

    def test_weight_is_capped(self):
        intent = decide_order(snap(target_weight=0.5, conviction=2.0))
        self.assertAlmostEqual(intent.weight, 0.35)
        self.assertEqual(intent.quantity, 350)

    def test_budget_limited_by_available_funds(self):
        intent = decide_order(snap(available_funds=5_000.0))
        self.assertEqual(intent.quantity, 47)  # 95% of 5,000 at $100

    def test_budget_limited_by_gross_exposure_room(self):
        intent = decide_order(snap(gross_exposure=97_000.0))
        self.assertEqual(intent.quantity, 30)

    def test_gross_cap_reached(self):
        intent = decide_order(snap(gross_exposure=100_000.0))
        self.assertEqual(intent.reason_code, "GROSS_EXPOSURE_CAP")
        self.assertFalse(intent.is_order)

    def test_no_buying_power(self):
        intent = decide_order(snap(available_funds=50.0))
        self.assertEqual(intent.reason_code, "NO_BUYING_POWER")

    def test_risk_gate_blocks_entries(self):
        intent = decide_order(snap(entries_allowed=False))
        self.assertEqual(intent.reason_code, "RISK_ENTRIES_BLOCKED")

    def test_risk_gate_never_blocks_exits(self):
        intent = decide_order(
            snap(action="SELL", current_pos=40, entries_allowed=False)
        )
        self.assertEqual((intent.reason_code, intent.quantity), ("ORDER_SELL", 40))
        self.assertTrue(intent.closing)

    def test_same_direction_does_not_add(self):
        self.assertEqual(decide_order(snap(current_pos=10)).reason_code, "ALREADY_LONG")
        self.assertEqual(
            decide_order(snap(action="SELL", current_pos=-10)).reason_code,
            "ALREADY_SHORT",
        )

    def test_opposite_signal_closes_whole_position_without_flipping(self):
        intent = decide_order(snap(action="BUY", current_pos=-25))
        self.assertEqual((intent.quantity, intent.closing), (25, True))

    def test_short_needs_allow_shorting(self):
        intent = decide_order(snap(action="SELL", allow_shorting=False))
        self.assertEqual(intent.reason_code, "SHORTING_DISABLED")

    def test_entry_must_clear_cost_check(self):
        intent = decide_order(snap(take_profit_price=101.0), NeverViable())
        self.assertEqual(intent.reason_code, "COST_NOT_VIABLE")

    def test_live_cost_model_rejects_thin_edge_and_accepts_wide_one(self):
        model = build_execution_cost_model(None)
        thin = decide_order(snap(take_profit_price=100.01), model)
        wide = decide_order(snap(take_profit_price=103.0), model)
        self.assertEqual(thin.reason_code, "COST_NOT_VIABLE")
        self.assertEqual(wide.reason_code, "ORDER_BUY")

    def test_entry_must_clear_min_trade(self):
        intent = decide_order(snap(target_weight=0.001, min_trade_usd=500.0))
        self.assertEqual(intent.reason_code, "BELOW_MIN_TRADE")

    def test_close_skips_cost_and_min_trade(self):
        intent = decide_order(
            snap(action="SELL", current_pos=1, min_trade_usd=1e9), NeverViable()
        )
        self.assertEqual((intent.reason_code, intent.quantity), ("ORDER_SELL", 1))


class RegimeScaleTests(unittest.TestCase):
    def test_reducing_is_never_scaled(self):
        self.assertEqual(regime_size_scale(MarketRegime.BEAR_TREND, 0.9, True), 1.0)

    def test_bull_scales_up_and_reversal_risk_scales_down(self):
        bull = regime_size_scale(MarketRegime.BULL_TREND, 0.0, False)
        risky = regime_size_scale(MarketRegime.BULL_TREND, 0.8, False)
        self.assertAlmostEqual(bull, 1.25)
        self.assertLess(risky, bull)


def bars(closes):
    idx = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c}, index=idx)


def replay(signals):
    seen = []

    def decide(ts, row, position, equity):
        seen.append(position)
        return signals.get(len(seen) - 1)

    return decide, seen


class SharedWithBacktestTests(unittest.TestCase):
    def engine(self, **kw):
        kw.setdefault("position_pct", 0.1)
        kw.setdefault("slippage_pct", 0.0)
        kw.setdefault("commission_per_share", 0.0)
        return BacktestEngine(initial_capital=100_000.0, **kw)

    def test_live_and_backtest_call_the_same_function(self):
        self.assertIs(trading_engine.decide_order, decide_order)
        self.assertIs(backtest_engine.decide_order, decide_order)

    def test_backtest_honours_min_trade(self):
        decide, seen = replay({0: {"action": "BUY"}})
        self.engine(position_pct=0.001, min_trade_usd=500.0).simulate(
            bars([100.0] * 3), decide
        )
        self.assertEqual(seen[1], 0)

    def test_backtest_uses_signal_target_weight(self):
        decide, seen = replay({0: {"action": "BUY", "target_weight": 0.05}})
        self.engine().simulate(bars([100.0] * 3), decide)
        self.assertEqual(seen[1], 50)

    def test_backtest_blocks_entries_but_not_exits_when_gate_closed(self):
        decide, seen = replay(
            {
                0: {"action": "BUY"},
                1: {"action": "SELL", "entries_allowed": False},
                2: {"action": "BUY", "entries_allowed": False},
            }
        )
        self.engine().simulate(bars([100.0] * 4), decide)
        self.assertEqual(seen, [0, 100, 0, 0])


def make_engine(dry_run=True):
    eng = TradingEngine.__new__(TradingEngine)
    eng.dry_run = dry_run
    eng.db = None
    eng.cached_funds = 50_000.0
    eng._cycle_decisions = {}

    async def contract(symbol):
        return types.SimpleNamespace(symbol=symbol)

    eng._get_qualified_contract = contract
    return eng


SIGNAL = {
    "action": "BUY",
    "stop_loss_price": 95.0,
    "take_profit_price": 110.0,
    "regime": "BULL_TREND",
}


class ExecuteIntentTests(unittest.TestCase):
    def test_skip_intent_only_records_reason(self):
        eng = make_engine()
        intent = Intent("BUY", 0, False, 0.1, "BELOW_MIN_TRADE", "too small")
        asyncio.run(eng._execute_intent("NVDA", intent, SIGNAL, 100.0))
        self.assertEqual(eng._cycle_decisions, {"BELOW_MIN_TRADE": 1})
        self.assertEqual(eng.cached_funds, 50_000.0)

    def test_dry_run_entry_reserves_funds(self):
        eng = make_engine()
        intent = Intent("BUY", 100, False, 0.1, "ORDER_BUY")
        asyncio.run(eng._execute_intent("NVDA", intent, SIGNAL, 100.0))
        self.assertEqual(eng._cycle_decisions, {"ORDER_BUY": 1})
        self.assertEqual(eng.cached_funds, 40_000.0)


class RunTickWiringTests(unittest.TestCase):
    """Drives run_tick through every stage with stubs, in dry run."""

    def make(self, entries_allowed=True, position=0.0):
        eng = make_engine()
        eng._last_tick_at, eng._last_prices = {}, {}
        eng._collapse_warned = set()
        eng.cached_positions = {"NVDA": position}
        eng.cached_net_liq = 100_000.0
        eng._pending_gross = 0.0
        eng._position_avg_cost = {}
        eng._live_market_price = lambda s: None
        eng.config = types.SimpleNamespace(
            get=lambda k, d=None: {
                "strategy_settings.min_trade_usd": 0.0,
                "strategy_settings.allow_shorting": True,
            }.get(k, d)
        )
        eng.ext = eng.pipeline = eng.regime_detector = eng.market_analyzer = None
        eng._effective_term = lambda s: "long_term"
        close = np.linspace(100.0, 110.0, 80)

        async def fetch(**kw):
            idx = pd.date_range("2024-01-02", periods=80, freq="B", tz="UTC")
            return pd.DataFrame(
                {"open": close, "high": close, "low": close, "close": close},
                index=idx,
            ).assign(volume=1.0)

        eng.data = types.SimpleNamespace(fetch_historical_data=fetch)
        eng._get_dynamic_benchmark = lambda s: "SPY"

        async def bench(*a):
            return pd.DataFrame()

        eng._load_benchmark_frame = bench
        eng.session_clock = types.SimpleNamespace(
            classify=lambda: types.SimpleNamespace(is_extended=False)
        )
        eng.equity_watermark = types.SimpleNamespace(snapshot=lambda nlv: {})
        eng.risk = types.SimpleNamespace(
            check_entries_allowed=lambda ctx: (entries_allowed, "test gate"),
            check_trade_allowed=None,
        )
        eng._collect_predictions = lambda *a: ({"ARIMA": 115.0}, {})
        eng._log_predictions = lambda *a: None
        eng._update_model_liveness = lambda s, p: {"ARIMA": 1.0}
        eng.models = types.SimpleNamespace(predict=lambda *a, **k: (None, 0.01))
        eng.vol_estimator = types.SimpleNamespace(
            estimate=lambda *a: types.SimpleNamespace(
                value=0.01, source="model", realized=0.01, ratio=1.0
            )
        )
        eng.cost_model = build_execution_cost_model(None)
        eng._record_to_ledger = lambda *a: None
        eng.strategies = {
            "NVDA": types.SimpleNamespace(
                min_edge_pct=0.0,
                last_decision={"reason_code": "SIGNAL"},
                generate_signal=lambda **kw: dict(SIGNAL, take_profit_price=120.0),
            )
        }
        return eng

    def test_entry_reaches_execution(self):
        eng = self.make()
        asyncio.run(eng.run_tick("NVDA"))
        self.assertEqual(eng._cycle_decisions.get("ORDER_BUY"), 1)
        self.assertLess(eng.cached_funds, 50_000.0)

    def test_closed_gate_stops_entry_before_sizing(self):
        eng = self.make(entries_allowed=False)
        asyncio.run(eng.run_tick("NVDA"))
        self.assertEqual(eng._cycle_decisions.get("RISK_ENTRIES_BLOCKED"), 2)
        self.assertNotIn("ORDER_BUY", eng._cycle_decisions)

    def test_closed_gate_still_closes_short(self):
        eng = self.make(entries_allowed=False, position=-30.0)
        asyncio.run(eng.run_tick("NVDA"))
        self.assertEqual(eng._cycle_decisions.get("ORDER_BUY"), 1)
        self.assertEqual(eng.cached_funds, 50_000.0)


if __name__ == "__main__":
    unittest.main()
