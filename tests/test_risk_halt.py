"""Portfolio kill switch and intra-loop gross exposure."""

import os
import tempfile
import types
import unittest

from ibkrpy.manager.trading_engine import TradingEngine
from ibkrpy.strategy.strategy_components import (
    DrawdownHaltRule,
    EquityWatermark,
    RiskController,
)


class DrawdownHaltTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "risk_state.json")
        self.rule = DrawdownHaltRule(max_daily_loss_pct=0.02, max_drawdown_pct=0.10)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, wm, nlv):
        return RiskController([self.rule]).check_entries_allowed(wm.snapshot(nlv))

    def test_allows_normal_day(self):
        wm = EquityWatermark(self.path)
        wm.update(100_000, "2026-11-09")
        allowed, _ = self.check(wm, 99_000)
        self.assertTrue(allowed)

    def test_daily_loss_blocks_and_stays_latched_for_the_day(self):
        wm = EquityWatermark(self.path)
        wm.update(100_000, "2026-11-09")
        self.assertFalse(self.check(wm, 97_900)[0])
        wm.update(99_500, "2026-11-09")  # partial recovery, same day
        self.assertFalse(self.check(wm, 99_500)[0])
        wm.update(99_500, "2026-11-10")  # next day resets the day start
        self.assertTrue(self.check(wm, 99_500)[0])

    def test_drawdown_from_peak_blocks_until_recovered(self):
        wm = EquityWatermark(self.path)
        wm.update(120_000, "2026-11-01")
        wm.update(107_000, "2026-11-09")
        self.assertFalse(self.check(wm, 107_000)[0])  # 10.8% below peak
        wm.update(109_000, "2026-11-10")
        self.assertTrue(self.check(wm, 109_000)[0])  # 9.2% below peak

    def test_state_survives_restart(self):
        wm = EquityWatermark(self.path)
        wm.update(120_000, "2026-11-01")
        wm.update(100_000, "2026-11-09")
        reloaded = EquityWatermark(self.path)
        self.assertEqual(reloaded.peak, 120_000)
        self.assertEqual(reloaded.day_start, 100_000)
        reloaded.update(97_000, "2026-11-09")  # restart mid-day keeps the day start
        self.assertEqual(reloaded.day_start, 100_000)

    def test_missing_nlv_does_not_block(self):
        wm = EquityWatermark(self.path)
        self.assertTrue(self.check(wm, 0.0)[0])


class GrossExposureTests(unittest.TestCase):
    def make_engine(self):
        eng = TradingEngine.__new__(TradingEngine)
        eng.cached_positions = {"AAA": 100, "BBB": -50}
        eng._last_prices = {"AAA": 10.0}
        eng._position_avg_cost = {"AAA": 9.0, "BBB": 20.0}
        eng._pending_gross = 0.0
        eng.data = types.SimpleNamespace(ib=types.SimpleNamespace(portfolio=lambda: []))
        return eng

    def test_untouched_positions_count_at_avg_cost(self):
        eng = self.make_engine()
        # AAA at last tick price 10 x 100, BBB not ticked -> avg cost 20 x 50
        self.assertAlmostEqual(eng._current_gross_exposure(), 2000.0)

    def test_orders_sent_this_loop_count_toward_gross(self):
        eng = self.make_engine()
        eng._pending_gross = 5000.0
        self.assertAlmostEqual(eng._current_gross_exposure(), 7000.0)


if __name__ == "__main__":
    unittest.main()
