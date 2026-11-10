"""The tuner and backtest trade the way live does, and score on unseen data."""

import types
import unittest

import numpy as np
import pandas as pd

from ibkrpy.evaluation.backtest_engine import BacktestEngine
from ibkrpy.evaluation.model_tuner import TUNED_PARAMS, ModelTuner, _DictConfig


def bars(closes, opens=None, highs=None, lows=None):
    idx = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes if opens is None else opens,
            "High": closes if highs is None else highs,
            "Low": closes if lows is None else lows,
            "Close": closes,
        },
        index=idx,
    )


def scripted(actions):
    """decide() that replays actions by bar number and records the positions it saw."""
    seen = []

    def decide(ts, row, position, equity):
        seen.append(position)
        a = actions.get(len(seen) - 1)
        return {"action": a} if a else None

    return decide, seen


class BacktestMirrorsLiveTests(unittest.TestCase):
    def engine(self, **kw):
        kw.setdefault("position_pct", 0.1)
        kw.setdefault("slippage_pct", 0.0)
        kw.setdefault("commission_per_share", 0.0)
        return BacktestEngine(initial_capital=100000.0, **kw)

    def test_entry_is_sized_at_position_pct(self):
        decide, seen = scripted({0: "BUY"})
        self.engine().simulate(bars([100.0] * 3), decide)
        self.assertEqual(seen[1], 100)  # 10% of 100k at $100

    def test_opposite_signal_closes_without_flipping(self):
        decide, seen = scripted({0: "BUY", 1: "SELL"})
        self.engine().simulate(bars([100.0] * 4), decide)
        self.assertEqual(seen[1], 100)
        self.assertEqual(seen[2], 0)
        self.assertEqual(seen[3], 0)

    def test_same_direction_signal_does_not_add(self):
        decide, seen = scripted({0: "BUY", 1: "BUY"})
        self.engine().simulate(bars([100.0] * 3), decide)
        self.assertEqual(seen[2], 100)

    def test_no_short_when_shorting_disabled(self):
        decide, seen = scripted({0: "SELL"})
        self.engine(allow_shorting=False).simulate(bars([100.0] * 3), decide)
        self.assertEqual(seen[1], 0)

    def test_gap_through_stop_fills_at_open(self):
        def decide(ts, row, position, equity):
            if position == 0 and ts == df.index[0]:
                return {"action": "BUY", "stop_loss_price": 95.0}
            return None

        df = bars(
            [100.0, 85.0, 85.0],
            opens=[100.0, 80.0, 85.0],
            highs=[100.0, 86.0, 85.0],
            lows=[100.0, 79.0, 85.0],
        )
        perf = self.engine().simulate(df, decide)
        # 100 shares bought at 100, stopped at the 80 open, not at 95.
        self.assertAlmostEqual(perf["final_equity"], 100000.0 - 100 * 20, places=2)

    def test_cost_model_is_charged_per_fill(self):
        class FlatCost:
            def estimate(self, price, shares):
                return types.SimpleNamespace(total=10.0)

        decide, _ = scripted({0: "BUY", 1: "SELL"})
        perf = self.engine(cost_model=FlatCost()).simulate(bars([100.0] * 3), decide)
        self.assertAlmostEqual(perf["final_equity"], 100000.0 - 10.0, places=2)


class TunerTests(unittest.TestCase):
    def test_only_parameters_live_reads_are_tuned(self):
        self.assertNotIn("min_prediction_threshold_pct", TUNED_PARAMS)
        self.assertEqual(
            set(TUNED_PARAMS),
            {"volatility_stop_loss_multiplier", "volatility_take_profit_multiplier"},
        )

    def test_search_uses_tune_half_and_score_uses_validate_half(self):
        rows = bars(np.linspace(100, 110, 100))
        rows["prediction"] = rows["Close"] * 1.01
        rows["volatility"] = 0.01
        rows["regime"] = "SIDEWAYS_QUIET"

        tuner = ModelTuner(config=_DictConfig())
        calls = []

        def fake_evaluate(symbol, segment, params, term, warmup=None):
            calls.append((segment.index.min(), segment.index.max(), warmup))
            return {"sortino_ratio": 1.0, "total_trades": 10, "profit_factor": 2.0}

        tuner.evaluate = fake_evaluate
        tuner.optimize_strategy_params("TEST", rows, rows, n_trials=3)

        cut = rows.index[50]
        *search, final = calls
        self.assertEqual(len(search), 3)
        for lo, hi, warmup in search:
            self.assertLess(hi, cut)
            self.assertIsNone(warmup)
        self.assertEqual(final[0], cut)
        self.assertEqual(len(final[2]), 50)

    def test_evaluate_runs_the_live_strategy_end_to_end(self):
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
        rows = bars(close, highs=close * 1.005, lows=close * 0.995)
        rows["prediction"] = np.r_[close[1:], close[-1]]  # perfect next-bar foresight
        rows["volatility"] = 0.01
        rows["regime"] = "SIDEWAYS_QUIET"

        tuner = ModelTuner(config=_DictConfig({"strategy_settings": {}}))
        perf = tuner.evaluate(
            "TEST",
            rows,
            {
                "volatility_stop_loss_multiplier": 1.0,
                "volatility_take_profit_multiplier": 2.0,
            },
            "long_term",
        )
        self.assertGreater(perf["total_trades"], 0)


class WalkForwardAlignmentTests(unittest.TestCase):
    def test_prediction_at_row_t_uses_close_t(self):
        from ibkrpy.manager.pipeline_manager import PipelineManager

        n, jump_at = 300, 250
        close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, n))
        close[jump_at:] += 100.0
        df = bars(close)

        fake = types.SimpleNamespace(
            config=_DictConfig(
                {
                    "tuning_settings": {
                        "walk_forward_models": ["ARIMA"],
                        "oos_fraction": 0.3,
                    }
                }
            ),
            _vectorised_regimes=lambda d: pd.Series("SIDEWAYS_QUIET", index=d.index),
        )
        oos = PipelineManager._walk_forward_predictions(fake, "TEST", df, "long_term")

        # Live predicts bar t+1 after seeing Close[t]; the forecast on the jump bar
        # must already know about the jump.
        self.assertGreater(oos.loc[df.index[jump_at], "prediction"], 150.0)


if __name__ == "__main__":
    unittest.main()
