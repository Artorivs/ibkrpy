"""Equal-weight ensemble; liveness is only a yes/no collapse check."""

import unittest
from collections import defaultdict, deque

import numpy as np

from ibkrpy.manager.trading_engine import TradingEngine
from ibkrpy.strategy.core_strategy import CoreStrategy
from ibkrpy.strategy.strategy_components import MarketRegime


class EqualWeightTests(unittest.TestCase):
    def setUp(self):
        self.strategy = CoreStrategy("TEST", {})

    def test_mean_is_equal_weight_in_every_regime(self):
        preds = {"ARIMA": 100.0, "LSTM": 103.0, "Transformer": 106.0}
        for regime in MarketRegime:
            value = self.strategy._calculate_ensemble_prediction(preds, regime)
            self.assertAlmostEqual(value, 103.0, msg=regime)

    def test_collapsed_model_is_left_out(self):
        self.strategy.set_model_liveness(
            {"ARIMA": 1.0, "LSTM": 0.0, "Transformer": 1.0}
        )
        preds = {"ARIMA": 100.0, "LSTM": 150.0, "Transformer": 104.0}
        value = self.strategy._calculate_ensemble_prediction(
            preds, MarketRegime.BULL_TREND
        )
        self.assertAlmostEqual(value, 102.0)
        self.assertEqual(self.strategy._last_filtered_models, ["ARIMA", "Transformer"])

    def test_all_collapsed_falls_back_to_all_models(self):
        self.strategy.set_model_liveness({"ARIMA": 0.0, "LSTM": 0.0})
        value = self.strategy._calculate_ensemble_prediction(
            {"ARIMA": 100.0, "LSTM": 102.0}, MarketRegime.SIDEWAYS_QUIET
        )
        self.assertAlmostEqual(value, 101.0)


class LivenessTests(unittest.TestCase):
    def engine(self):
        e = TradingEngine.__new__(TradingEngine)
        e._model_pred_hist = defaultdict(lambda: defaultdict(lambda: deque(maxlen=120)))
        return e

    def feed(self, e, series):
        out = None
        for i in range(len(next(iter(series.values())))):
            out = e._update_model_liveness("TEST", {k: v[i] for k, v in series.items()})
        return out

    def test_liveness_is_binary_and_does_not_reward_noise(self):
        rng = np.random.default_rng(0)
        out = self.feed(
            self.engine(),
            {
                "ARIMA": rng.normal(0, 0.001, 20),
                "LSTM": rng.normal(0, 0.010, 20),
            },
        )
        self.assertEqual(out, {"ARIMA": 1.0, "LSTM": 1.0})

    def test_constant_model_is_marked_collapsed(self):
        rng = np.random.default_rng(1)
        out = self.feed(
            self.engine(),
            {"ARIMA": rng.normal(0, 0.01, 20), "LSTM": np.full(20, 0.002)},
        )
        self.assertEqual(out["LSTM"], 0.0)
        self.assertEqual(out["ARIMA"], 1.0)

    def test_too_few_samples_counts_as_alive(self):
        out = self.feed(self.engine(), {"ARIMA": [0.01, 0.02], "LSTM": [0.0, 0.0]})
        self.assertEqual(out, {"ARIMA": 1.0, "LSTM": 1.0})


if __name__ == "__main__":
    unittest.main()
