"""Models, walk-forward, live σ and the ledger all use the term's multi-bar horizon."""

import math
import types
import unittest

import numpy as np
import pandas as pd

from ibkrpy.data.data_pipeline import DataPipeline
from ibkrpy.evaluation.model_tuner import _DictConfig
from ibkrpy.shared.horizon import (
    DEFAULT_HORIZON_BARS,
    forecast_horizon,
    horizon_sigma,
)
from ibkrpy.strategy.core_strategy import CoreStrategy
from ibkrpy.strategy.prediction_calibrator import InMemoryPredictionHistory


class HorizonSettingTests(unittest.TestCase):
    def test_defaults_per_term(self):
        for term, bars in DEFAULT_HORIZON_BARS.items():
            self.assertEqual(forecast_horizon(None, term), bars)
        self.assertGreater(forecast_horizon(None, "long_term"), 1)

    def test_config_overrides_and_bad_values_fall_back(self):
        cfg = _DictConfig(
            {"forecast_settings": {"horizon_bars": {"long_term": 3, "mid_term": 0}}}
        )
        self.assertEqual(forecast_horizon(cfg, "long_term"), 3)
        self.assertEqual(forecast_horizon(cfg, "mid_term"), 1)
        self.assertEqual(forecast_horizon(cfg, "unknown"), 1)

    def test_sigma_scales_with_square_root_of_bars(self):
        self.assertAlmostEqual(horizon_sigma(0.01, 4), 0.02)
        self.assertAlmostEqual(horizon_sigma(0.01, 1), 0.01)


class SequenceTargetTests(unittest.TestCase):
    def test_target_is_log_return_over_horizon(self):
        close = np.linspace(100.0, 130.0, 40)
        df = pd.DataFrame({"Close": close, "f": np.arange(40.0)})
        pipe = DataPipeline.__new__(DataPipeline)
        X, y = pipe.create_sequences(
            df, ["f"], 10, raw_close=df["Close"], horizon=5, target_scale=100.0
        )
        self.assertEqual(len(X), 40 - 10 - 5 + 1)
        self.assertAlmostEqual(y[0], math.log(close[14] / close[9]) * 100.0)
        self.assertEqual(X[0][-1, 0], 9.0)

    def test_one_bar_horizon_unchanged(self):
        close = np.linspace(100.0, 130.0, 40)
        df = pd.DataFrame({"Close": close, "f": np.arange(40.0)})
        pipe = DataPipeline.__new__(DataPipeline)
        X, y = pipe.create_sequences(df, ["f"], 10, raw_close=df["Close"])
        self.assertEqual(len(X), 30)
        self.assertAlmostEqual(y[0], math.log(close[10] / close[9]) * 100.0)


def load_arima_module():
    """Load models/arima.py alone; the package __init__ imports TensorFlow."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "ibkrpy/models/arima.py"
    spec = importlib.util.spec_from_file_location("arima_alone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArimaHorizonTests(unittest.TestCase):
    def test_forecasts_the_trained_horizon(self):
        from statsmodels.tsa.arima.model import ARIMA

        ARIMAModel = load_arima_module().ARIMAModel

        rng = np.random.default_rng(3)
        close = 100 + np.cumsum(0.5 + rng.normal(0, 0.2, 200))
        res = ARIMA(close, order=(5, 1, 0)).fit()

        m = ARIMAModel()
        m.model_results = res
        m.horizon_bars = 5
        df = pd.DataFrame({"Close": close})
        expected = res.apply(close).forecast(steps=5)[4]
        self.assertAlmostEqual(m.predict_next_price(df), float(expected))
        m.horizon_bars = 1
        self.assertLess(m.predict_next_price(df), float(expected))


class HistoryKeyTests(unittest.TestCase):
    def test_each_horizon_keeps_its_own_history(self):
        hist = InMemoryPredictionHistory()
        s = CoreStrategy("SPY", {"horizon_bars": 1}, prediction_history=hist)
        s.generate_signal(current_price=100.0, prediction=100.1, volatility=0.01)
        s.horizon_bars = 5
        s.generate_signal(current_price=100.0, prediction=101.0, volatility=0.02)
        self.assertEqual(len(hist.get("SPY")), 1)
        self.assertEqual(len(hist.get("SPY@5")), 1)


class WalkForwardHorizonTests(unittest.TestCase):
    def test_arima_rows_forecast_h_bars_and_sigma_is_horizon_sigma(self):
        from ibkrpy.manager.pipeline_manager import PipelineManager

        rng = np.random.default_rng(1)
        n = 300
        close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, n))
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        df = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close}, index=idx
        )

        def run(h):
            fake = types.SimpleNamespace(
                config=_DictConfig(
                    {
                        "tuning_settings": {
                            "walk_forward_models": ["ARIMA"],
                            "oos_fraction": 0.3,
                        },
                        "forecast_settings": {"horizon_bars": {"long_term": h}},
                    }
                ),
                _vectorised_regimes=lambda d: pd.Series(
                    "SIDEWAYS_QUIET", index=d.index
                ),
            )
            return PipelineManager._walk_forward_predictions(
                fake, "TEST", df, "long_term"
            )

        from statsmodels.tsa.arima.model import ARIMA

        one, five = run(1), run(5)
        split = int(n * 0.7)
        base = ARIMA(df["Close"].values[:split], order=(5, 1, 0)).fit()
        t = split + 20
        expected = base.apply(df["Close"].values[: t + 1]).forecast(steps=5)[4]
        self.assertAlmostEqual(five.loc[idx[t], "prediction"], float(expected))
        self.assertTrue(
            np.allclose(five["volatility"], one["volatility"] * math.sqrt(5))
        )


class LiveHorizonTests(unittest.TestCase):
    def engine(self, manifest):
        from ibkrpy.manager.trading_engine import TradingEngine

        eng = TradingEngine.__new__(TradingEngine)
        eng.pipeline = types.SimpleNamespace(load_manifest=lambda s: manifest)
        return eng

    def test_horizon_comes_from_training_manifest(self):
        eng = self.engine({"horizon_bars": 5})
        strategy = types.SimpleNamespace(horizon_bars=2)
        self.assertEqual(eng._horizon_bars("SPY", strategy), 5)

    def test_falls_back_to_tuned_params_then_one(self):
        eng = self.engine(None)
        self.assertEqual(
            eng._horizon_bars("SPY", types.SimpleNamespace(horizon_bars=7)), 7
        )
        self.assertEqual(eng._horizon_bars("SPY", None), 1)

    def test_ledger_records_horizon_and_predicted_move(self):
        eng = self.engine(None)
        eng._ledger_buffer = []
        eng._model_generation = "g"
        eng._record_to_ledger(
            "SPY",
            "long_term",
            {"current_price": 100.0, "regime": "BULL_TREND"},
            {"ARIMA": 101.0},
            types.SimpleNamespace(value=0.01, source="model", realized=0.01),
            {
                "predicted_pct": 0.01,
                "threshold_pct": 0.004,
                "outcome": "BUY",
                "reason_code": "SIGNAL",
            },
            0.0,
            horizon_bars=5,
        )
        rec = eng._ledger_buffer[0]
        self.assertEqual(rec.horizon_bars, 5)
        self.assertAlmostEqual(rec.predicted_return, 0.01)
        self.assertAlmostEqual(rec.threshold, 0.004)
        self.assertEqual(rec.action, "BUY")


class DaemonLedgerFlushTests(unittest.TestCase):
    def test_market_cycle_flushes_the_ledger(self):
        import asyncio
        import datetime
        from unittest import mock

        from ibkrpy.core import system_daemon

        calls = []

        async def run_tick(symbol):
            calls.append(symbol)

        engine = types.SimpleNamespace(
            run_tick=run_tick,
            update_system_state=mock.AsyncMock(),
            log_cycle_summary=lambda: calls.append("summary"),
        )
        d = system_daemon.SystemDaemon.__new__(system_daemon.SystemDaemon)
        d.engine = engine
        d.symbols = ["SPY", "QQQ"]
        d._scan_offset = 0
        d.tick_interval_minutes = 0
        d.logger = mock.Mock()
        d.ib_manager = types.SimpleNamespace(
            ib=types.SimpleNamespace(isConnected=lambda: True)
        )

        with mock.patch.object(system_daemon.asyncio, "sleep", mock.AsyncMock()):
            asyncio.run(d._run_market_cycle(datetime.datetime(2026, 11, 10, 10, 0)))

        self.assertEqual(calls[-1], "summary")
        self.assertEqual(sorted(calls[:-1]), ["QQQ", "SPY"])


if __name__ == "__main__":
    unittest.main()
