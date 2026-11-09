"""GARCH must return per-bar σ in the same units as the bars it is given."""

import importlib.util
import math
import pathlib
import unittest

import numpy as np
import pandas as pd

from ibkrpy.strategy.volatility_estimator import VolatilityEstimator

# Load garch.py directly: ibkrpy/models/__init__.py imports TensorFlow.
_spec = importlib.util.spec_from_file_location(
    "ibkrpy_garch",
    pathlib.Path(__file__).resolve().parents[1] / "ibkrpy" / "models" / "garch.py",
)
_garch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_garch)
GARCHModel = _garch.GARCHModel


def price_frame(sigma_per_bar: float, n: int, freq: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma_per_bar, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2025-01-02 14:30", periods=n, freq=freq, tz="UTC")
    high = close * (1 + sigma_per_bar / 2)
    low = close * (1 - sigma_per_bar / 2)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": 1e6},
        index=idx,
    )


class GarchScaleTests(unittest.TestCase):
    def assert_close_to(self, value, target, tol=0.3):
        self.assertTrue(math.isfinite(value))
        self.assertLess(abs(value / target - 1), tol, f"{value} vs {target}")

    def test_hourly_bars_give_per_bar_sigma(self):
        df = price_frame(0.004, 1200, "h")
        self.assert_close_to(GARCHModel().predict_volatility(df), 0.004)

    def test_five_minute_bars_give_per_bar_sigma(self):
        df = price_frame(0.0012, 1500, "5min")
        self.assert_close_to(GARCHModel().predict_volatility(df), 0.0012)

    def test_daily_bars_give_per_bar_sigma(self):
        df = price_frame(0.015, 600, "D")
        self.assert_close_to(GARCHModel().predict_volatility(df), 0.015)

    def test_too_little_data_returns_nan(self):
        df = price_frame(0.01, 10, "D")
        self.assertTrue(math.isnan(GARCHModel().predict_volatility(df)))

    def test_estimator_accepts_hourly_garch_at_ratio_near_one(self):
        df = price_frame(0.004, 1200, "h")
        model_sigma = GARCHModel().predict_volatility(df)
        est = VolatilityEstimator().estimate("TEST", df, model_sigma)
        self.assertEqual(est.source, "model")
        self.assertGreater(est.ratio, 0.7)
        self.assertLess(est.ratio, 1.4)


if __name__ == "__main__":
    unittest.main()
