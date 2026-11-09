# Run: python -m unittest discover -s tests
# Expected values are pandas_ta 0.3.14b output (without TA-Lib) on the same data.

import unittest

import numpy as np
import pandas as pd

from ibkrpy.shared.indicators import add_default_indicators, adx


def synthetic(seed=1, n=600):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, n)))
    # Flat bars (High == Low) exercise the epsilon branch.
    h[100:105] = c[100:105]
    lo[100:105] = c[100:105]
    o[100:105] = c[100:105]
    vol = rng.integers(1e5, 1e6, n).astype(float)
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": lo, "Close": c, "Volume": vol}, index=idx
    )


PANDAS_TA_LAST_ROW = {
    "SMA_10": 61.41588033078641,
    "EMA_20": 65.59016437597668,
    "RSI_14": 26.337648071141054,
    "ATRr_14": 2.091981631033675,
    "MACD_12_26_9": -3.961190316422396,
    "MACDh_12_26_9": -0.7899317021874888,
    "MACDs_12_26_9": -3.1712586142349073,
    "BBL_20_2.0": 54.650786874983936,
    "BBM_20_2.0": 67.26739160519541,
    "BBU_20_2.0": 79.88399633540689,
    "BBB_20_2.0": 37.51180008364418,
    "BBP_20_2.0": 0.17496172056993017,
}

PANDAS_TA_ADX_LAST_ROW = {
    "ADX_14": 30.105505663857535,
    "DMP_14": 18.920870096355774,
    "DMN_14": 41.500536731038196,
}


class IndicatorTests(unittest.TestCase):
    def test_columns_match_pandas_ta(self):
        out = add_default_indicators(synthetic())
        self.assertEqual(
            list(out.columns),
            ["Open", "High", "Low", "Close", "Volume", *PANDAS_TA_LAST_ROW],
        )

    def test_values_match_pandas_ta(self):
        last = add_default_indicators(synthetic()).iloc[-1]
        for col, expected in PANDAS_TA_LAST_ROW.items():
            self.assertAlmostEqual(last[col], expected, delta=abs(expected) * 1e-9)

    def test_adx_matches_pandas_ta(self):
        df = synthetic()
        last = adx(df["High"], df["Low"], df["Close"], 14).iloc[-1]
        for col, expected in PANDAS_TA_ADX_LAST_ROW.items():
            self.assertAlmostEqual(last[col], expected, delta=abs(expected) * 1e-9)

    def test_warmup_rows_are_nan(self):
        out = add_default_indicators(synthetic())
        self.assertEqual(out["SMA_10"].isna().sum(), 9)
        self.assertEqual(out["EMA_20"].isna().sum(), 19)
        self.assertEqual(out["MACDs_12_26_9"].isna().sum(), 33)

    def test_short_history_does_not_raise(self):
        out = add_default_indicators(synthetic(n=30))
        self.assertTrue(out["MACDs_12_26_9"].isna().all())
        self.assertNotIn("SMA_10", add_default_indicators(synthetic(n=5)).columns)


class AddTechnicalIndicatorsTests(unittest.TestCase):
    def test_text_columns_are_kept_and_ignored(self):
        from ibkrpy.data.data_pipeline import DataPipeline

        df = synthetic()
        df["symbol"] = "SPY"
        df["timeframe"] = "1 day"
        out = DataPipeline(artifact_store=object()).add_technical_indicators(df)
        self.assertEqual(len(out), len(df) - 33)
        self.assertTrue((out["symbol"] == "SPY").all())
        self.assertIn("BBP_20_2.0", out.columns)


if __name__ == "__main__":
    unittest.main()
