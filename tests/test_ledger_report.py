"""--mode ledger resolves outcomes h bars later and reports per generation and horizon."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ibkrpy.data.prediction_ledger import PredictionLedger, PredictionRecord, report
from ibkrpy.shared.db_manager import DatabaseManager


class LedgerReportTests(unittest.TestCase):
    def test_report_resolves_and_judges(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            db = DatabaseManager(path)
            idx = pd.bdate_range("2026-01-01", periods=80, tz="UTC")
            rng = np.random.default_rng(0)
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
            bars = pd.DataFrame(
                {
                    "Open": close,
                    "High": close,
                    "Low": close,
                    "Close": close,
                    "Volume": 1.0,
                },
                index=idx,
            )
            db.save_bulk_market_data("SPY", bars, timeframe="1 day")

            ledger = PredictionLedger(path)
            recs = []
            for i in range(60):
                realized = close[i + 5] / close[i] - 1
                recs.append(
                    PredictionRecord(
                        symbol="SPY",
                        timestamp=idx[i].isoformat(),
                        term="long_term",
                        timeframe="1 day",
                        horizon_bars=5,
                        price_at_prediction=float(close[i]),
                        predicted_return=float(realized * 0.5),
                        model_generation="g1",
                    )
                )
            ledger.record_many(recs)

            text = report(ledger)

            self.assertIn("60 resolved rows", text)
            self.assertIn("horizon 5 bar(s)", text)
            self.assertIn("significantly positive", text)


if __name__ == "__main__":
    unittest.main()
