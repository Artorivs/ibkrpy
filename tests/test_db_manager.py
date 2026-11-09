# tests/test_db_manager.py
# 執行: python -m unittest discover -s tests

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from ibkrpy.shared.db_manager import DatabaseManager


def bars(dates, start=100.0):
    idx = pd.to_datetime(dates, utc=True)
    close = [start + i for i in range(len(idx))]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [1000.0] * len(idx),
        },
        index=idx,
    )


class MarketDataRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = DatabaseManager(os.path.join(self.tmp, "t.db"))

    def test_save_then_load_keeps_ohlcv(self):
        # 回歸測試：v0.5.3-beta 把欄位改名方向寫反，只剩 3 欄，8 個佔位符的 INSERT 失敗
        df = bars(["2024-01-02", "2024-01-03", "2024-01-04"])
        self.db.save_bulk_market_data("SPY", df)
        out = self.db.get_market_data_sync("SPY")
        self.assertEqual(len(out), 3)
        self.assertEqual(list(out["Close"]), [100.0, 101.0, 102.0])
        for col in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, out.columns)

    def test_load_drops_duplicate_timestamps(self):
        # 回歸測試：v0.5.3-beta 的遮罩算在 set_index 之前的 RangeIndex 上，永遠不去重
        def row(ts, close):
            return {
                "timestamp": ts,
                "symbol": "SPY",
                "timeframe": "1 day",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
            }

        rows = [
            row("2024-01-02 00:00:00", 1.0),
            row("2024-01-02 00:00:00", 2.0),
            row("2024-01-03 00:00:00", 3.0),
        ]
        with mock.patch.object(self.db, "_fetch_sync", return_value=rows):
            out = self.db.get_market_data_sync("SPY")
        self.assertFalse(out.index.duplicated().any())
        self.assertEqual(list(out["Close"]), [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
