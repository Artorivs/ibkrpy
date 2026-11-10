"""Incremental downloads leave no holes, keep daily series daily, and repair split jumps."""

import asyncio
import os
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

from ibkrpy.data import bar_sync


def daily_index(start, end):
    return pd.bdate_range(start, end, tz="UTC")


class HelperTests(unittest.TestCase):
    def test_duration_never_short(self):
        self.assertEqual(bar_sync.duration_str(59, "5 mins"), "2 M")
        self.assertEqual(bar_sync.duration_str(400, "1 day"), "2 Y")
        self.assertEqual(bar_sync.duration_str(45, "1 day"), "45 D")
        self.assertEqual(bar_sync.duration_str(0, "1 day"), "1 D")

    def test_gap_of_three_weekdays_is_found(self):
        idx = daily_index("2026-07-01", "2026-07-31")
        holed = idx[(idx < "2026-07-14") | (idx > "2026-07-16")]
        at = bar_sync.find_gap_start(holed, pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertEqual(at, pd.Timestamp("2026-07-13", tz="UTC"))

    def test_single_missing_weekday_is_a_holiday_not_a_gap(self):
        idx = daily_index("2026-07-01", "2026-07-31")
        idx = idx[idx != "2026-07-03"]
        self.assertIsNone(
            bar_sync.find_gap_start(idx, pd.Timestamp("2026-01-01", tz="UTC"))
        )

    def test_gap_before_window_is_ignored(self):
        idx = daily_index("2026-07-01", "2026-07-31")
        holed = idx[(idx < "2026-07-14") | (idx > "2026-07-16")]
        self.assertIsNone(
            bar_sync.find_gap_start(holed, pd.Timestamp("2026-07-20", tz="UTC"))
        )

    def test_split_jump_found_but_ordinary_big_move_is_not(self):
        idx = daily_index("2026-10-01", "2026-10-20")
        close = pd.Series(300.0, index=idx)
        close[idx >= "2026-10-10"] = 101.0
        since = pd.Timestamp("2026-01-01", tz="UTC")
        self.assertEqual(
            bar_sync.find_split_jump(close, since), pd.Timestamp("2026-10-05", tz="UTC")
        )
        close[idx >= "2026-10-10"] = 300.0 / 1.7
        self.assertIsNone(bar_sync.find_split_jump(close, since))

    def test_ratio_and_rescale(self):
        idx = daily_index("2026-10-01", "2026-10-10")
        stored = pd.Series(300.0, index=idx)
        fetched = pd.Series(100.0, index=idx[3:])
        ratio = bar_sync.adjustment_ratio(stored, fetched)
        self.assertAlmostEqual(ratio, 3.0)
        self.assertTrue(bar_sync.needs_rescale(ratio))
        self.assertFalse(bar_sync.needs_rescale(1.02))
        bars = pd.DataFrame({"Close": [300.0], "Volume": [10.0]})
        out = bar_sync.rescale_bars(bars, 3.0)
        self.assertAlmostEqual(out["Close"][0], 100.0)
        self.assertAlmostEqual(out["Volume"][0], 30.0)


class DailyGuardTests(unittest.TestCase):
    def test_intraday_rows_are_kept_out_of_daily_series(self):
        from ibkrpy.shared.db_manager import DatabaseManager

        with tempfile.TemporaryDirectory() as d:
            db = DatabaseManager(os.path.join(d, "t.db"))
            idx = pd.DatetimeIndex(
                ["2026-05-01 00:00", "2026-05-01 14:30", "2026-05-04 00:00"], tz="UTC"
            )
            df = pd.DataFrame(
                {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
                index=idx,
            )
            db.save_bulk_market_data("SPY", df, timeframe="1 day")
            db.save_bulk_market_data("SPY", df, timeframe="1 hour")
            self.assertEqual(len(db.get_market_data_sync("SPY", "1 day")), 2)
            self.assertEqual(len(db.get_market_data_sync("SPY", "1 hour")), 3)


class FakeDB:
    def __init__(self, df):
        self.df = df

    def get_market_data_sync(self, symbol, timeframe):
        return self.df.copy()

    def save_bulk_market_data(self, symbol, df, timeframe):
        self.df = df.combine_first(self.df).sort_index()


class FakeIB:
    """Serves bars from `truth` inside the requested (end - duration, end] window."""

    def __init__(self, truth):
        self.truth = truth
        self.requests = []

    async def fetch_historical_data(
        self, contract, end_datetime, duration, bar_size, what_to_show, **kw
    ):
        self.requests.append((end_datetime, duration))
        n, unit = duration.split()
        days = int(n) * {"D": 1, "M": 30, "Y": 365}[unit]
        end = (
            pd.Timestamp.now(tz="UTC")
            if not end_datetime
            else pd.to_datetime(end_datetime, format="%Y%m%d-%H:%M:%S", utc=True)
        )
        t = self.truth
        return t[(t.index > end - pd.Timedelta(days=days)) & (t.index <= end)].copy()


def bars(close):
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1000.0,
        },
        index=close.index,
    )


class SyncTests(unittest.TestCase):
    def run_sync(self, stored, truth, term="long_term"):
        from ibkrpy.manager.pipeline_manager import PipelineManager

        pm = PipelineManager.__new__(PipelineManager)
        pm.config = types.SimpleNamespace(get=lambda k, d=None: d)
        pm.db = FakeDB(stored)
        pm.ib_data = FakeIB(truth)

        async def no_sleep(*a):
            return None

        orig = asyncio.sleep
        asyncio.sleep = no_sleep
        try:
            asyncio.run(pm._sync_symbol_term_data("SPY", term, None))
        finally:
            asyncio.sleep = orig
        return pm.db.df, pm.ib_data.requests

    def truth(self):
        today = pd.Timestamp.now(tz="UTC").normalize()
        idx = daily_index(today - pd.Timedelta(days=700), today)
        rng = np.random.default_rng(0)
        return bars(
            pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), idx)
        )

    def test_hole_and_long_outage_are_filled(self):
        truth = self.truth()
        idx = truth.index
        hole = (idx > idx[-120]) & (idx < idx[-116])
        stored = truth[~hole & (idx < idx[-60])]

        out, _ = self.run_sync(stored, truth)

        self.assertTrue(truth.index[-200:].isin(out.index).all())

    def test_up_to_date_series_makes_no_request(self):
        truth = self.truth()
        _, requests = self.run_sync(truth.copy(), truth)
        self.assertEqual(requests, [])

    def test_split_jump_is_rescaled(self):
        truth = self.truth()
        idx = truth.index
        split_at = idx[-30]
        stored = truth[idx < idx[-5]].copy()
        before = stored.index < split_at
        for col in ("Open", "High", "Low", "Close"):
            stored.loc[before, col] *= 3.0

        out, _ = self.run_sync(stored, truth)

        common = truth.index.intersection(out.index)
        self.assertTrue(
            np.allclose(out.loc[common, "Close"], truth.loc[common, "Close"])
        )


if __name__ == "__main__":
    unittest.main()
