"""
Exit-path safety: closing a position must never leave two exit orders working
at once (a bracket stop plus a closing order), because if both fill the account
ends up short.
"""

import asyncio
import types
import unittest

from ibkrpy.manager.trading_engine import TradingEngine


class FakeTrade:
    _next_id = 1

    def __init__(self, symbol, order, status="Submitted"):
        self.contract = types.SimpleNamespace(symbol=symbol)
        self.order = order
        if not getattr(order, "orderId", 0):
            order.orderId = FakeTrade._next_id
            FakeTrade._next_id += 1
        self.orderStatus = types.SimpleNamespace(status=status, filled=0.0)

    def isDone(self):
        return self.orderStatus.status in ("Filled", "Cancelled", "Inactive")


class FakeIB:
    """Minimal IB stand-in. fill_mode controls what a new closing order does."""

    def __init__(self, fill_mode="fill", honour_cancel=True):
        self.trades = []
        self.placed = []
        self.fill_mode = fill_mode
        self.honour_cancel = honour_cancel

    def openTrades(self):
        return [t for t in self.trades if not t.isDone()]

    def cancelOrder(self, order):
        for t in self.trades:
            if t.order is order and self.honour_cancel:
                t.orderStatus.status = "Cancelled"

    def placeOrder(self, contract, order):
        trade = FakeTrade(contract.symbol, order)
        self.trades.append(trade)
        self.placed.append(order)
        qty = float(order.totalQuantity)
        if self.fill_mode == "fill":
            trade.orderStatus.status = "Filled"
            trade.orderStatus.filled = qty
        elif self.fill_mode == "partial":
            trade.orderStatus.filled = qty / 2
        return trade


def make_engine(ib, positions=None):
    eng = TradingEngine.__new__(TradingEngine)
    eng.data = types.SimpleNamespace(ib=ib)
    eng.dry_run = False
    eng.cached_positions = dict(positions or {})
    eng.config = types.SimpleNamespace(
        get=lambda key, default=None: (
            0.5 if key == "strategy_settings.close_fill_timeout_sec" else default
        )
    )
    eng.session_clock = types.SimpleNamespace(
        classify=lambda: types.SimpleNamespace(is_extended=False)
    )
    eng.CANCEL_CONFIRM_SECONDS = 0.5
    eng.protected = []

    async def fake_attach(symbol, qty):
        eng.protected.append((symbol, qty))

    eng._attach_oca_protection = fake_attach
    return eng


def bracket_stop(ib, symbol, action, qty):
    order = types.SimpleNamespace(
        action=action, orderType="STP", totalQuantity=qty, orderId=0
    )
    ib.trades.append(FakeTrade(symbol, order))
    return order


CONTRACT = types.SimpleNamespace(symbol="NVDA")


class ClosePositionTests(unittest.TestCase):
    def run_close(self, eng, qty=10, action="SELL"):
        return asyncio.run(eng._close_position(CONTRACT, "NVDA", action, qty, 100.0))

    def test_cancels_stop_before_closing_and_updates_position(self):
        ib = FakeIB("fill")
        bracket_stop(ib, "NVDA", "SELL", 10)
        eng = make_engine(ib, {"NVDA": 10})

        filled = self.run_close(eng)

        self.assertEqual(filled, 10)
        self.assertEqual(ib.openTrades(), [])
        self.assertEqual(eng.cached_positions["NVDA"], 0)
        self.assertEqual(eng.protected, [])
        self.assertEqual(ib.placed[-1].orderType, "MKT")

    def test_no_close_sent_when_stop_cancel_unconfirmed(self):
        ib = FakeIB("fill", honour_cancel=False)
        bracket_stop(ib, "NVDA", "SELL", 10)
        eng = make_engine(ib, {"NVDA": 10})

        filled = self.run_close(eng)

        self.assertEqual(filled, 0)
        self.assertEqual(ib.placed, [])  # the stop stays the only exit order
        self.assertEqual(eng.cached_positions["NVDA"], 10)

    def test_unfilled_close_is_cancelled_then_reprotected(self):
        ib = FakeIB("none")
        bracket_stop(ib, "NVDA", "SELL", 10)
        eng = make_engine(ib, {"NVDA": 10})

        filled = self.run_close(eng)

        self.assertEqual(filled, 0)
        self.assertEqual(ib.openTrades(), [])  # no closing order left working
        self.assertEqual(eng.protected, [("NVDA", 10)])

    def test_partial_fill_reprotects_only_the_remainder(self):
        ib = FakeIB("partial")
        eng = make_engine(ib, {"NVDA": -10})

        filled = self.run_close(eng, qty=10, action="BUY")

        self.assertEqual(filled, 5)
        self.assertEqual(eng.protected, [("NVDA", -5)])
        self.assertEqual(eng.cached_positions["NVDA"], -5)

    def test_extended_session_uses_outside_rth_limit(self):
        ib = FakeIB("fill")
        eng = make_engine(ib, {"NVDA": 10})
        eng.session_clock = types.SimpleNamespace(
            classify=lambda: types.SimpleNamespace(is_extended=True)
        )

        self.run_close(eng)

        order = ib.placed[-1]
        self.assertEqual(order.orderType, "LMT")
        self.assertTrue(order.outsideRth)
        self.assertAlmostEqual(order.lmtPrice, 99.8)

    def test_stop_that_fills_during_cancel_is_not_sold_twice(self):
        ib = FakeIB("fill")
        stop = bracket_stop(ib, "NVDA", "SELL", 10)

        def cancel_but_stop_fills(order):
            for t in ib.trades:
                if t.order is stop:
                    t.orderStatus.status = "Filled"
                    t.orderStatus.filled = 10

        ib.cancelOrder = cancel_but_stop_fills
        eng = make_engine(ib, {"NVDA": 10})

        self.run_close(eng)

        self.assertEqual(ib.placed, [])  # no extra SELL that would open a short
        self.assertEqual(eng.cached_positions["NVDA"], 0)


class ProtectUnhedgedTests(unittest.TestCase):
    def test_orphan_closing_limit_is_cancelled_before_new_protection(self):
        ib = FakeIB()
        orphan = types.SimpleNamespace(
            action="SELL", orderType="LMT", totalQuantity=10, orderId=0
        )
        ib.trades.append(FakeTrade("NVDA", orphan))
        eng = make_engine(ib, {"NVDA": 10})

        asyncio.run(eng._protect_unhedged_positions())

        self.assertEqual(ib.openTrades(), [])
        self.assertEqual(eng.protected, [("NVDA", 10)])

    def test_protected_position_is_left_alone(self):
        ib = FakeIB()
        bracket_stop(ib, "NVDA", "SELL", 10)
        eng = make_engine(ib, {"NVDA": 10})

        asyncio.run(eng._protect_unhedged_positions())

        self.assertEqual(len(ib.openTrades()), 1)
        self.assertEqual(eng.protected, [])

    def test_no_protection_when_orphan_cancel_unconfirmed(self):
        ib = FakeIB(honour_cancel=False)
        orphan = types.SimpleNamespace(
            action="SELL", orderType="LMT", totalQuantity=10, orderId=0
        )
        ib.trades.append(FakeTrade("NVDA", orphan))
        eng = make_engine(ib, {"NVDA": 10})

        asyncio.run(eng._protect_unhedged_positions())

        self.assertEqual(eng.protected, [])


if __name__ == "__main__":
    unittest.main()
