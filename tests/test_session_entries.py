"""Extended hours are close-only: the session rule blocks entries, never exits."""

import unittest

from ibkrpy.shared.session_clock import Session
from ibkrpy.strategy.strategy_components import RiskController, SessionEntryRule


class SessionEntryRuleTests(unittest.TestCase):
    def setUp(self):
        self.rule = SessionEntryRule()

    def test_regular_hours_allow_entries(self):
        allowed, _ = self.rule.assess({"session": Session.RTH})
        self.assertTrue(allowed)

    def test_extended_sessions_block_entries(self):
        for s in (Session.PRE, Session.POST, Session.OVERNIGHT):
            allowed, reason = self.rule.assess({"session": s})
            self.assertFalse(allowed, s)
            self.assertIn("close-only", reason)

    def test_opt_in_allows_extended_entries(self):
        rule = SessionEntryRule(allow_extended_entries=True)
        allowed, _ = rule.assess({"session": Session.PRE})
        self.assertTrue(allowed)

    def test_unknown_session_does_not_block(self):
        allowed, _ = self.rule.assess({})
        self.assertTrue(allowed)

    def test_controller_vetoes_in_extended_hours(self):
        controller = RiskController([self.rule])
        allowed, reason = controller.check_entries_allowed({"session": Session.POST})
        self.assertFalse(allowed)
        self.assertIn("post", reason)


if __name__ == "__main__":
    unittest.main()
