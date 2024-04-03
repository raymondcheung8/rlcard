from rlcard.agents import EquityAgent
from rlcard.games.nolimitholdem.game import Stage
import unittest


class TestEquityAgent(unittest.TestCase):
    def setUp(self):
        self.equity_agent = EquityAgent(num_actions=5)

    def test_preflop_pair(self):
        state = {'raw_obs': {'hand': ['C2', 'D2'], 'public_cards': [], 'stage': Stage.PREFLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['pair'], 1.0)

    def test_flop_pair(self):
        state = {'raw_obs': {'hand': ['C2', 'CQ'], 'public_cards': ['D2', 'C4', 'H6'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['pair'], 1.0)

    def test_turn_pair(self):
        state = {'raw_obs': {'hand': ['C2', 'CQ'], 'public_cards': ['D2', 'C4', 'H6', 'S8'], 'stage': Stage.TURN}}
        self.assertEqual(self.equity_agent.get_equities(state)['pair'], 1.0)

    def test_river_pair(self):
        state = {'raw_obs': {'hand': ['C2', 'CQ'], 'public_cards': ['D2', 'C4', 'H6', 'S8', 'DT'],
                             'stage': Stage.RIVER}}
        self.assertEqual(self.equity_agent.get_equities(state)['pair'], 1.0)

    def test_showdown_pair(self):
        state = {'raw_obs': {'hand': ['C2', 'CQ'], 'public_cards': ['D2', 'C4', 'H6', 'S8', 'DT'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['pair'], 1.0)

    def test_flop_two_pair(self):
        state = {'raw_obs': {'hand': ['C2', 'H4'], 'public_cards': ['D2', 'C4', 'H6'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['two_pair'], 1.0)

    def test_showdown_two_pair(self):
        state = {'raw_obs': {'hand': ['C2', 'H4'], 'public_cards': ['D2', 'C4', 'H6', 'S8', 'DT'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['two_pair'], 1.0)

    def test_flop_three_of_a_kind(self):
        state = {'raw_obs': {'hand': ['C2', 'H4'], 'public_cards': ['D2', 'C4', 'H2'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['three_of_a_kind'], 1.0)

    def test_showdown_three_of_a_kind(self):
        state = {'raw_obs': {'hand': ['C2', 'H4'], 'public_cards': ['D2', 'C4', 'H2', 'S8', 'DT'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['three_of_a_kind'], 1.0)

    def test_flop_straight(self):
        state = {'raw_obs': {'hand': ['C2', 'H3'], 'public_cards': ['D4', 'C5', 'H6'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['straight'], 1.0)

    def test_showdown_straight(self):
        state = {'raw_obs': {'hand': ['C2', 'H3'], 'public_cards': ['D4', 'C5', 'H6', 'S8', 'DT'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['straight'], 1.0)

    def test_flop_flush(self):
        state = {'raw_obs': {'hand': ['C2', 'C4'], 'public_cards': ['C6', 'C8', 'CT'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['flush'], 1.0)

    def test_showdown_flush(self):
        state = {'raw_obs': {'hand': ['C2', 'C4'], 'public_cards': ['C6', 'C8', 'CT', 'S5', 'D7'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['straight'], 1.0)

    def test_flop_full_house(self):
        state = {'raw_obs': {'hand': ['C2', 'H4'], 'public_cards': ['D2', 'C4', 'H2'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['full_house'], 1.0)

    def test_showdown_full_house(self):
        state = {'raw_obs': {'hand': ['C2', 'H4'], 'public_cards': ['D2', 'C4', 'H2', 'S5', 'D7'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['full_house'], 1.0)

    def test_flop_four_of_a_kind(self):
        state = {'raw_obs': {'hand': ['C2', 'H2'], 'public_cards': ['D2', 'S2', 'D3'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['four_of_a_kind'], 1.0)

    def test_showdown_four_of_a_kind(self):
        state = {'raw_obs': {'hand': ['C2', 'H2'], 'public_cards': ['D2', 'S2', 'D3', 'S5', 'D7'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['four_of_a_kind'], 1.0)

    def test_flop_straight_flush(self):
        state = {'raw_obs': {'hand': ['C2', 'C3'], 'public_cards': ['C4', 'C5', 'C6'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['straight_flush'], 1.0)

    def test_showdown_straight_flush(self):
        state = {'raw_obs': {'hand': ['C2', 'C3'], 'public_cards': ['C4', 'C5', 'C6', 'S5', 'D7'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['straight_flush'], 1.0)

    def test_flop_royal_flush(self):
        state = {'raw_obs': {'hand': ['CT', 'CJ'], 'public_cards': ['CQ', 'CK', 'CA'], 'stage': Stage.FLOP}}
        self.assertEqual(self.equity_agent.get_equities(state)['royal_flush'], 1.0)

    def test_showdown_royal_flush(self):
        state = {'raw_obs': {'hand': ['CT', 'CJ'], 'public_cards': ['CQ', 'CK', 'CA', 'S5', 'D7'],
                             'stage': Stage.SHOWDOWN}}
        self.assertEqual(self.equity_agent.get_equities(state)['royal_flush'], 1.0)


if __name__ == "__main__":
    unittest.main()
