from collections import Counter
from rlcard.games.nolimitholdem.game import Stage, Action


class EquityAgent(object):
    ''' An agent that makes moves based on equity.
    '''

    def __init__(self, num_actions):
        ''' Initialize the equity agent

        Args:
            num_actions (int): The size of the output action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    def step(self, state):
        ''' Predict an action given the current state by using equity.

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action predicted by the equity agent
        '''
        equities = self.get_equities(state)
        print(equities)
        pot = state['raw_obs']['pot']
        all_chips = state['raw_obs']['all_chips']
        bet = max(all_chips) - all_chips[state['raw_obs']['current_player']]
        pot_odds = 1.0 * bet / pot if pot != 0 else 0.0

        # Remaps each hand to true or false depending on whether an equity is 100%
        equities_vs_100 = dict(map(
            lambda hand_equity: (hand_equity[0], hand_equity[1] == 1),
            equities.items()
        ))

        # equities_vs_100 but without pair
        equities_vs_100_no_pair = equities_vs_100.copy()
        equities_vs_100_no_pair.pop('pair')

        # equities_vs_100 but without pair, two pair and three of a kind
        equities_vs_100_no_p_no_2p_no_3oak = equities_vs_100_no_pair.copy()
        equities_vs_100_no_p_no_2p_no_3oak.pop('two_pair')
        equities_vs_100_no_p_no_2p_no_3oak.pop('three_of_a_kind')

        # Remaps each hand to true or false depending on whether an equity is better than the pot odds
        equities_vs_pot_odds = dict(map(
            lambda hand_equity: (hand_equity[0], hand_equity[1] >= pot_odds),
            equities.items()
        ))

        # equities_vs_pot_odds but without pair
        equities_vs_pot_odds_no_pair = equities_vs_pot_odds.copy()
        equities_vs_pot_odds_no_pair.pop('pair')

        # equities_vs_pot_odds but without pair, two pair and three of a kind
        equities_vs_pot_odds_no_p_no_2p_no_3oak = equities_vs_pot_odds_no_pair.copy()
        equities_vs_pot_odds_no_p_no_2p_no_3oak.pop('two_pair')
        equities_vs_pot_odds_no_p_no_2p_no_3oak.pop('three_of_a_kind')

        legal_actions = state['raw_obs']['legal_actions']

        # CHECK_CALL and FOLD will always be in legal actions
        # If a hand better than a three of a kind already exists, go all in
        if Action.ALL_IN in legal_actions and True in equities_vs_100_no_p_no_2p_no_3oak.values():
            return Action.ALL_IN
        elif bet == 0:
            # If there is no bet and a hand better than a pair exists, raise pot
            if Action.RAISE_POT in legal_actions and True in equities_vs_100_no_pair.values():
                return Action.RAISE_POT
            # If there is no bet and a hand better than a high card exists, raise half pot
            elif Action.RAISE_HALF_POT in legal_actions and True in equities_vs_100.values():
                return Action.RAISE_HALF_POT
            # If there is no bet and only a high card exists, check
            else:
                return Action.CHECK_CALL
        else:
            # If there is a bet and there exists an equity greater than pot odds for a hand
            # that is better than a three of a kind, raise pot
            if Action.RAISE_POT in legal_actions and True in equities_vs_pot_odds_no_p_no_2p_no_3oak.values():
                return Action.RAISE_POT
            # If there is a bet and there exists an equity greater than pot odds for a hand
            # that is better than a pair, raise half pot
            elif Action.RAISE_HALF_POT in legal_actions and True in equities_vs_pot_odds_no_pair.values():
                return Action.RAISE_HALF_POT
            # If there is a bet and there exists an equity greater than pot odds for a hand
            # that is better than a high card, call
            elif True in equities_vs_pot_odds.values():
                return Action.CHECK_CALL
            # Otherwise, fold
            else:
                return Action.FOLD

    def eval_step(self, state):
        ''' Predict an action given the current state by using equity for evaluation.
            Since the equity agents are not trained. This function is equivalent to step function

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action predicted by the equity agent
            probs (list): The list of action probabilities
        '''
        info = {}
        return self.step(state), info

    def get_equities(self, state):
        '''
        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            equities (dict) A dictionary that contains the equity of getting each hand
        '''
        all_cards = state['raw_obs']['hand'] + state['raw_obs']['public_cards']
        public_cards = state['raw_obs']['public_cards']
        # Add 0.02 to the multiplier for every card left to come
        equity_multiplier = 0.0
        if state['raw_obs']['stage'] == Stage.PREFLOP:
            equity_multiplier = 0.1
        elif state['raw_obs']['stage'] == Stage.FLOP:
            equity_multiplier = 0.04
        elif state['raw_obs']['stage'] == Stage.TURN:
            equity_multiplier = 0.02

        return {
            'pair': self.get_equity(
                self.get_pair_outs(all_cards),
                self.get_pair_outs(public_cards),
                equity_multiplier
            ),
            'two_pair': self.get_equity(
                self.get_two_pair_outs(all_cards),
                self.get_two_pair_outs(public_cards),
                equity_multiplier
            ),
            'three_of_a_kind': self.get_equity(
                self.get_three_of_a_kind_outs(all_cards),
                self.get_three_of_a_kind_outs(public_cards),
                equity_multiplier
            ),
            'straight': self.get_equity(
                self.get_straight_outs(all_cards),
                self.get_straight_outs(public_cards),
                equity_multiplier
            ),
            'flush': self.get_equity(
                self.get_flush_outs(all_cards),
                self.get_flush_outs(public_cards),
                equity_multiplier
            ),
            'full_house': self.get_equity(
                self.get_full_house_outs(all_cards),
                self.get_full_house_outs(public_cards),
                equity_multiplier
            ),
            'four_of_a_kind': self.get_equity(
                self.get_four_of_a_kind_outs(all_cards),
                self.get_four_of_a_kind_outs(public_cards),
                equity_multiplier
            ),
            'straight_flush': self.get_equity(
                self.get_straight_flush_outs(all_cards),
                self.get_straight_flush_outs(public_cards),
                equity_multiplier
            ),
            'royal_flush': self.get_equity(
                self.get_royal_flush_outs(all_cards),
                self.get_royal_flush_outs(public_cards),
                equity_multiplier
            )
        }

    @staticmethod
    def get_equity(outs_all, outs_public, equity_multiplier):
        '''
        Args:
            outs_all (int): The outs when considering all cards
            outs_public (int): The outs when considering only public cards
            equity_multiplier (float): This number is proportional to the number of opportunities left to hit an out

        Returns:
            equity (float): The equity based on the outs passed in and the multiplier
        '''
        # Work out the difference in equity
        # If outs_all is -1 and outs_public isn't -1, then equity is 100%
        return equity_multiplier * (outs_all - outs_public) \
            if outs_all != -1 or outs_public == -1 else 1.0

    @staticmethod
    def get_pair_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a pair
        '''
        ranks = set()
        for card in cards:
            if card[-1] in ranks:
                # Return -1 if a pair already exists
                return -1
            else:
                ranks.add(card[-1])

        # There are 4 suits in a deck so there are 3 remaining suits to choose from
        return 3 * len(ranks)

    @staticmethod
    def get_two_pair_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a two pair
        '''
        pair_count = 0
        ranks = set()
        for card in cards:
            # Check if new card creates a pair
            if card[-1] in ranks:
                if pair_count == 0:
                    pair_count += 1
                    ranks.remove(card[-1])
                else:
                    # Return -1 if a two pair already exists
                    return -1
            else:
                ranks.add(card[-1])

        # There are 4 suits in a deck so there are 3 remaining suits to choose from
        return 3 * len(ranks) if pair_count == 1 else 0

    @staticmethod
    def get_three_of_a_kind_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a three of a kind
        '''
        rank_counts = Counter(map(lambda card: card[-1], cards))

        # There are 4 suits in a deck so there are 2 remaining suits to choose from
        # Return -1 if a three of a kind already exists
        return 2 * len(list(filter(lambda count: count == 2, rank_counts.values()))) \
            if 3 not in rank_counts.values() else -1

    @staticmethod
    def get_straight_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a straight
        '''
        ranks = map(lambda card: card[-1], cards)
        all_ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        possible_straights = []
        while len(all_ranks) >= 5:
            straight = all_ranks[0:5]
            for rank in ranks:
                if rank in straight:
                    straight.remove(rank)

            if len(straight) == 0:
                # Return -1 if a straight already exists
                return -1
            elif len(straight) == 1:
                possible_straights.append(straight)

            all_ranks.pop(0)

        # There are 4 suits in a deck so there are 3 remaining suits to choose from
        return 3 * len(possible_straights)

    @staticmethod
    def get_flush_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a flush
        '''
        suit_counts = Counter(map(lambda card: card[0], cards))

        # There are 13 ranks in a deck so there are 9 remaining ranks to choose from
        # Return -1 if a flush already exists
        return 9 * len(list(filter(lambda count: count == 4, suit_counts.values()))) \
            if 5 not in suit_counts.values() else -1

    @staticmethod
    def get_full_house_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a full house
        '''
        rank_counts = Counter(map(lambda card: card[-1], cards))
        pair_count = len(list(filter(lambda count: count == 2, rank_counts.values())))

        if 3 in rank_counts.values():
            if pair_count > 0:
                # Return -1 if a full house already exists
                return -1
            else:
                # There are 4 suits in a deck so there are 3 remaining suits to choose from
                return 3 * (len(rank_counts) - pair_count)

        # There are 4 suits in a deck so there are 2 remaining suits to choose from
        return 2 * pair_count if pair_count >= 2 else 0

    @staticmethod
    def get_four_of_a_kind_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a four of a kind
        '''
        rank_counts = Counter(map(lambda card: card[-1], cards))

        # There are 4 suits in a deck so there are 1 remaining suits to choose from
        # Return -1 if a four of a kind already exists
        return len(list(filter(lambda count: count == 3, rank_counts.values()))) \
            if 4 not in rank_counts.values() else -1

    @staticmethod
    def get_straight_flush_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a straight flush
        '''
        all_suits = ['D', 'C', 'H', 'S']
        all_ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        possible_straight_flushes = []
        while len(all_ranks) >= 5:
            for suit in all_suits:
                straight_flush = list(map(lambda rank: suit + rank, all_ranks[:5]))
                for card in cards:
                    if card in straight_flush:
                        straight_flush.remove(card)

                if len(straight_flush) == 0:
                    # Return -1 if a straight flush already exists
                    return -1
                elif len(straight_flush) == 1:
                    possible_straight_flushes.append(straight_flush)

            all_ranks.pop(0)

        # All cards have to be the same suit in a straight flush
        return len(possible_straight_flushes)

    @staticmethod
    def get_royal_flush_outs(cards):
        '''
        Args:
            cards (list): A list of strings that represent cards

        Returns:
            outs (int): The number of outs for getting a royal flush
        '''
        all_suits = ['D', 'C', 'H', 'S']
        all_ranks = ['10', 'J', 'Q', 'K', 'A']
        possible_royal_flushes = []
        for suit in all_suits:
            royal_flush = list(map(lambda rank: suit + rank, all_ranks))
            for card in cards:
                if card in royal_flush:
                    royal_flush.remove(card)

            if len(royal_flush) == 0:
                # Return -1 if a straight already exists
                return -1
            elif len(royal_flush) == 1:
                possible_royal_flushes.append(royal_flush)

        all_ranks.pop(0)

        # All cards have to be the same suit in a royal flush
        return len(possible_royal_flushes)
