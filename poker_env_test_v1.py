"""
Example: Build a Leduc Hold'em environment and force specific hole cards and flop via deck_state_dict.

Run directly to see a deterministic hand: two preflop checks advance to flop with the specified board card.
"""

from typing import List, Tuple

from PokerRL.game.wrappers import FlatLimitPokerEnvBuilder
from PokerRL.game.poker_env_args import LimitPokerEnvArgs
from PokerRL.game.games import StandardLeduc
from PokerRL.game._.rl_env.base._Deck import DeckOfCards
from PokerRL.game.Poker import Poker


def build_cards_state_dict_leduc(
    n_ranks: int,
    n_suits: int,
    hole_p0: Tuple[int, int],
    hole_p1: Tuple[int, int],
    flop: Tuple[int, int],
):
    """
    Build a cards_state_dict compatible with PokerEnv.load_cards_state_dict for Leduc.

    - hole_p0, hole_p1, flop are (rank, suit) with 0 <= rank < n_ranks and 0 <= suit < n_suits.
    - Ensures deck_remaining top card equals the desired flop, and excludes the hole cards and flop from the deck.
    """
    deck = DeckOfCards(num_suits=n_suits, num_ranks=n_ranks)
    # Start from an ordered deck, we will set our own order deterministically
    all_cards = [tuple(card.tolist()) for card in deck.deck_remaining]

    def not_eq(a, b):
        return not (a[0] == b[0] and a[1] == b[1])

    # Remove chosen cards
    remaining = [
        c
        for c in all_cards
        if not_eq(c, hole_p0) and not_eq(c, hole_p1) and not_eq(c, flop)
    ]
    # Place flop on top, rest arbitrary deterministic order
    new_order = [flop] + remaining
    deck_remaining = new_order  # list of (rank, suit)

    board = [[Poker.CARD_NOT_DEALT_TOKEN_1D, Poker.CARD_NOT_DEALT_TOKEN_1D]]
    hands: List[list] = [
        [list(hole_p0)],
        [list(hole_p1)],
    ]

    return {
        "deck": {"deck_remaining": deck_remaining},
        "board": board,
        "hand": hands,
    }


def demo_run(
    hole_p0=(0, 0),
    hole_p1=(1, 0),
    flop=(2, 1),
):
    # Build env and wrapper
    limit_args = LimitPokerEnvArgs(n_seats=2, starting_stack_sizes_list=[10000, 10000])
    env_builder = FlatLimitPokerEnvBuilder(StandardLeduc, limit_args)
    wrapper = env_builder.get_new_wrapper(
        is_evaluating=True, stack_size=limit_args.starting_stack_sizes_list
    )

    # Build deterministic cards_state
    csd = build_cards_state_dict_leduc(
        n_ranks=StandardLeduc.N_RANKS,
        n_suits=StandardLeduc.N_SUITS,
        hole_p0=hole_p0,
        hole_p1=hole_p1,
        flop=flop,
    )

    _obs, _r, _done, _info = wrapper.reset(deck_state_dict=csd)

    # Verify hole cards from env directly
    p0_hand = wrapper.env.seats[0].hand
    p1_hand = wrapper.env.seats[1].hand
    assert tuple(p0_hand[0]) == hole_p0
    assert tuple(p1_hand[0]) == hole_p1

    # Two checks to go to flop
    _obs, _r, _done, _info = wrapper.step(Poker.CHECK_CALL)
    _obs, _r, _done, _info = wrapper.step(Poker.CHECK_CALL)

    # Verify flop dealt as desired
    board = wrapper.env.board
    assert tuple(board[0]) == flop

    print(
        "P0 hole:",
        p0_hand.tolist(),
        "P1 hole:",
        p1_hand.tolist(),
        "Flop:",
        board[0].tolist(),
    )


if __name__ == "__main__":
    # Example desired cards: P0 rank 0 suit 0, P1 rank 1 suit 1, Flop rank 2 suit 0
    demo_run(hole_p0=(0, 0), hole_p1=(1, 1), flop=(2, 0))
