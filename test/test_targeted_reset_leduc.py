from PokerRL.game.wrappers import FlatLimitPokerEnvBuilder
from PokerRL.game.poker_env_args import LimitPokerEnvArgs
from PokerRL.game.games import StandardLeduc
from poker_env_test_v1 import build_cards_state_dict_leduc
from PokerRL.game.Poker import Poker


def build_env_wrapper():
    limit_args = LimitPokerEnvArgs(n_seats=2, starting_stack_sizes_list=[10000, 10000])
    env_builder = FlatLimitPokerEnvBuilder(StandardLeduc, limit_args)
    wrapper = env_builder.get_new_wrapper(
        is_evaluating=True, stack_size=limit_args.starting_stack_sizes_list
    )
    return wrapper


def test_targeted_reset_hole_and_flop():
    hole_p0 = (0, 0)
    hole_p1 = (1, 1)
    flop = (2, 0)

    wrapper = build_env_wrapper()
    csd = build_cards_state_dict_leduc(
        n_ranks=StandardLeduc.N_RANKS,
        n_suits=StandardLeduc.N_SUITS,
        hole_p0=hole_p0,
        hole_p1=hole_p1,
        flop=flop,
    )

    _obs, _r, _done, _info = wrapper.reset(deck_state_dict=csd)

    # Verify hole cards
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


def test_repeatable_reset_same_csd():
    hole_p0 = (1, 0)
    hole_p1 = (2, 1)
    flop = (0, 0)

    csd = build_cards_state_dict_leduc(
        n_ranks=StandardLeduc.N_RANKS,
        n_suits=StandardLeduc.N_SUITS,
        hole_p0=hole_p0,
        hole_p1=hole_p1,
        flop=flop,
    )

    # First run
    w1 = build_env_wrapper()
    w1.reset(deck_state_dict=csd)
    h10 = tuple(w1.env.seats[0].hand[0])
    h11 = tuple(w1.env.seats[1].hand[0])
    w1.step(Poker.CHECK_CALL)
    w1.step(Poker.CHECK_CALL)
    b1 = tuple(w1.env.board[0])

    # Second run
    w2 = build_env_wrapper()
    w2.reset(deck_state_dict=csd)
    h20 = tuple(w2.env.seats[0].hand[0])
    h21 = tuple(w2.env.seats[1].hand[0])
    w2.step(Poker.CHECK_CALL)
    w2.step(Poker.CHECK_CALL)
    b2 = tuple(w2.env.board[0])

    assert (h10, h11, b1) == (h20, h21, b2) == (hole_p0, hole_p1, flop)
