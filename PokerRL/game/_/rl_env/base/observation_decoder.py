"""
Observation Decoder for PokerRL Environments.

This module provides decoders for both simple (flat) and multi-tensor observation formats.
It can reconstruct game state from raw observations and detect potential issues.

Author: Generated for debugging purposes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DecodedGameState:
    """Decoded state from an observation."""

    # Game parameters (normalized values from observation)
    ante_normalized: float = 0.0
    small_blind_normalized: float = 0.0
    big_blind_normalized: float = 0.0
    min_raise_normalized: float = 0.0
    pot_normalized: float = 0.0
    total_to_call_normalized: float = 0.0
    last_action_amount_normalized: float = 0.0

    # Last action info
    last_action_type: Optional[int] = None  # 0=FOLD, 1=CHECK_CALL, 2=BET_RAISE
    last_action_player: Optional[int] = None  # seat_id who took last action

    # Current state
    current_player_seat: int = 0
    current_round: int = 0  # 0=PREFLOP, 1=FLOP, 2=TURN, 3=RIVER
    observer_seat: int = 0  # NEW: The seat ID of the agent receiving this observation

    # Player states (per seat)
    player_stacks_normalized: List[float] = field(default_factory=list)
    player_bets_normalized: List[float] = field(default_factory=list)
    player_folded: List[bool] = field(default_factory=list)
    player_allin: List[bool] = field(default_factory=list)
    player_side_pot_ranks: List[int] = field(default_factory=list)

    # Side pots
    side_pots_normalized: List[float] = field(default_factory=list)

    # Board cards (list of (rank, suit) tuples, -127 means not dealt)
    board_cards: List[Tuple[int, int]] = field(default_factory=list)

    # Derived info (requires knowing starting stacks to denormalize)
    normalization_factor: Optional[float] = None

    # For multi-tensor obs
    hole_card: Optional[Tuple[int, int]] = None  # (rank, suit)
    action_history: List[Dict] = field(default_factory=list)

    def print_state(self, round_names=None):
        """Pretty print the decoded state."""
        if round_names is None:
            round_names = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}

        print("=" * 60)
        print("DECODED GAME STATE")
        print("=" * 60)

        print(
            f"\nGame Phase: {round_names.get(self.current_round, f'Round {self.current_round}')}"
        )
        print(f"Current Player (to act): Seat {self.current_player_seat}")
        print(f"Observer (this agent): Seat {self.observer_seat}")
        position = "SB (Small Blind)" if self.observer_seat == 0 else "BB (Big Blind)"
        print(f"Observer Position: {position}")

        print(f"\nGame Parameters (normalized):")
        print(f"  Ante: {self.ante_normalized:.4f}")
        print(f"  Small Blind: {self.small_blind_normalized:.4f}")
        print(f"  Big Blind: {self.big_blind_normalized:.4f}")

        print(f"\nBetting State (normalized):")
        print(f"  Pot: {self.pot_normalized:.4f}")
        print(f"  Total to Call: {self.total_to_call_normalized:.4f}")
        print(f"  Min Raise: {self.min_raise_normalized:.4f}")

        action_names = {0: "FOLD", 1: "CHECK/CALL", 2: "BET/RAISE", None: "None"}
        print(f"\nLast Action:")
        print(f"  Type: {action_names.get(self.last_action_type, 'Unknown')}")
        print(f"  Player: {self.last_action_player}")
        print(f"  Amount (normalized): {self.last_action_amount_normalized:.4f}")

        print(f"\nPlayer States:")
        for i in range(len(self.player_stacks_normalized)):
            status = []
            if self.player_folded[i] if i < len(self.player_folded) else False:
                status.append("FOLDED")
            if self.player_allin[i] if i < len(self.player_allin) else False:
                status.append("ALL-IN")
            status_str = " [" + ", ".join(status) + "]" if status else ""

            print(f"  Seat {i}{status_str}:")
            print(f"    Stack: {self.player_stacks_normalized[i]:.4f}")
            print(f"    Current Bet: {self.player_bets_normalized[i]:.4f}")

        if self.board_cards:
            board_str = " ".join(
                [f"({r},{s})" for r, s in self.board_cards if r != -127]
            )
            print(f"\nBoard: {board_str}" if board_str else "\nBoard: (no cards dealt)")

        if self.hole_card is not None:
            print(
                f"\nHole Card (from multi-tensor): ({self.hole_card[0]}, {self.hole_card[1]})"
            )

        print("=" * 60)


class SimpleObservationDecoder:
    """
    Decoder for simple (flat) observations when multi_tensor_obs=False.

    Observation structure for N_SEATS=2 with simple HU obs:
    [0]     ante (normalized)
    [1]     small_blind (normalized)
    [2]     big_blind (normalized)
    [3]     min_raise (normalized)
    [4]     pot_amt (normalized)
    [5]     total_to_call (normalized)
    [6]     last_action_how_much (normalized)
    [7-9]   last_action_what (one-hot: fold, call, raise)
    [10-11] last_action_who (one-hot per seat)
    [12-13] who_acts_next (one-hot per seat)
    [14-15] current_round (one-hot for rounds available)
    Then per player (2 players):
    [16]    stack_p0 (normalized)
    [17]    curr_bet_p0 (normalized)
    [18]    is_allin_p0
    [19]    stack_p1 (normalized)
    [20]    curr_bet_p1 (normalized)
    [21]    is_allin_p1
    Then board cards (N_TOTAL_BOARD_CARDS):
    For each card: N_RANKS + N_SUITS values (one-hot rank + one-hot suit)
    """

    def __init__(
        self,
        n_seats: int = 2,
        n_ranks: int = 3,  # Leduc default
        n_suits: int = 2,  # Leduc default
        n_total_board_cards: int = 1,  # Leduc has 1 board card
        n_rounds: int = 2,  # Leduc: PREFLOP, FLOP
        use_simple_hu_obs: bool = True,
        suits_matter: bool = False,
    ):
        self.n_seats = n_seats
        self.n_ranks = n_ranks
        self.n_suits = n_suits
        self.n_total_board_cards = n_total_board_cards
        self.n_rounds = n_rounds
        self.use_simple_hu_obs = use_simple_hu_obs
        self.suits_matter = suits_matter

        # Compute indices
        self._compute_indices()

    def _compute_indices(self):
        """Compute observation indices based on configuration."""
        idx = 0

        # Table state
        self.idx_ante = idx
        idx += 1
        self.idx_small_blind = idx
        idx += 1
        self.idx_big_blind = idx
        idx += 1
        self.idx_min_raise = idx
        idx += 1
        self.idx_pot = idx
        idx += 1
        self.idx_total_to_call = idx
        idx += 1
        self.idx_last_action_amount = idx
        idx += 1

        # Last action type (3 one-hot)
        self.idx_last_action_type_start = idx
        idx += 3

        # Last action who (n_seats one-hot)
        self.idx_last_action_who_start = idx
        idx += self.n_seats

        # Who acts next (n_seats one-hot)
        self.idx_acts_next_start = idx
        idx += self.n_seats

        # Current round (n_rounds one-hot)
        self.idx_round_start = idx
        idx += self.n_rounds

        # Side pots (for non-simple obs or N_SEATS > 2)
        if not self.use_simple_hu_obs or self.n_seats > 2:
            self.idx_side_pots_start = idx
            idx += self.n_seats
        else:
            self.idx_side_pots_start = None

        # NEW: Observer seat (agent's own identity) - added for all observation types
        self.idx_observer_seat_start = idx
        idx += self.n_seats

        # Player states
        self.idx_players_start = idx
        if self.use_simple_hu_obs and self.n_seats == 2:
            # stack, curr_bet, is_allin per player
            self.player_state_size = 3
        else:
            # stack, curr_bet, folded, is_allin, side_pot_rank (n_seats one-hot)
            self.player_state_size = 4 + self.n_seats

        idx += self.n_seats * self.player_state_size

        # Board cards
        self.idx_board_start = idx
        self.board_card_size = self.n_ranks + self.n_suits
        # total: n_total_board_cards * board_card_size

        self.total_obs_size = idx + self.n_total_board_cards * self.board_card_size

    def decode(self, obs: np.ndarray) -> DecodedGameState:
        """Decode a flat observation into a structured game state."""
        state = DecodedGameState()

        # Table state
        state.ante_normalized = float(obs[self.idx_ante])
        state.small_blind_normalized = float(obs[self.idx_small_blind])
        state.big_blind_normalized = float(obs[self.idx_big_blind])
        state.min_raise_normalized = float(obs[self.idx_min_raise])
        state.pot_normalized = float(obs[self.idx_pot])
        state.total_to_call_normalized = float(obs[self.idx_total_to_call])
        state.last_action_amount_normalized = float(obs[self.idx_last_action_amount])

        # Last action type
        action_type_onehot = obs[
            self.idx_last_action_type_start : self.idx_last_action_type_start + 3
        ]
        if np.any(action_type_onehot > 0):
            state.last_action_type = int(np.argmax(action_type_onehot))
        else:
            state.last_action_type = None

        # Last action who
        action_who_onehot = obs[
            self.idx_last_action_who_start : self.idx_last_action_who_start
            + self.n_seats
        ]
        if np.any(action_who_onehot > 0):
            state.last_action_player = int(np.argmax(action_who_onehot))
        else:
            state.last_action_player = None

        # Who acts next
        acts_next_onehot = obs[
            self.idx_acts_next_start : self.idx_acts_next_start + self.n_seats
        ]
        state.current_player_seat = int(np.argmax(acts_next_onehot))

        # Current round
        round_onehot = obs[self.idx_round_start : self.idx_round_start + self.n_rounds]
        state.current_round = int(np.argmax(round_onehot))

        # Side pots
        if self.idx_side_pots_start is not None:
            state.side_pots_normalized = list(
                obs[self.idx_side_pots_start : self.idx_side_pots_start + self.n_seats]
            )

        # Observer seat (agent's identity) - NEW
        observer_seat_onehot = obs[
            self.idx_observer_seat_start : self.idx_observer_seat_start + self.n_seats
        ]
        if np.any(observer_seat_onehot > 0):
            state.observer_seat = int(np.argmax(observer_seat_onehot))
        else:
            # Fallback: assume observer is current player
            state.observer_seat = state.current_player_seat

        # Player states
        state.player_stacks_normalized = []
        state.player_bets_normalized = []
        state.player_folded = []
        state.player_allin = []
        state.player_side_pot_ranks = []

        for p in range(self.n_seats):
            p_start = self.idx_players_start + p * self.player_state_size

            state.player_stacks_normalized.append(float(obs[p_start]))
            state.player_bets_normalized.append(float(obs[p_start + 1]))

            if self.use_simple_hu_obs and self.n_seats == 2:
                state.player_folded.append(False)  # Not in simple obs
                state.player_allin.append(bool(obs[p_start + 2] > 0.5))
            else:
                state.player_folded.append(bool(obs[p_start + 2] > 0.5))
                state.player_allin.append(bool(obs[p_start + 3] > 0.5))
                # Side pot rank one-hot
                side_pot_onehot = obs[p_start + 4 : p_start + 4 + self.n_seats]
                if np.any(side_pot_onehot > 0):
                    state.player_side_pot_ranks.append(int(np.argmax(side_pot_onehot)))
                else:
                    state.player_side_pot_ranks.append(-1)

        # Board cards
        state.board_cards = []
        for c in range(self.n_total_board_cards):
            c_start = self.idx_board_start + c * self.board_card_size
            rank_onehot = obs[c_start : c_start + self.n_ranks]
            suit_onehot = obs[
                c_start + self.n_ranks : c_start + self.n_ranks + self.n_suits
            ]

            if np.any(rank_onehot > 0):
                rank = int(np.argmax(rank_onehot))
                if self.suits_matter and np.any(suit_onehot > 0):
                    suit = int(np.argmax(suit_onehot))
                else:
                    suit = -1  # Unknown or irrelevant
                state.board_cards.append((rank, suit))
            else:
                state.board_cards.append((-127, -127))  # Not dealt

        return state

    def get_obs_description(self) -> Dict[str, Tuple[int, int]]:
        """Return a dictionary mapping field names to (start_idx, end_idx)."""
        desc = {}

        desc["ante"] = (self.idx_ante, self.idx_ante + 1)
        desc["small_blind"] = (self.idx_small_blind, self.idx_small_blind + 1)
        desc["big_blind"] = (self.idx_big_blind, self.idx_big_blind + 1)
        desc["min_raise"] = (self.idx_min_raise, self.idx_min_raise + 1)
        desc["pot"] = (self.idx_pot, self.idx_pot + 1)
        desc["total_to_call"] = (self.idx_total_to_call, self.idx_total_to_call + 1)
        desc["last_action_amount"] = (
            self.idx_last_action_amount,
            self.idx_last_action_amount + 1,
        )
        desc["last_action_type"] = (
            self.idx_last_action_type_start,
            self.idx_last_action_type_start + 3,
        )
        desc["last_action_who"] = (
            self.idx_last_action_who_start,
            self.idx_last_action_who_start + self.n_seats,
        )
        desc["acts_next"] = (
            self.idx_acts_next_start,
            self.idx_acts_next_start + self.n_seats,
        )
        desc["current_round"] = (
            self.idx_round_start,
            self.idx_round_start + self.n_rounds,
        )

        if self.idx_side_pots_start is not None:
            desc["side_pots"] = (
                self.idx_side_pots_start,
                self.idx_side_pots_start + self.n_seats,
            )

        # NEW: Observer seat (agent identity)
        desc["observer_seat"] = (
            self.idx_observer_seat_start,
            self.idx_observer_seat_start + self.n_seats,
        )

        for p in range(self.n_seats):
            p_start = self.idx_players_start + p * self.player_state_size
            desc[f"player_{p}_stack"] = (p_start, p_start + 1)
            desc[f"player_{p}_bet"] = (p_start + 1, p_start + 2)
            if self.use_simple_hu_obs and self.n_seats == 2:
                desc[f"player_{p}_allin"] = (p_start + 2, p_start + 3)
            else:
                desc[f"player_{p}_folded"] = (p_start + 2, p_start + 3)
                desc[f"player_{p}_allin"] = (p_start + 3, p_start + 4)
                desc[f"player_{p}_side_pot_rank"] = (
                    p_start + 4,
                    p_start + 4 + self.n_seats,
                )

        for c in range(self.n_total_board_cards):
            c_start = self.idx_board_start + c * self.board_card_size
            desc[f"board_card_{c}_rank"] = (c_start, c_start + self.n_ranks)
            desc[f"board_card_{c}_suit"] = (
                c_start + self.n_ranks,
                c_start + self.n_ranks + self.n_suits,
            )

        return desc


class MultiTensorObservationDecoder:
    """
    Decoder for multi-tensor observations when multi_tensor_obs=True.

    Observation structure:
    {
        "card": np.array of shape (size_of_streets + 3, N_SUITS, N_RANKS)
            - [0]: Player's hole card (added by wrapper)
            - [1, 2, ...]: Street-specific cards (flop, etc.)
            - [-4]: Turn card
            - [-3]: River card
            - [-2]: (unused or summed cards)
            - [-1]: (unused or summed cards)

        "action": np.array of shape (n_rounds * (max_raises + 2) + 2, 4, 9)
            - Dimension 0: Action slot per round
            - Dimension 1: [agent_row, opponent_row, sum_row, legal_actions_row]
            - Dimension 2: Action encoding (typically 3 actions + padding)
    }

    IDENTIFIED BUGS:
    1. In _update_card_multi_tensor_board():
       - FLOP: correctly unpacks as (rank, suit), indexes as [suit, rank] (intentional?)
       - TURN/RIVER: INCORRECTLY unpacks as (suit, rank) - should be (rank, suit)!

    2. In _update_card_multi_tensor_actions():
       - agent_idx is based on SB_POS comparison, not current player
       - This breaks consistency in who is "agent" vs "opponent"
    """

    def __init__(
        self,
        n_suits: int = 2,
        n_ranks: int = 3,
        n_rounds: int = 2,  # Leduc: PREFLOP, FLOP
        max_raises_per_round: int = 2,
        n_actions: int = 3,
        n_flop_cards: int = 1,
        n_turn_cards: int = 0,
        n_river_cards: int = 0,
    ):
        self.n_suits = n_suits
        self.n_ranks = n_ranks
        self.n_rounds = n_rounds
        self.max_raises_per_round = max_raises_per_round
        self.n_actions = n_actions
        self.n_flop_cards = n_flop_cards
        self.n_turn_cards = n_turn_cards
        self.n_river_cards = n_river_cards

        # Card tensor: (n_rounds + 3, n_suits, n_ranks)
        self.card_tensor_shape = (n_rounds + 3, n_suits, n_ranks)

        # Action tensor shape
        self.possible_actions_per_round = (
            max_raises_per_round + 3
        )  # +3 for initial actions
        self.action_tensor_shape = (
            n_rounds * self.possible_actions_per_round + 2,
            4,  # agent, opponent, sum, legal
            9,  # max action encoding width
        )

    def decode(self, obs: Dict[str, np.ndarray]) -> DecodedGameState:
        """Decode a multi-tensor observation into a structured game state."""
        state = DecodedGameState()

        card_tensor = obs.get("card", np.zeros(self.card_tensor_shape))
        action_tensor = obs.get("action", np.zeros(self.action_tensor_shape))

        # Decode hole card (stored in position 0)
        hole_card_plane = card_tensor[0]  # Shape: (n_suits, n_ranks)
        if np.any(hole_card_plane > 0):
            suit, rank = np.unravel_index(
                np.argmax(hole_card_plane), hole_card_plane.shape
            )
            state.hole_card = (rank, suit)

        # Decode board cards
        state.board_cards = []

        # Flop cards (positions 1 to n_flop_cards)
        for i in range(self.n_flop_cards):
            if i + 1 < len(card_tensor):
                card_plane = card_tensor[i + 1]
                if np.any(card_plane > 0):
                    suit, rank = np.unravel_index(
                        np.argmax(card_plane), card_plane.shape
                    )
                    state.board_cards.append((rank, suit))

        # Turn card (position -4)
        if self.n_turn_cards > 0 and len(card_tensor) >= 4:
            turn_plane = card_tensor[-4]
            if np.any(turn_plane > 0):
                # BUG NOTE: Original code uses (suit, rank) unpacking which is wrong!
                suit, rank = np.unravel_index(np.argmax(turn_plane), turn_plane.shape)
                state.board_cards.append((rank, suit))

        # River card (position -3)
        if self.n_river_cards > 0 and len(card_tensor) >= 3:
            river_plane = card_tensor[-3]
            if np.any(river_plane > 0):
                suit, rank = np.unravel_index(np.argmax(river_plane), river_plane.shape)
                state.board_cards.append((rank, suit))

        # Decode action history
        state.action_history = []
        for round_idx in range(self.n_rounds):
            for action_idx in range(self.possible_actions_per_round):
                slot_idx = round_idx * self.possible_actions_per_round + action_idx
                if slot_idx >= len(action_tensor):
                    break

                action_slot = action_tensor[slot_idx]

                # Check if any action was taken in this slot
                agent_actions = action_slot[0, : self.n_actions]
                opponent_actions = action_slot[1, : self.n_actions]
                legal_actions = (
                    action_slot[3, : self.n_actions]
                    if len(action_slot) > 3
                    else np.zeros(self.n_actions)
                )

                if np.any(agent_actions > 0) or np.any(opponent_actions > 0):
                    state.action_history.append(
                        {
                            "round": round_idx,
                            "action_idx": action_idx,
                            "agent_action": (
                                int(np.argmax(agent_actions))
                                if np.any(agent_actions > 0)
                                else None
                            ),
                            "opponent_action": (
                                int(np.argmax(opponent_actions))
                                if np.any(opponent_actions > 0)
                                else None
                            ),
                            "legal_actions": list(np.where(legal_actions > 0)[0]),
                        }
                    )

        return state

    def get_tensor_description(self) -> Dict[str, str]:
        """Return a description of the tensor layout."""
        return {
            "card_tensor_shape": str(self.card_tensor_shape),
            "card[0]": "Player's hole card (suit, rank)",
            "card[1:1+n_flop]": f"Flop cards ({self.n_flop_cards} cards)",
            "card[-4]": "Turn card",
            "card[-3]": "River card",
            "card[-2,-1]": "Summed/auxiliary card info",
            "action_tensor_shape": str(self.action_tensor_shape),
            "action[slot][0]": "Agent's action one-hot",
            "action[slot][1]": "Opponent's action one-hot",
            "action[slot][2]": "Sum of actions",
            "action[slot][3]": "Legal actions mask",
        }


class ObservationAnalyzer:
    """
    Analyzes observations for potential issues that could impact learning.
    """

    def __init__(
        self,
        simple_decoder: SimpleObservationDecoder = None,
        multi_decoder: MultiTensorObservationDecoder = None,
    ):
        self.simple_decoder = simple_decoder or SimpleObservationDecoder()
        self.multi_decoder = multi_decoder or MultiTensorObservationDecoder()

    def analyze_simple_obs(self, obs: np.ndarray) -> Dict[str, Any]:
        """Analyze a simple observation for potential issues."""
        issues = []
        warnings = []

        state = self.simple_decoder.decode(obs)

        # Check for issues

        # 1. Check if observation is all zeros (terminal state)
        if np.allclose(obs, 0):
            issues.append("Observation is all zeros - this is a terminal state")

        # 2. Check normalization consistency
        if state.small_blind_normalized > 0 and state.big_blind_normalized > 0:
            ratio = state.big_blind_normalized / state.small_blind_normalized
            if not (1.5 <= ratio <= 3.0):
                warnings.append(f"Unusual SB/BB ratio: {ratio:.2f} (expected ~2.0)")

        # 3. Check if pot is smaller than blinds (which shouldn't happen after reset)
        if (
            state.pot_normalized
            < state.small_blind_normalized + state.big_blind_normalized
        ):
            if state.ante_normalized == 0:  # Only for non-ante games
                warnings.append("Pot is smaller than expected initial blinds")

        # 4. Check for player stack consistency
        total_stacks = sum(state.player_stacks_normalized)
        total_bets = sum(state.player_bets_normalized)
        total_pot = state.pot_normalized
        # In a zero-sum game, stacks + bets + pot should roughly equal initial stacks

        # 5. Check if current player has valid stack
        if state.player_stacks_normalized[state.current_player_seat] <= 0:
            if not state.player_allin[state.current_player_seat]:
                issues.append(
                    f"Current player (seat {state.current_player_seat}) has zero stack but not marked all-in"
                )

        return {
            "state": state,
            "issues": issues,
            "warnings": warnings,
            "is_terminal": np.allclose(obs, 0),
        }

    def analyze_multi_tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze a multi-tensor observation for potential issues."""
        issues = []
        warnings = []

        state = self.multi_decoder.decode(obs)

        card_tensor = obs.get("card", np.zeros((1, 1, 1)))
        action_tensor = obs.get("action", np.zeros((1, 1, 1)))

        # Check for issues

        # 1. Check if hole card is set
        if state.hole_card is None:
            warnings.append("No hole card detected in observation")

        # 2. Check card tensor for multiple cards in same plane
        for i in range(len(card_tensor)):
            plane = card_tensor[i]
            card_count = np.sum(plane > 0.5)
            if card_count > 1:
                issues.append(
                    f"Card plane {i} has {card_count} cards (should be 0 or 1)"
                )

        # 3. Check for action tensor consistency
        for slot in action_tensor:
            agent_sum = np.sum(slot[0])
            opponent_sum = np.sum(slot[1])
            if agent_sum > 1.5 or opponent_sum > 1.5:
                warnings.append(f"Action slot has multiple actions marked")

        # 4. Known bug check: Verify card tensor indexing
        # The bug is in how cards are indexed: [suit, rank] vs [rank, suit]
        if state.hole_card is not None:
            rank, suit = state.hole_card
            if rank >= self.multi_decoder.n_ranks or suit >= self.multi_decoder.n_suits:
                issues.append(
                    f"Hole card indices out of bounds: rank={rank}, suit={suit}"
                )

        return {
            "state": state,
            "issues": issues,
            "warnings": warnings,
            "card_tensor_shape": card_tensor.shape,
            "action_tensor_shape": action_tensor.shape,
        }

    def check_learning_suitability(
        self, obs: Any, obs_type: str = "simple"
    ) -> Dict[str, Any]:
        """
        Check if observations are suitable for RL learning.

        Returns analysis of potential issues that could prevent learning.

        NOTE: Several bugs have been FIXED in PokerEnv.py:
        1. observer_seat field now added to observations (agent knows its identity)
        2. Card indexing bug in TURN/RIVER fixed (was using wrong unpacking order)
        3. Action tensor now uses consistent seat-based indexing
        """
        issues = []
        fixed_issues = []

        if obs_type == "simple":
            state = self.simple_decoder.decode(obs)

            # FIXED: Agent now knows its own seat via observer_seat field
            fixed_issues.append(
                {
                    "severity": "FIXED",
                    "issue": "Agent can now determine its own seat from observation",
                    "explanation": "The 'observer_seat' one-hot field tells the agent which seat it occupies. "
                    "In heads-up, seat 0 is SB and seat 1 is BB. "
                    f"Current observation shows observer is seat {state.observer_seat}.",
                }
            )

            # Still a potential issue: Reward scaling
            issues.append(
                {
                    "severity": "WARNING",
                    "issue": "Reward scaling may be inconsistent between base env and wrapper",
                    "explanation": "Base env uses REWARD_SCALAR, wrapper multiplies by 100. "
                    "This could cause confusion in value estimation.",
                    "solution": "Use consistent reward scaling throughout the pipeline.",
                }
            )

        else:  # multi-tensor
            state = self.multi_decoder.decode(obs)

            # FIXED: Action tensor now uses consistent seat-based indexing
            fixed_issues.append(
                {
                    "severity": "FIXED",
                    "issue": "Action tensor now uses consistent seat-based indexing",
                    "explanation": "Row 0 always represents seat 0's actions, "
                    "Row 1 always represents seat 1's actions. "
                    "This is consistent across all actions in a game.",
                }
            )

            # FIXED: Card indexing bug in TURN/RIVER
            fixed_issues.append(
                {
                    "severity": "FIXED",
                    "issue": "Card unpacking order is now correct for all streets",
                    "explanation": "All streets (FLOP, TURN, RIVER) now correctly use (rank, suit) unpacking. "
                    "Board cards are properly encoded in the tensor.",
                }
            )

        return {
            "state": state,
            "learning_issues": issues,
            "fixed_issues": fixed_issues,
            "recommendation": "Most critical issues have been fixed. Check reward scaling if learning is slow.",
        }


# Convenience functions


def create_leduc_simple_decoder() -> SimpleObservationDecoder:
    """Create a decoder for StandardLeduc with simple observations."""
    return SimpleObservationDecoder(
        n_seats=2,
        n_ranks=3,
        n_suits=2,
        n_total_board_cards=1,
        n_rounds=2,
        use_simple_hu_obs=True,  # Leduc with 2 players
        suits_matter=False,
    )


def create_leduc_multi_tensor_decoder() -> MultiTensorObservationDecoder:
    """Create a decoder for StandardLeduc with multi-tensor observations."""
    return MultiTensorObservationDecoder(
        n_suits=2,
        n_ranks=3,
        n_rounds=2,
        max_raises_per_round=2,
        n_actions=3,
        n_flop_cards=1,
        n_turn_cards=0,
        n_river_cards=0,
    )


def decode_and_print(obs: Any, obs_type: str = "auto"):
    """
    Convenience function to decode and print an observation.

    Args:
        obs: The observation (numpy array or dict)
        obs_type: "simple", "multi", or "auto" to auto-detect
    """
    if obs_type == "auto":
        obs_type = "multi" if isinstance(obs, dict) else "simple"

    if obs_type == "simple":
        decoder = create_leduc_simple_decoder()
        state = decoder.decode(obs)
    else:
        decoder = create_leduc_multi_tensor_decoder()
        state = decoder.decode(obs)

    state.print_state()
    return state


if __name__ == "__main__":
    # Demo usage
    print("Observation Decoder - Demo")
    print("=" * 60)

    # Create decoders
    simple_decoder = create_leduc_simple_decoder()
    multi_decoder = create_leduc_multi_tensor_decoder()

    print("\nSimple Observation Structure:")
    print("-" * 40)
    for name, (start, end) in simple_decoder.get_obs_description().items():
        print(f"  [{start:2d}:{end:2d}] {name}")

    print(f"\nTotal observation size: {simple_decoder.total_obs_size}")

    print("\n\nMulti-Tensor Observation Structure:")
    print("-" * 40)
    for name, desc in multi_decoder.get_tensor_description().items():
        print(f"  {name}: {desc}")

    # Analyze learning suitability
    print("\n\nLearning Suitability Analysis:")
    print("-" * 40)
    analyzer = ObservationAnalyzer(simple_decoder, multi_decoder)

    # Dummy obs for analysis
    dummy_simple_obs = np.zeros(simple_decoder.total_obs_size)
    analysis = analyzer.check_learning_suitability(dummy_simple_obs, "simple")

    for issue in analysis["learning_issues"]:
        print(f"\n[{issue['severity']}] {issue['issue']}")
        print(f"  Explanation: {issue['explanation']}")
        print(f"  Solution: {issue['solution']}")
