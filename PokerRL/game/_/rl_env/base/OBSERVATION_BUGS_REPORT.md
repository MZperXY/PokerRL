# PokerRL Observation System - Bug Report & Analysis

## Executive Summary

After analyzing the observation construction in `PokerEnv.py` and its wrappers, I identified several **critical bugs** that could prevent an RL agent from learning effectively. 

### ✅ BUGS FIXED (Jan 2025)

1. **Agent Identity Issues** - ✅ FIXED: Added `observer_seat` field to observations
2. **Card Indexing Bug** - ✅ FIXED: TURN/RIVER now correctly unpack as (rank, suit)
3. **Inconsistent Action Indexing** - ✅ FIXED: Now uses consistent seat-based indexing

### ⚠️ Remaining Issues

1. **Reward Scaling Inconsistency** - Multiple conflicting reward scaling approaches (base env vs wrapper)

---

## ✅ FIXED: Bug #1 - Agent Cannot Determine Its Own Seat

### Status: **FIXED**

### What was fixed
Added `observer_seat` one-hot field to both simple and full observations.

### Changes made in `PokerEnv.py`:

1. **Observation space construction** (`_construct_obs_space()`):
   - Added `observer_seat_N` fields after round encoding

2. **Table state generation** (`_get_table_state()`):
   - Now accepts optional `observer_seat` parameter
   - Adds observer seat one-hot to output

3. **Observation getter** (`get_current_obs()`):
   - Now accepts optional `observer_seat` parameter
   - Passes it to `_get_table_state()`

### New observation structure
```
... existing fields ...
[N-N+1]  current_round one-hot
[N+2-N+3] observer_seat one-hot  # NEW: tells agent which seat it is
... player states ...
```

---

## ✅ FIXED: Bug #2 - Multi-Tensor Card Indexing Bug

### Status: **FIXED**

### What was wrong
TURN and RIVER cards were unpacked as `(suit, rank)` instead of `(rank, suit)`.

### Fix applied in `_update_card_multi_tensor_board()`:
```python
elif current_round == Poker.TURN:
    # FIX: Changed from (suit, rank) to (rank, suit)
    for rank, suit in self.board[...]:
        card_multi_tensor[len(card_multi_tensor) - 4, suit, rank] = 1
        
elif current_round == Poker.RIVER:
    # FIX: Changed from (suit, rank) to (rank, suit)
    for rank, suit in self.board[...]:
        card_multi_tensor[len(card_multi_tensor) - 3, suit, rank] = 1
```

---

## ✅ FIXED: Bug #3 - Inconsistent Agent/Opponent Indexing

### Status: **FIXED**

### What was wrong
Action tensor used SB_POS-based indexing, causing agent/opponent to flip depending on who was acting.

### Fix applied in `_update_card_multi_tensor_actions()`:
```python
# FIX: Use consistent seat-based indexing
# Row 0 always represents seat 0's actions
# Row 1 always represents seat 1's actions
current_seat = self.current_player.seat_id
opponent_seat = 1 - current_seat

current_tensor[current_seat, : len(last_action)] = last_action
current_tensor[opponent_seat, : len(last_action)] = previous_action
```

---

## ⚠️ Warning: Reward Scaling Inconsistency (NOT FIXED)

### Locations
- `PokerEnv._get_step_reward()` (line 1377): Uses `REWARD_SCALAR`
- `PokerEnvSimple._edit_returns()` (line 344): Multiplies by 100

### Problem
Two different scaling factors are applied:

1. Base env: `reward / REWARD_SCALAR` where `REWARD_SCALAR = starting_stack / 5`
2. Wrapper: `reward * 100`

For Leduc with stacks of 10000:
- REWARD_SCALAR = 10000 / 5 = 2000
- A 100 chip win becomes: (100 / 2000) * 100 = 5.0

### Impact
- Reward magnitudes are hard to interpret
- Value function targets might be in unexpected ranges
- Could affect learning stability

### Suggested Fix
Choose one consistent scaling approach:

```python
# Option 1: Remove wrapper scaling, use only REWARD_SCALAR
# In PokerEnvSimple._edit_returns():
r_for_all = {"0": r_for_all[0], "1": r_for_all[1]}  # No * 100

# Option 2: Document the combined effect clearly
```

---

## ✅ FIXED: Warning #5 - Missing Position Information

### Status: **FIXED**

The `observer_seat` field now tells the agent which seat it occupies.

---

## Updated Observation Structure Reference

### Simple Observation (multi_tensor_obs=False)

For Leduc Hold'em with 2 players:

| Index | Field | Description |
|-------|-------|-------------|
| 0 | ante | Normalized ante |
| 1 | small_blind | Normalized SB |
| 2 | big_blind | Normalized BB |
| 3 | min_raise | Normalized min raise |
| 4 | pot_amt | Normalized pot |
| 5 | total_to_call | Normalized to-call amount |
| 6 | last_action_how_much | Normalized bet amount |
| 7-9 | last_action_what | One-hot [fold, call, raise] |
| 10-11 | last_action_who | One-hot [seat 0, seat 1] |
| 12-13 | who_acts_next | One-hot [seat 0, seat 1] |
| 14-15 | current_round | One-hot [preflop, flop] |
| **16-17** | **observer_seat** | **NEW: One-hot [seat 0, seat 1] - agent's own identity** |
| 18 | stack_p0 | Player 0 stack (normalized) |
| 19 | curr_bet_p0 | Player 0 current bet |
| 20 | is_allin_p0 | Player 0 all-in flag |
| 21 | stack_p1 | Player 1 stack |
| 22 | curr_bet_p1 | Player 1 current bet |
| 23 | is_allin_p1 | Player 1 all-in flag |
| 24-28 | board_card | One-hot rank (3) + suit (2) |

**Total: 29 elements** (for Leduc) - increased by 2 for observer_seat

**Note:** Hole cards are added by wrapper as a prefix (6 elements for Leduc RANGE_SIZE).

### Multi-Tensor Observation (multi_tensor_obs=True)

```
{
    "card": shape (5, 2, 3)  # (n_rounds+3, n_suits, n_ranks)
        [0]: Player's hole card
        [1]: Flop card
        [2]: (unused for Leduc)
        [3]: (unused for Leduc)
        [4]: Summed card info
        
    "action": shape (12, 4, 9)  # (n_rounds*(max_raises+3)+2, 4, 9)
        [slot][0]: Seat 0's action one-hot  # FIXED: consistent indexing
        [slot][1]: Seat 1's action one-hot  # FIXED: consistent indexing
        [slot][2]: Sum of actions
        [slot][3]: Legal actions mask
}
```

---

## Fix Status Summary

| Issue | Status | Priority |
|-------|--------|----------|
| Agent identity (observer_seat) | ✅ FIXED | Was CRITICAL |
| Card indexing TURN/RIVER | ✅ FIXED | Was CRITICAL |
| Action tensor indexing | ✅ FIXED | Was CRITICAL |
| Reward scaling inconsistency | ⚠️ NOT FIXED | WARNING |
## Remaining Recommended Fix

1. **MEDIUM**: Standardize reward scaling between base env and wrappers

---

## Testing Recommendations

1. Use the `observation_decoder.py` module to decode observations during training
2. Print decoded states periodically to verify correctness
3. Add unit tests that:
   - Verify card encoding/decoding round-trips correctly
   - Verify action history is consistent across perspectives
   - Verify rewards sum to zero (zero-sum game)

---

## Files Created

1. `observation_decoder.py` - Decoder classes for both observation types
   - `SimpleObservationDecoder` - For flat observations
   - `MultiTensorObservationDecoder` - For multi-tensor observations
   - `ObservationAnalyzer` - For detecting issues
   - Convenience functions for Leduc Hold'em

Usage:
```python
from PokerRL.game._.rl_env.base.observation_decoder import (
    decode_and_print,
    create_leduc_simple_decoder,
    ObservationAnalyzer,
)

# Decode and print any observation
decode_and_print(obs)

# Analyze for issues
analyzer = ObservationAnalyzer()
issues = analyzer.check_learning_suitability(obs, "simple")
```

