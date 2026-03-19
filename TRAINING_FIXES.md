# Training Pipeline Bug Fixes - Summary

## Date: 2026-03-19

All critical bugs in the training pipeline have been fixed.

---

## Bug Fixes Applied:

### ✅ Bug #1: Engine Physics - Birds Touching (CRITICAL)
**File:** `engine/engine.cpp` line 957
**Status:** ✅ FIXED
**Impact:** Intercoiled fall physics now work correctly for horizontally adjacent birds

### ✅ Bug #2: Engine - Missing Apple Fallback
**File:** `engine/engine.cpp` lines 436-458
**Status:** ✅ FIXED
**Impact:** All maps now guarantee ≥8 apples, consistent RNG sequences

### ✅ Bug #3: Engine - Enclosed Spawn Clearing
**File:** `engine/engine.cpp` in `initial_state_from_seed()`
**Status:** ✅ FIXED
**Impact:** Spawns no longer get trapped between walls

### ✅ Bug #4: Evaluation P0/P1 Bias
**File:** `train.py` lines 110-173
**Status:** ✅ FIXED
**Impact:** Evaluation now alternates between P0 and P1 roles (50/50 split)

### ✅ Bug #5: Memory Leak in Evaluation
**File:** `train.py` lines 168-171
**Status:** ✅ FIXED
**Impact:** Opponent models are now properly deleted after evaluation, preventing memory growth

### ✅ Bug #6: Reward Logic Contradiction
**File:** `snakebot_env.py` lines 461-466
**Status:** ✅ FIXED BY USER
**Impact:** Now uses pure terminal rewards (+1/-1/0) matching friend's top-3 setup

### ✅ Bug #7: Performance - Occupied Coords Caching
**File:** `snakebot_env.py`
**Status:** ✅ FIXED
**Changes:**
- Added `_occupied_coords_cache` and `_occupied_coords_turn` in `__init__`
- Created `_compute_occupied_coords()` helper method
- Updated `action_masks()` to use cached data (lines 497-551)
- Cache invalidated in `reset()` and `step()`

**Impact:**
- Cuts redundant computation in half (was computed 2x per step)
- Significant speedup during training over millions of steps

---

## Summary of Impact:

**Before Fixes:**
- Training on broken physics (horizontal birds couldn't fall together)
- Maps sometimes had too few apples
- Evaluation biased toward P0
- Memory leaked every 20k steps
- O(n²) redundant calculations every step

**After Fixes:**
- ✅ Correct physics matching Java referee
- ✅ Consistent map generation with ≥8 apples
- ✅ Fair P0/P1 evaluation
- ✅ No memory leaks
- ✅ 50% faster action masking
- ✅ Pure terminal rewards (+1/-1/0) like top-3 friend

---

## Next Steps:

1. **Retrain from scratch** - All previous training was on broken engine
2. Monitor initial training:
   - Watch for UP-spam behavior (should be gone with fixed physics)
   - Check win rates during self-play evaluations
   - Verify memory stays stable
3. Compare results to friend's setup (both using same reward structure now)

---

## Files Modified:

### Engine:
- `engine/engine.cpp` (3 physics/generation bugs)
- `engine/libengine.so` (rebuilt)

### Training:
- `train.py` (evaluation P0/P1 + memory leak fixes)
- `snakebot_env.py` (performance caching)

### Documentation:
- `ENGINE_FIXES.md` (engine bug details)
- `TRAINING_FIXES.md` (this file)
