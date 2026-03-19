# C++ Engine Bug Fixes - Summary

## Date: 2026-03-19

## Bugs Found and Fixed

### 1. **CRITICAL: Incorrect Birds Touching Detection**
**File:** `engine/engine.cpp` line 957
**Bug:** Only detected vertical adjacency (same X column)
```cpp
// BEFORE (WRONG):
if (ac.manhattan_to(bc) == 1 && ac.x == bc.x) return true;

// AFTER (CORRECT):
if (ac.manhattan_to(bc) == 1) return true;
```

**Impact:**
- Intercoiled fall physics were completely broken
- Birds touching horizontally were not detected as groups
- Model trained on DIFFERENT gravity physics than real game

---

### 2. **Missing Apple Fallback Generation**
**File:** `engine/engine.cpp` lines 425-434
**Bug:** No fallback when random generation produces <8 apples
**Fix:** Added lines 436-458 matching Java's GridMaker.java lines 170-184

```cpp
/* Fallback apple generation if too few apples */
if (grid.apples.size() < 8) {
    grid.apples.clear();
    std::vector<Coord> free_tiles;
    for (auto& c : all_coords) {
        if (grid.get(c) == TILE_EMPTY) {
            free_tiles.push_back(c);
        }
    }
    random.shuffle(free_tiles);

    int min_apple_coords = std::max(4, static_cast<int>(0.025 * free_tiles.size()));
    while (static_cast<int>(grid.apples.size()) < min_apple_coords * 2 && !free_tiles.empty()) {
        Coord c = free_tiles.back();
        free_tiles.pop_back();
        grid.apples.push_back(c);
        Coord opp = grid.opposite(c);
        grid.apples.push_back(opp);
        free_tiles.erase(
            std::remove_if(free_tiles.begin(), free_tiles.end(),
                [&](Coord fc) { return fc == opp; }),
            free_tiles.end());
    }
}
```

**Impact:**
- RNG sequence diverged on seeds where fallback triggered
- Different apple counts and positions
- Model trained on maps that don't match CodinGame

---

### 3. **Missing Enclosed Spawn Clearing**
**File:** `engine/engine.cpp` in `initial_state_from_seed()`
**Bug:** Didn't clear walls when spawn head is enclosed left/right
**Fix:** Added wall clearing matching Java's Game.java lines 69-77

```cpp
/* Clear walls if spawn head is enclosed (matching Java Game.java lines 69-77) */
if (!body.empty()) {
    Coord head = body[0];
    Coord left = head.add(-1, 0);
    Coord right = head.add(1, 0);
    if (state.grid.get(left) == TILE_WALL && state.grid.get(right) == TILE_WALL) {
        state.grid.set(left, TILE_EMPTY);
        state.grid.set(state.grid.opposite(left), TILE_EMPTY);
    }
}
```

**Impact:**
- Some initial states had enclosed spawns (unfair starts)
- Different initial grid configurations

---

## Root Cause Analysis

The C++ engine was ported from a **Rust implementation**, not directly from the **Java CodinGame referee**. The Rust version had:
1. Simplified physics (vertical-only touching)
2. Missing fallback logic for edge cases
3. Missing spawn safety checks

**The model was training on a fundamentally DIFFERENT GAME than what runs on CodinGame.**

---

## Testing

Run `test_engine_fixes.py` to verify:
- ✓ All grids now generate >=8 apples
- ✓ Grid generation doesn't crash
- ✓ Basic sanity checks pass

---

## Next Steps

1. **RETRAIN THE MODEL** - All previous training was on wrong physics
2. Compare engine output vs Java referee on same seeds to verify full correctness
3. Consider adding more comprehensive physics tests

---

## Files Modified

- `engine/engine.cpp` (3 fixes applied)
- `Makefile` (rebuilt with `make clean && make`)

Rebuilt engine: `engine/libengine.so`
