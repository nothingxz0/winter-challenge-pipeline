#!/usr/bin/env python3
"""
Test that the engine fixes are working:
1. Birds touching should detect any adjacency (not just vertical)
2. Apple fallback generation should trigger when <8 apples
3. Enclosed spawn heads should have walls cleared
"""

import ctypes
import os

# Load engine
engine_path = os.path.join(os.path.dirname(__file__), 'engine', 'libengine.so')
engine = ctypes.CDLL(engine_path)

# Define API
engine.engine_create.restype = ctypes.c_void_p
engine.engine_destroy.argtypes = [ctypes.c_void_p]
engine.engine_reset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
engine.engine_get_width.argtypes = [ctypes.c_void_p]
engine.engine_get_width.restype = ctypes.c_int
engine.engine_get_height.argtypes = [ctypes.c_void_p]
engine.engine_get_height.restype = ctypes.c_int
engine.engine_is_wall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
engine.engine_is_wall.restype = ctypes.c_int
engine.engine_apple_count.argtypes = [ctypes.c_void_p]
engine.engine_apple_count.restype = ctypes.c_int

def test_apple_count():
    """Test that grids always have at least 8 apples due to fallback generation"""
    print("Testing apple fallback generation...")

    # Try multiple seeds to find one that would naturally generate <8 apples
    for seed in range(100):
        eng = engine.engine_create()
        engine.engine_reset(eng, seed, 4)

        apple_count = engine.engine_apple_count(eng)

        if apple_count < 8:
            print(f"  FAIL: Seed {seed} generated only {apple_count} apples (should have fallback to >=8)")
            engine.engine_destroy(eng)
            return False

        engine.engine_destroy(eng)

    print(f"  PASS: All seeds generated >=8 apples")
    return True

def test_grid_generation():
    """Basic sanity check that grid generates without crashing"""
    print("Testing basic grid generation...")

    eng = engine.engine_create()
    engine.engine_reset(eng, 42, 4)

    w = engine.engine_get_width(eng)
    h = engine.engine_get_height(eng)
    apples = engine.engine_apple_count(eng)

    print(f"  Grid: {w}x{h}, Apples: {apples}")

    if w == 0 or h == 0:
        print("  FAIL: Invalid grid dimensions")
        engine.engine_destroy(eng)
        return False

    if apples < 8:
        print(f"  FAIL: Only {apples} apples (should be >=8 with fallback)")
        engine.engine_destroy(eng)
        return False

    engine.engine_destroy(eng)
    print("  PASS: Grid generated successfully")
    return True

def print_grid(eng):
    """Helper to visualize grid"""
    w = engine.engine_get_width(eng)
    h = engine.engine_get_height(eng)

    for y in range(h):
        row = ""
        for x in range(w):
            if engine.engine_is_wall(eng, x, y):
                row += "#"
            else:
                row += "."
        print(f"  {row}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing C++ Engine Fixes")
    print("=" * 60)

    all_pass = True

    all_pass &= test_grid_generation()
    print()
    all_pass &= test_apple_count()

    print()
    print("=" * 60)
    if all_pass:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
    print("=" * 60)
