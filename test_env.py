#!/usr/bin/env python3
"""
Test suite for snakebot_env.py — verifies reset, step, obs shape, reward values.
"""

import sys
import os
import numpy as np

# Ensure we can import from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snakebot_env import SnakeBotEnv


def test_reset_returns_correct_obs():
    """Reset should return (obs, info) with obs shape (7, 64, 64)."""
    env = SnakeBotEnv(league=4)
    obs, info = env.reset(seed=42)

    assert obs.shape == (7, 64, 64), f"Expected (7,64,64), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert isinstance(info, dict)

    # Walls channel should have some 1s (at least bottom row)
    assert obs[0].sum() > 0, "Wall channel should be non-zero"
    # Apples channel should have some 1s
    assert obs[1].sum() > 0, "Apple channel should be non-zero"
    # Our heads channel should have some 1s
    assert obs[2].sum() > 0, "My-heads channel should be non-zero"
    # Height map channel should have non-zero values
    assert obs[6].sum() > 0, "Height map should be non-zero"

    env.close()
    print("  [OK] test_reset_returns_correct_obs")


def test_step_returns_correct_types():
    """Step should return (obs, reward, terminated, truncated, info)."""
    env = SnakeBotEnv(league=4)
    obs, _ = env.reset(seed=42)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    assert obs2.shape == (7, 64, 64), f"Step obs shape: {obs2.shape}"
    assert isinstance(reward, float), f"Reward type: {type(reward)}"
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # With dense rewards, non-terminal steps have small rewards (time tax, apples, etc.)
    # Just check it's a reasonable range
    assert -10.0 <= reward <= 10.0, f"Reward out of range: {reward}"

    env.close()
    print("  [OK] test_step_returns_correct_types")


def test_game_terminates():
    """Playing random moves should eventually end the game."""
    env = SnakeBotEnv(league=4, max_turns=200)
    obs, _ = env.reset(seed=123)

    steps = 0
    terminated = False
    while not terminated and steps < 250:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert terminated, f"Game should terminate within 200 turns, took {steps}"
    assert steps <= 200, f"Expected max 200 turns, got {steps}"
    # With dense rewards, terminal reward includes accumulated step rewards
    assert -15.0 <= reward <= 15.0, f"Terminal reward out of range: {reward}"

    env.close()
    print(f"  [OK] test_game_terminates (after {steps} steps, reward={reward})")


def test_reward_values():
    """Dense rewards: check we get varied rewards across games."""
    env = SnakeBotEnv(league=4, max_turns=50)  # Short games for speed
    results = []

    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        results.append(reward)

    # With dense rewards, results should be in reasonable range
    assert all(-15.0 <= r <= 15.0 for r in results), f"Rewards out of range: {results}"
    # Should get varied results (not all identical)
    assert len(set(results)) >= 2, f"All same reward: {results}"

    env.close()
    print(f"  [OK] test_reward_values (results: {dict(zip(*np.unique(results, return_counts=True)))})")


def test_obs_values_range():
    """All observation values should be in [0, 1]."""
    env = SnakeBotEnv(league=4)
    obs, _ = env.reset(seed=77)

    assert obs.min() >= 0.0, f"Obs min: {obs.min()}"
    assert obs.max() <= 1.0, f"Obs max: {obs.max()}"

    # After a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
        assert obs.min() >= 0.0, f"Obs min after step: {obs.min()}"
        assert obs.max() <= 1.0, f"Obs max after step: {obs.max()}"

    env.close()
    print("  [OK] test_obs_values_range")


def test_multiple_seeds():
    """Different seeds should produce different initial states."""
    env = SnakeBotEnv(league=4)
    obs_list = []
    for seed in [1, 2, 3, 100, 999]:
        obs, _ = env.reset(seed=seed)
        obs_list.append(obs.copy())

    # At least some should differ
    diffs = 0
    for i in range(len(obs_list)):
        for j in range(i + 1, len(obs_list)):
            if not np.array_equal(obs_list[i], obs_list[j]):
                diffs += 1
    assert diffs > 0, "All seeds produced identical observations"

    env.close()
    print(f"  [OK] test_multiple_seeds ({diffs} different pairs)")


def test_opponent_callback():
    """Test that opponent callback is called correctly."""
    call_count = [0]

    def dummy_opponent(obs):
        call_count[0] += 1
        assert obs.shape == (7, 64, 64)
        return np.array([0, 0, 0, 0])  # All NORTH

    env = SnakeBotEnv(league=4, opponent=dummy_opponent)
    obs, _ = env.reset(seed=42)

    for _ in range(3):
        action = env.action_space.sample()
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break

    assert call_count[0] >= 1, "Opponent callback should have been called"

    env.close()
    print(f"  [OK] test_opponent_callback (called {call_count[0]} times)")


def test_action_space_sample():
    """Action space sampling should produce valid actions."""
    env = SnakeBotEnv(league=4)
    env.reset(seed=42)

    for _ in range(10):
        action = env.action_space.sample()
        assert len(action) == 4
        assert all(0 <= a <= 3 for a in action)

    env.close()
    print("  [OK] test_action_space_sample")


if __name__ == "__main__":
    print("Running snakebot_env tests...\n")

    tests = [
        test_reset_returns_correct_obs,
        test_step_returns_correct_types,
        test_game_terminates,
        test_reward_values,
        test_obs_values_range,
        test_multiple_seeds,
        test_opponent_callback,
        test_action_space_sample,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
