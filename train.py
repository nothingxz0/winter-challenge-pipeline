#!/usr/bin/env python3
"""
PPO Self-Play Training for Snakebot.

Phase 1: PPO vs random opponent (warmup)
Phase 2: Self-play (frozen opponent, update when win_rate > 60%)

Usage:
    python train.py --total-steps 1000000
    python train.py --total-steps 500000 --warmup-steps 50000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from snakebot_env import SnakeBotEnv
from unet_policy import CompactUNetExtractor

SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
LOG_DIR = SCRIPT_DIR / "logs"


class SelfPlayCallback(BaseCallback):
    """
    Self-play callback: periodically evaluate and update the frozen opponent.

    Every `eval_freq` steps:
    - Play 100 games (50 as P0, 50 as P1) vs frozen opponent
    - If win_rate > update_threshold: snapshot model → new opponent
    """

    def __init__(
        self,
        eval_freq=20_000,
        n_eval_games=100,
        update_threshold=0.80,
        patience_threshold=0.55,
        patience_evals=3,
        checkpoint_dir=None,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.update_threshold = update_threshold
        self.patience_threshold = patience_threshold
        self.patience_evals = patience_evals
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.best_win_rate = 0.0
        self.opponent_version = 0
        self.patience_count = 0
        self.eval_results = []

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True

        win_rate = self._evaluate()
        self.eval_results.append(win_rate)

        if self.verbose:
            print(f"\n[SelfPlay] Step {self.num_timesteps}: "
                  f"win_rate={win_rate:.1%} vs opponent v{self.opponent_version}")

        should_update = False
        if win_rate >= self.update_threshold:
            should_update = True
            reason = f"win_rate {win_rate:.1%} >= {self.update_threshold:.1%}"
        elif win_rate >= self.patience_threshold:
            self.patience_count += 1
            if self.patience_count >= self.patience_evals:
                should_update = True
                reason = f"patience: {self.patience_count} evals >= {self.patience_threshold:.1%}"
        else:
            self.patience_count = 0

        if should_update:
            self.opponent_version += 1
            self.patience_count = 0

            # Save checkpoint
            ckpt_path = Path(self.checkpoint_dir) / f"opponent_v{self.opponent_version}.zip"
            self.model.save(str(ckpt_path))

            if self.verbose:
                print(f"[SelfPlay] Updated opponent to v{self.opponent_version} ({reason})")

            # Update opponent in all envs
            self._update_opponent()

            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                best_path = Path(self.checkpoint_dir) / "best_model.zip"
                self.model.save(str(best_path))

        return True

    def _evaluate(self):
        """Evaluate current model vs frozen opponent."""
        wins = 0
        total = self.n_eval_games

        # Create a temporary env for evaluation
        env = SnakeBotEnv(league=4, max_turns=200)

        for game_idx in range(total):
            # Alternate which player we are
            we_are_p0 = (game_idx % 2 == 0)
            obs, _ = env.reset(seed=10000 + game_idx)
            terminated = False

            while not terminated:
                # Get our action from the model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, _, _ = env.step(action)

            if reward > 0:
                wins += 1

        env.close()
        return wins / total

    def _update_opponent(self):
        """Update the opponent policy in training environments."""
        # Load the latest checkpoint as the opponent
        ckpt_path = Path(self.checkpoint_dir) / f"opponent_v{self.opponent_version}.zip"
        opponent_model = MaskablePPO.load(str(ckpt_path))

        def make_opponent(obs):
            action, _ = opponent_model.predict(obs, deterministic=False)
            return action

        # Update opponent in all vectorized environments
        for env_idx in range(self.training_env.num_envs):
            env = self.training_env.envs[env_idx]
            # Navigate through Monitor wrapper if present
            while hasattr(env, 'env'):
                if hasattr(env, 'opponent'):
                    env.opponent = make_opponent
                    break
                env = env.env
            else:
                if hasattr(env, 'opponent'):
                    env.opponent = make_opponent


class PeriodicCheckpoint(BaseCallback):
    """Save checkpoint every N steps."""

    def __init__(self, save_freq=50_000, checkpoint_dir=None, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            path = Path(self.checkpoint_dir) / f"step_{self.num_timesteps}.zip"
            self.model.save(str(path))
            if self.verbose:
                print(f"[Checkpoint] Saved {path}")
        return True


def mask_fn(env):
    """Get action masks from the environment (unwrap if needed)."""
    while hasattr(env, 'env'):
        env = env.env
    return env.action_masks()


def make_env(seed_offset=0, league=4, max_turns=200, opponent=None):
    """Create a monitored environment with action masking."""
    def _init():
        env = SnakeBotEnv(league=league, max_turns=max_turns, opponent=opponent)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)  # ActionMasker last so it's visible
        return env
    return _init


def create_model(env, device="auto"):
    """Create MaskablePPO model with CompactUNetExtractor."""
    # Dynamic batch size: divide total buffer into 4 chunks
    n_steps = 1024
    total_states = n_steps * env.num_envs
    dynamic_batch_size = total_states // 4  # 4 chunks

    print(f"Total buffer: {total_states} states. Batch size (1/4 chunk): {dynamic_batch_size}")

    model = MaskablePPO(
        "CnnPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": CompactUNetExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": torch.nn.ReLU,
        },
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=dynamic_batch_size,
        n_epochs=4,  # pairs with 4 chunks
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )
    return model


def train(args):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    n_envs = args.n_envs
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Training config: {n_envs} envs, device={device}")
    print(f"Total steps: {args.total_steps:,}, warmup: {args.warmup_steps:,}")

    # Phase 1: Warmup vs random opponent
    if args.warmup_steps > 0:
        print("\n" + "=" * 60)
        print("Phase 1: Warmup vs random opponent")
        print("=" * 60)

        envs = DummyVecEnv([make_env(seed_offset=i, league=args.league)
                            for i in range(n_envs)])
        model = create_model(envs, device=device)

        model.learn(
            total_timesteps=args.warmup_steps,
            callback=PeriodicCheckpoint(save_freq=50_000),
            progress_bar=True,
        )

        warmup_path = CHECKPOINT_DIR / "warmup_final.zip"
        model.save(str(warmup_path))
        print(f"Warmup saved: {warmup_path}")
        envs.close()
    else:
        warmup_path = None

    # Phase 2: Self-play
    print("\n" + "=" * 60)
    print("Phase 2: Self-play training")
    print("=" * 60)

    remaining_steps = args.total_steps - args.warmup_steps

    envs = DummyVecEnv([make_env(seed_offset=i, league=args.league)
                        for i in range(n_envs)])

    if warmup_path and warmup_path.exists():
        model = MaskablePPO.load(str(warmup_path), env=envs, device=device)
        print(f"Loaded warmup model from {warmup_path}")
    else:
        model = create_model(envs, device=device)

    # Save initial opponent
    opp_path = CHECKPOINT_DIR / "opponent_v0.zip"
    model.save(str(opp_path))

    callbacks = [
        SelfPlayCallback(
            eval_freq=args.eval_freq,
            n_eval_games=args.n_eval_games,
            update_threshold=0.80,
            patience_threshold=0.55,
            patience_evals=3,
            checkpoint_dir=str(CHECKPOINT_DIR),
        ),
        PeriodicCheckpoint(save_freq=50_000, checkpoint_dir=str(CHECKPOINT_DIR)),
    ]

    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = CHECKPOINT_DIR / "self_play_final.zip"
    model.save(str(final_path))
    print(f"\nTraining complete! Final model: {final_path}")

    envs.close()


def main():
    parser = argparse.ArgumentParser(description="PPO Self-Play Training for Snakebot")
    parser.add_argument("--total-steps", type=int, default=1_000_000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--warmup-steps", type=int, default=50_000,
                        help="Warmup steps vs random (default: 50K)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--league", type=int, default=4,
                        help="Game league level (default: 4)")
    parser.add_argument("--eval-freq", type=int, default=20_000,
                        help="Evaluation frequency in steps (default: 20K)")
    parser.add_argument("--n-eval-games", type=int, default=100,
                        help="Number of eval games per round (default: 100)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
