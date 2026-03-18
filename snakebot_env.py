"""
Snakebot Gymnasium Environment — thin ctypes wrapper around libengine.so.

Observation: 7 channels × 64 × 64 (padded) spatial tensor.
Action: MultiDiscrete([4, 4, 4, 4]) — direction per bird (up to 4 birds).
Reward: terminal only (+1 win, -1 loss, 0 draw).
"""

import ctypes
import os
import random
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium import spaces

# Locate shared library
_ENGINE_DIR = Path(__file__).parent / "engine"
_LIB_PATH = _ENGINE_DIR / "libengine.so"


def _load_lib():
    lib = ctypes.CDLL(str(_LIB_PATH))

    # --- Lifecycle ---
    lib.engine_create.argtypes = []
    lib.engine_create.restype = ctypes.c_void_p
    lib.engine_destroy.argtypes = [ctypes.c_void_p]
    lib.engine_destroy.restype = None

    # --- Setup ---
    lib.engine_set_grid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int)]
    lib.engine_set_grid.restype = None
    lib.engine_add_bird.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.engine_add_bird.restype = None
    lib.engine_set_bird_direction.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.engine_set_bird_direction.restype = None
    lib.engine_add_apple.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.engine_add_apple.restype = None
    lib.engine_clear_apples.argtypes = [ctypes.c_void_p]
    lib.engine_clear_apples.restype = None
    lib.engine_set_turn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.engine_set_turn.restype = None
    lib.engine_set_losses.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.engine_set_losses.restype = None

    # --- Reset ---
    lib.engine_reset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.engine_reset.restype = None

    # --- Step ---
    lib.engine_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.engine_step.restype = ctypes.c_int

    # --- Queries ---
    lib.engine_get_turn.argtypes = [ctypes.c_void_p]
    lib.engine_get_turn.restype = ctypes.c_int
    lib.engine_get_width.argtypes = [ctypes.c_void_p]
    lib.engine_get_width.restype = ctypes.c_int
    lib.engine_get_height.argtypes = [ctypes.c_void_p]
    lib.engine_get_height.restype = ctypes.c_int
    lib.engine_is_wall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.engine_is_wall.restype = ctypes.c_int
    lib.engine_apple_count.argtypes = [ctypes.c_void_p]
    lib.engine_apple_count.restype = ctypes.c_int
    lib.engine_get_apples.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.engine_get_apples.restype = None
    lib.engine_bird_count.argtypes = [ctypes.c_void_p]
    lib.engine_bird_count.restype = ctypes.c_int
    lib.engine_get_bird.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int)
    ]
    lib.engine_get_bird.restype = None
    lib.engine_body_score.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.engine_body_score.restype = ctypes.c_int
    lib.engine_get_losses.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.engine_get_losses.restype = ctypes.c_int
    lib.engine_is_game_over.argtypes = [ctypes.c_void_p]
    lib.engine_is_game_over.restype = ctypes.c_int
    lib.engine_is_terminal.argtypes = [ctypes.c_void_p]
    lib.engine_is_terminal.restype = ctypes.c_int
    lib.engine_winner.argtypes = [ctypes.c_void_p]
    lib.engine_winner.restype = ctypes.c_int

    # --- Legal moves ---
    lib.engine_legal_moves.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int)]
    lib.engine_legal_moves.restype = ctypes.c_int

    # --- Spatial obs ---
    lib.engine_get_spatial_obs.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_float)]
    lib.engine_get_spatial_obs.restype = None

    # --- Max turns ---
    lib.engine_set_max_turns.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.engine_set_max_turns.restype = None

    return lib


# Module-level singleton for the shared library
_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib


class SnakeBotEnv(gymnasium.Env):
    """
    Snakebot game environment using C++ engine via ctypes.

    Observation: (7, 64, 64) float32 spatial tensor, CHW order.
    Action: MultiDiscrete([4]*4) — one direction per bird (up to 4 per player).
             Directions: 0=N, 1=E, 2=S, 3=W.
    Reward: terminal only. +1 win, -1 loss, 0 draw.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, league=4, max_turns=200, opponent=None, render_mode=None):
        super().__init__()

        self.lib = _get_lib()
        self.handle = self.lib.engine_create()
        self.league = league
        self.max_turns = max_turns
        self.opponent = opponent  # callable: obs → action array, or None (random)
        self.render_mode = render_mode

        # Fixed 64×64 padded observation
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7, 64, 64), dtype=np.float32
        )

        # 4 directions per bird, up to 4 birds per player
        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4])

        # Cached bird info
        self._my_bird_ids = []
        self._opp_bird_ids = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = seed if seed is not None else random.randint(0, 999999)
        self.lib.engine_reset(self.handle, game_seed, self.league)
        self.lib.engine_set_max_turns(self.handle, self.max_turns)
        self._cache_bird_ids()
        obs = self._get_obs(viewer=0)
        return obs, {}

    def _get_head_positions(self, player=0):
        """Returns a dictionary of {bird_id: (head_x, head_y)} for alive birds."""
        positions = {}
        n = self.lib.engine_bird_count(self.handle)

        for i in range(n):
            bid = ctypes.c_int()
            owner = ctypes.c_int()
            alive = ctypes.c_int()
            length = ctypes.c_int()
            # Create a C-array to hold the body coordinates (up to 200 parts * 2 for X/Y)
            body_xy = (ctypes.c_int * 400)()

            # Call the actual C++ function exported in your engine
            self.lib.engine_get_bird(
                self.handle, i, 
                ctypes.byref(bid), ctypes.byref(owner), ctypes.byref(alive),
                body_xy, ctypes.byref(length)
            )

            # If the bird belongs to us and is alive
            if owner.value == player and alive.value == 1 and length.value > 0:
                # The Head is always the first two coordinates in the array!
                hx = body_xy[0]
                hy = body_xy[1]
                positions[bid.value] = (hx, hy)
                
        return positions
    
    # def step(self, action):
    #     # 1. Track State BEFORE the move
    #     my_old_score = self.lib.engine_body_score(self.handle, 0)
    #     prev_positions = self._get_head_positions(player=0)
        
    #     # Build actions
    #     my_actions = self._build_actions(action, player=0)

    #     # Opponent actions
    #     if self.opponent is not None:
    #         opp_obs = self._get_obs(viewer=1)
    #         opp_masks = self.action_masks(player=1) # Get opponent masks
    #         opp_act = self.opponent(opp_obs, opp_masks) # Pass them in!
    #         opp_actions = self._build_actions(opp_act, player=1)
    #     else:
    #         opp_actions = self._random_actions(player=1)

    #     # Merge all actions and step the C++ engine
    #     all_acts = my_actions + opp_actions
    #     n = len(all_acts) // 2
    #     arr = (ctypes.c_int * len(all_acts))(*all_acts)
    #     terminal = self.lib.engine_step(self.handle, arr, n)

    #     # 2. Track State AFTER the move
    #     my_new_score = self.lib.engine_body_score(self.handle, 0)
    #     curr_positions = self._get_head_positions(player=0)
        
    #     # 3. CALCULATE SHAPED REWARDS
        
    #     # A. The Apple Breadcrumbs (+0.2)
    #     apples_eaten = max(0, my_new_score - my_old_score)
    #     step_reward = apples_eaten * 1

    #     # B. The "Hot Stove" Stuck Penalty (-0.5)
    #     position_penalty = 0.0
    #     for bid, prev_pos in prev_positions.items():
    #         if bid in curr_positions:  # If the bird is still alive
    #             if prev_pos == curr_positions[bid]:
    #                 position_penalty -= 0.5  # Punish grinding against a wall

    #     # 4. Calculate Final Terminal Reward (+5.0, -5.0, 0.0)
    #     obs = self._get_obs(viewer=0)
    #     terminal_reward = self._compute_reward(bool(terminal))
        
    #     # 5. Combine everything!
    #     total_reward = step_reward + position_penalty + terminal_reward

    #     terminated = bool(terminal)
    #     truncated = False

    #     return obs, total_reward, terminated, truncated, {}
    
    def step(self, action):
        # Build my actions
        my_actions = self._build_actions(action, player=0)

        # Opponent actions (KEEPING THE MASK FIX!)
        if self.opponent is not None:
            opp_obs = self._get_obs(viewer=1)
            opp_masks = self.action_masks(player=1) 
            opp_act = self.opponent(opp_obs, opp_masks) 
            opp_actions = self._build_actions(opp_act, player=1)
        else:
            opp_actions = self._random_actions(player=1)

        # Merge all actions and step the C++ engine
        all_acts = my_actions + opp_actions
        n = len(all_acts) // 2
        arr = (ctypes.c_int * len(all_acts))(*all_acts)
        terminal = self.lib.engine_step(self.handle, arr, n)

        # Pure terminal reward calculation
        obs = self._get_obs(viewer=0)
        reward = self._compute_reward(bool(terminal))
        
        terminated = bool(terminal)
        truncated = False

        return obs, reward, terminated, truncated, {}

    # def step(self, action):
    #     # Build my actions
    #     my_actions = self._build_actions(action, player=0)

    #     # Opponent actions
    #     if self.opponent is not None:
    #         opp_obs = self._get_obs(viewer=1)
    #         opp_act = self.opponent(opp_obs)
    #         opp_actions = self._build_actions(opp_act, player=1)
    #     else:
    #         opp_actions = self._random_actions(player=1)

    #     # Merge all actions
    #     all_acts = my_actions + opp_actions
    #     n = len(all_acts) // 2
    #     arr = (ctypes.c_int * len(all_acts))(*all_acts)
    #     terminal = self.lib.engine_step(self.handle, arr, n)

    #     obs = self._get_obs(viewer=0)
    #     reward = self._compute_reward(bool(terminal))
    #     terminated = bool(terminal)
    #     truncated = False

    #     return obs, reward, terminated, truncated, {}

    def close(self):
        if self.handle is not None:
            self.lib.engine_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    # ---- Internal helpers ----

    def _cache_bird_ids(self):
        """Cache bird IDs for both players."""
        self._my_bird_ids = []
        self._opp_bird_ids = []
        n = self.lib.engine_bird_count(self.handle)
        for i in range(n):
            bid = ctypes.c_int()
            owner = ctypes.c_int()
            alive = ctypes.c_int()
            body = (ctypes.c_int * 400)()
            blen = ctypes.c_int()
            self.lib.engine_get_bird(self.handle, i,
                                      ctypes.byref(bid), ctypes.byref(owner),
                                      ctypes.byref(alive), body, ctypes.byref(blen))
            if owner.value == 0:
                self._my_bird_ids.append(bid.value)
            else:
                self._opp_bird_ids.append(bid.value)

    def _get_obs(self, viewer=0):
        """Get 7×64×64 padded spatial observation."""
        W = self.lib.engine_get_width(self.handle)
        H = self.lib.engine_get_height(self.handle)

        # Allocate grid-sized buffer for C++ to fill
        buf_size = 7 * W * H
        buf = (ctypes.c_float * buf_size)()
        self.lib.engine_get_spatial_obs(self.handle, viewer, buf)

        # Convert to numpy and pad to 64×64
        raw = np.ctypeslib.as_array(buf).reshape(7, H, W).copy()
        obs = np.zeros((7, 64, 64), dtype=np.float32)
        obs[:, :H, :W] = raw
        return obs

    def _build_actions(self, action, player=0):
        """Convert MultiDiscrete action [d0, d1, d2, d3] to flat action list."""
        # Loop over the ORIGINAL list to keep index 'i' aligned with the action array
        bird_ids = self._my_bird_ids if player == 0 else self._opp_bird_ids
        alive_birds = self.get_alive_bird_ids(player=player)
        acts = []

        for i, bid in enumerate(bird_ids):
            # Skip sending actions for dead birds to prevent C++ Segfaults
            if bid not in alive_birds:
                continue

            if i < len(action):
                d = int(action[i]) % 4
            else:
                d = random.randint(0, 3)
            acts.extend([bid, d])
        return acts

    def _random_actions(self, player=1):
        """Random actions for a player's birds."""
        bird_ids = self._my_bird_ids if player == 0 else self._opp_bird_ids
        alive_birds = self.get_alive_bird_ids(player=player)
        acts = []
        for bid in bird_ids:
            if bid not in alive_birds:
                continue
            acts.extend([bid, random.randint(0, 3)])
        return acts

    def _compute_reward(self, terminal):
        """Terminal-only reward: +1 win, -1 loss, 0 draw."""
        if not terminal:
            return 0.0

        my_score = self.lib.engine_body_score(self.handle, 0)
        opp_score = self.lib.engine_body_score(self.handle, 1)

        # You only get +1 if you ACTUALLY ate more apples than the opponent.
        if my_score > opp_score:
            return 1.0
        if my_score < opp_score:
            return -1.0

        # If the score is tied (including a 0-0 tie), nobody gets a win.
        # This completely destroys the "just outlive the idiot" strategy.
        return 0.0

    def get_alive_bird_ids(self, player=0):
        """Get list of alive bird IDs for a player."""
        bird_ids = self._my_bird_ids if player == 0 else self._opp_bird_ids
        alive = []
        n = self.lib.engine_bird_count(self.handle)
        for i in range(n):
            bid = ctypes.c_int()
            owner = ctypes.c_int()
            a = ctypes.c_int()
            body = (ctypes.c_int * 400)()
            blen = ctypes.c_int()
            self.lib.engine_get_bird(self.handle, i,
                                      ctypes.byref(bid), ctypes.byref(owner),
                                      ctypes.byref(a), body, ctypes.byref(blen))
            if bid.value in bird_ids and a.value:
                alive.append(bid.value)
        return alive

    def action_masks(self, player=0):
        # Support both players
        mask = np.zeros(16, dtype=bool)
        legal_arr = (ctypes.c_int * 5)()
        
        alive_birds = self.get_alive_bird_ids(player=player)
        bird_ids = self._my_bird_ids if player == 0 else self._opp_bird_ids

        for i, bid in enumerate(bird_ids):
            if i < 4:
                start_idx = i * 4
                if bid not in alive_birds:
                    mask[start_idx : start_idx + 4] = True
                    continue

                n_moves = self.lib.engine_legal_moves(self.handle, bid, legal_arr)
                safe_moves = []
                for m in range(n_moves):
                    move = legal_arr[m]
                    if move != 4:  # 4 is KEEP
                        safe_moves.append(move)

                if safe_moves:
                    for sm in safe_moves:
                        mask[start_idx + sm] = True
                else:
                    mask[start_idx] = True
        return mask
