"""Absorbing-state adapter for unbounded geometric rollouts."""

from __future__ import annotations

import numpy as np


class AbsorbingStateAdapter:
    """Continue terminated trajectories with zero rewards and a forced no-op."""

    def __init__(self, env):
        self.env = env
        self.absorbed = False
        self.last_observation = None

    def reset(self, *, seed=None):
        observation, info = self.env.reset(seed=seed)
        self.absorbed = False
        self.last_observation = observation
        return observation, info

    def reset_to_initial_state(self, snapshot, *, seed=None):
        observation, info = self.env.restore_initial_state(snapshot, seed=seed)
        self.absorbed = False
        self.last_observation = observation
        return observation, info

    def action_masks(self):
        if not self.absorbed:
            return np.asarray(self.env.action_masks(), dtype=bool)
        masks = []
        for action_count in self.env.action_space.nvec:
            masks.extend([True] + [False] * (int(action_count) - 1))
        return np.asarray(masks, dtype=bool)

    def step(self, action):
        if self.absorbed:
            return self.last_observation, 0.0, True, False, {"absorbing": True}
        observation, reward, terminated, truncated, info = self.env.step(action)
        if truncated:
            raise RuntimeError("Artificial truncation is disabled for geometric rollouts.")
        self.last_observation = observation
        self.absorbed = bool(terminated)
        return observation, float(reward), bool(terminated), False, info

    def close(self):
        self.env.close()
