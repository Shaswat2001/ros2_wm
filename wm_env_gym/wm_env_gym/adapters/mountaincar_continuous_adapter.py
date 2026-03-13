from __future__ import annotations

import numpy as np
import gymnasium as gym

from wm_env_gym.adapters.env_adapter import GymEnvAdapter

class MountainCarContinuousAdapter(GymEnvAdapter):
    """
    Adapter for Gymnasium MountainCarContinuous-v0.

    Observation/state:
        [position, velocity] -> shape [2]

    Action:
        [force] -> shape [1]
    """

    def obs_to_state(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def action_to_env(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).reshape(1)

    def get_state_dim(self) -> int:
        return 2

    def get_action_dim(self) -> int:
        return 1

    def set_env_from_state(self, env: gym.Env, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)

        if state.shape != (2,):
            raise ValueError(
                f"Expected MountainCarContinuous state shape (2,), got {state.shape}"
            )

        env.unwrapped.state = state.copy()