from __future__ import annotations

import numpy as np
import gymnasium as gym

from wm_env_gym.adapters.env_adapter import GymEnvAdapter

class PendulumAdapter(GymEnvAdapter):
    """
    Adapter for Gymnasium Pendulum-v1.
    
    Observation/state:
        [cos(theta), sin(theta), theta_dot]  -> shape [3]

    Action:
        [torque] -> shape [1]
    """

    def obs_to_state(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def action_to_env(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).reshape(1)

    def get_state_dim(self) -> int:
        return 3

    def get_action_dim(self) -> int:
        return 1

    def set_env_from_state(self, env: gym.Env, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (3,):
            raise ValueError(f"Expected Pendulum state shape (3,), got {state.shape}")

        cos_theta, sin_theta, theta_dot = state
        theta = np.arctan2(sin_theta, cos_theta).astype(np.float32)

        env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)