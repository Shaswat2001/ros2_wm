from __future__ import annotations

import numpy as np

from wm_env_gym.adapters.env_adapter import GymEnvAdapter

class PendulumAdapter(GymEnvAdapter):
    """
    Adapter for Gymnasium Pendulum-v1.

    Observation:
        [cos(theta), sin(theta), theta_dot]  -> shape [3]

    Action:
        [torque] -> shape [1]
    """

    def obs_to_state(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def action_to_env(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(1)
        return action

    def get_state_dim(self) -> int:
        return 3

    def get_action_dim(self) -> int:
        return 1