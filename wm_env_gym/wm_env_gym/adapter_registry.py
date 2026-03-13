from __future__ import annotations

from wm_env_gym.adapters.pendulum_adapter import PendulumAdapter
from wm_env_gym.adapters.mountaincar_continuous_adapter import MountainCarContinuousAdapter

def create_gym_adapter(env_id: str):
    """
    Factory for Gym environment adapters.
    """

    if env_id == "Pendulum-v1":
        return PendulumAdapter()

    if env_id == "MountainCarContinuous-v0":
        return MountainCarContinuousAdapter()
    
    raise ValueError(f"Unsupported Gym env_id: {env_id}")