from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

class GymEnvAdapter(ABC):
    @abstractmethod
    def obs_to_state(self, obs: np.ndarray) -> np.ndarray:
        """Convert Gym observation to wm_runtime state vector."""
        raise NotImplementedError

    @abstractmethod
    def action_to_env(self, action: np.ndarray) -> np.ndarray:
        """Convert wm_runtime action vector to environment action."""
        raise NotImplementedError

    @abstractmethod
    def get_state_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_action_dim(self) -> int:
        raise NotImplementedError