from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class RolloutResult:
    """
    Result returned by the world model backend rollout.

    Attributes:
        predicted_states:
            Includes the initial state at index 0 for each candidate.
        scores:
            Higher is better.
    """
    predicted_states: np.ndarray
    scores: np.ndarray

class WorldModelBackend(ABC):
    """
    Abstract backend interface for imagination-based rollout.
    """

    @abstractmethod
    def rollout(
        self,
        current_state: np.ndarray,
        action_sequences: np.ndarray
    ) -> RolloutResult:
        """
        Roll out candidate action sequences from the given current state

        Args:
            current_state:
                Current state vector
            action_sequences:
                Candidate action sequences

        Return:
            RolloutResult containing predicted states and candidate scores.
        """
        raise NotImplementedError