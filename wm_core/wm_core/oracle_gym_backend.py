from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import gymnasium as gym

from wm_core.backend import RolloutResult, WorldModelBackend

@dataclass
class OracleGymConfig:
    env_id: str

class OracleGymBackend(WorldModelBackend):
    """
    Oracle backend that uses a real Gymnasium environment instance to simulate
    candidate action sequences.

    This is a debugging / scaffolding backend, not the final learned backend.
    """

    def __init__(self, env_id: str, adapter) -> None:
        self._env_id = env_id
        self._adapter = adapter
        self._env = gym.make(env_id)
        self._env.reset(seed=0)

    def rollout(
        self,
        current_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> RolloutResult:
        current_state = np.asarray(current_state, dtype=np.float32)
        action_sequences = np.asarray(action_sequences, dtype=np.float32)

        state_dim = self._adapter.get_state_dim()
        action_dim = self._adapter.get_action_dim()

        if current_state.shape != (state_dim,):
            raise ValueError(
                f"Expected current_state shape ({state_dim},), got {current_state.shape}"
            )

        if action_sequences.ndim != 3:
            raise ValueError(
                "Expected action_sequences with shape "
                "[num_candidates, horizon, action_dim]"
            )

        num_candidates, horizon, got_action_dim = action_sequences.shape
        if got_action_dim != action_dim:
            raise ValueError(
                f"Expected action_dim={action_dim}, got {got_action_dim}"
            )

        predicted_states = np.zeros(
            (num_candidates, horizon + 1, state_dim),
            dtype=np.float32,
        )
        scores = np.zeros((num_candidates,), dtype=np.float32)

        predicted_states[:, 0, :] = current_state[None, :]

        for c in range(num_candidates):
            self._adapter.set_env_from_state(self._env, current_state)
            candidate_return = 0.0

            for t in range(horizon):
                env_action = self._adapter.action_to_env(action_sequences[c, t])
                obs, reward, terminated, truncated, _ = self._env.step(env_action)

                next_state = self._adapter.obs_to_state(obs)
                predicted_states[c, t + 1, :] = next_state
                candidate_return += float(reward)

                if terminated or truncated:
                    # Hold final state constant for remaining horizon
                    for k in range(t + 1, horizon):
                        predicted_states[c, k + 1, :] = next_state
                    break

            scores[c] = candidate_return

        return RolloutResult(
            predicted_states=predicted_states,
            scores=scores,
        )

    def close(self) -> None:
        self._env.close()