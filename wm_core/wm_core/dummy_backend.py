from __future__ import annotations

import numpy as np

from wm_core.backend import RolloutResult, WorldModelBackend


class DummyPointMassBackend(WorldModelBackend):
    """
    Simple deterministic point-mass backend for early wm_runtime development.

    State format:
        [x, y, goal_x, goal_y]

    Action format:
        [dx, dy]

    Transition:
        x_{t+1} = x_t + dx
        y_{t+1} = y_t + dy
        goal remains fixed
    """

    def rollout(
        self,
        current_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> RolloutResult:
        """
        Roll out candidate action sequences from the current state.
        """
        current_state = np.asarray(current_state, dtype=np.float32)
        action_sequences = np.asarray(action_sequences, dtype=np.float32)

        if current_state.shape != (4,):
            raise ValueError(
                f"Expected current_state shape (4,), got {current_state.shape}"
            )

        if action_sequences.ndim != 3:
            raise ValueError(
                "Expected action_sequences with shape "
                "[num_candidates, horizon, action_dim]"
            )

        num_candidates, horizon, action_dim = action_sequences.shape

        if action_dim != 2:
            raise ValueError(
                f"Expected action_dim=2 for [dx, dy], got {action_dim}"
            )

        predicted_states = np.zeros(
            (num_candidates, horizon + 1, 4),
            dtype=np.float32,
        )

        # Initial state for every candidate
        predicted_states[:, 0, :] = current_state[None, :]

        goal = current_state[2:4]

        for c in range(num_candidates):
            state = current_state.copy()

            for t in range(horizon):
                action = action_sequences[c, t]

                # Update robot position
                state[0:2] = state[0:2] + action

                # Goal remains unchanged
                state[2:4] = goal

                predicted_states[c, t + 1, :] = state

        final_positions = predicted_states[:, -1, 0:2]
        final_distances = np.linalg.norm(final_positions - goal[None, :], axis=1)

        action_energy = np.sum(np.linalg.norm(action_sequences, axis=2), axis=1)

        scores = -final_distances - 0.01 * action_energy
        scores = scores.astype(np.float32)

        return RolloutResult(
            predicted_states=predicted_states,
            scores=scores,
        )