from __future__ import annotations

import numpy as np
import torch
from torch import nn

from wm_core.backend import RolloutResult, WorldModelBackend

class DynamicsMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        input_dim = state_dim + action_dim
        output_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class LearnedMLPBackend(WorldModelBackend):
    """
    Learned dynamics backend using a trained MLP for one-step prediction.

    Rollout is performed autoregressively:
        s_{t+1} = model(s_t, a_t)
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self._checkpoint_path = str(checkpoint_path)
        self._device = torch.device(device)

        checkpoint = torch.load(self._checkpoint_path, map_location=self._device)

        self._state_dim = int(checkpoint["state_dim"])
        self._action_dim = int(checkpoint["action_dim"])
        self._hidden_dim = int(checkpoint["hidden_dim"])

        self._model = DynamicsMLP(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_dim=self._hidden_dim,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

        self._target_state = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def rollout(
        self,
        current_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> RolloutResult:
        current_state = np.asarray(current_state, dtype=np.float32)
        action_sequences = np.asarray(action_sequences, dtype=np.float32)

        if current_state.shape != (self._state_dim,):
            raise ValueError(
                f"Expected current_state shape ({self._state_dim},), got {current_state.shape}"
            )

        if action_sequences.ndim != 3:
            raise ValueError(
                "Expected action_sequences with shape "
                "[num_candidates, horizon, action_dim]"
            )

        num_candidates, horizon, action_dim = action_sequences.shape

        if action_dim != self._action_dim:
            raise ValueError(
                f"Expected action_dim={self._action_dim}, got {action_dim}"
            )

        predicted_states = np.zeros(
            (num_candidates, horizon + 1, self._state_dim),
            dtype=np.float32,
        )
        predicted_states[:, 0, :] = current_state[None, :]

        with torch.no_grad():
            state_batch = torch.from_numpy(
                np.repeat(current_state[None, :], num_candidates, axis=0)
            ).to(self._device)

            for t in range(horizon):
                action_batch = torch.from_numpy(action_sequences[:, t, :]).to(self._device)
                next_state_batch = self._model(state_batch, action_batch)
                next_state_np = next_state_batch.cpu().numpy().astype(np.float32)

                predicted_states[:, t + 1, :] = next_state_np
                state_batch = next_state_batch

        final_states = predicted_states[:, -1, :]

        state_error = np.linalg.norm(
            final_states - self._target_state[None, :],
            axis=1,
        )

        action_penalty = np.sum(
            np.linalg.norm(action_sequences, axis=2),
            axis=1,
        )

        scores = -state_error - 0.01 * action_penalty
        scores = scores.astype(np.float32)

        return RolloutResult(
            predicted_states=predicted_states,
            scores=scores,
        )