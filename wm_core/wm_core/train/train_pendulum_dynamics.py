from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


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


def load_dataset(dataset_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(dataset_path)

    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    next_states = data["next_states"].astype(np.float32)

    return states, actions, next_states


def train(
    dataset_path: str,
    output_path: str,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    num_epochs: int,
    val_ratio: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    states, actions, next_states = load_dataset(dataset_path)

    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    state_tensor = torch.from_numpy(states)
    action_tensor = torch.from_numpy(actions)
    next_state_tensor = torch.from_numpy(next_states)

    dataset = TensorDataset(state_tensor, action_tensor, next_state_tensor)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = DynamicsMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_states, batch_actions, batch_next_states in train_loader:
            pred_next_states = model(batch_states, batch_actions)
            loss = criterion(pred_next_states, batch_next_states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_states.shape[0]
            train_loss_sum += float(loss.item()) * batch_size_actual
            train_count += batch_size_actual

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_states, batch_actions, batch_next_states in val_loader:
                pred_next_states = model(batch_states, batch_actions)
                loss = criterion(pred_next_states, batch_next_states)

                batch_size_actual = batch_states.shape[0]
                val_loss_sum += float(loss.item()) * batch_size_actual
                val_count += batch_size_actual

        val_loss = val_loss_sum / max(val_count, 1)

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "hidden_dim": hidden_dim,
                    "best_val_loss": best_val_loss,
                },
                output,
            )

    print(f"Saved best model to: {output}")
    print(f"Best validation loss: {best_val_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/pendulum_dataset.npz",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoints/pendulum_dynamics.pt",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()