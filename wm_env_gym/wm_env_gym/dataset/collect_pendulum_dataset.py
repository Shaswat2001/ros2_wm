from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from wm_env_gym.adapters.pendulum_adapter import PendulumAdapter


def collect_dataset(
    env_id: str,
    num_steps: int,
    seed: int,
    output_path: str,
) -> None:
    if env_id != "Pendulum-v1":
        raise ValueError(f"This script currently supports only Pendulum-v1, got {env_id}")

    env = gym.make(env_id)
    adapter = PendulumAdapter()

    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)

    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []

    for step_idx in range(num_steps):
        state = adapter.obs_to_state(obs)

        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)

        action = rng.uniform(
            low=action_low,
            high=action_high,
            size=action_low.shape,
        ).astype(np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = adapter.obs_to_state(next_obs)
        done = bool(terminated or truncated)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(np.float32(reward))
        dones.append(done)

        obs = next_obs

        if done:
            obs, info = env.reset()

        if (step_idx + 1) % 1000 == 0:
            print(f"Collected {step_idx + 1}/{num_steps} transitions")

    env.close()

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    next_states = np.asarray(next_states, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        dones=dones,
        env_id=env_id,
    )

    print(f"Saved dataset to: {output}")
    print(f"states shape: {states.shape}")
    print(f"actions shape: {actions.shape}")
    print(f"next_states shape: {next_states.shape}")
    print(f"rewards shape: {rewards.shape}")
    print(f"dones shape: {dones.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/pendulum_dataset.npz",
    )
    args = parser.parse_args()

    collect_dataset(
        env_id=args.env_id,
        num_steps=args.num_steps,
        seed=args.seed,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()