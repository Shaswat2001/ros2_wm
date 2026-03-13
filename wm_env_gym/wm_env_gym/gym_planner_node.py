from __future__ import annotations

import numpy as np
import gymnasium as gym
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from wm_interfaces.srv import Imagine
from wm_env_gym.adapter_registry import create_gym_adapter


class GymPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("gym_planner_node")

        self.declare_parameter("env_id", "Pendulum-v1")
        self.declare_parameter("num_candidates", 32)
        self.declare_parameter("horizon", 5)
        self.declare_parameter("action_low", None)
        self.declare_parameter("action_high", None)
        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("max_steps_per_episode", 200)
        self.declare_parameter("seed", 0)

        self._env_id = str(self.get_parameter("env_id").value)
        self._num_candidates = int(self.get_parameter("num_candidates").value)
        self._horizon = int(self.get_parameter("horizon").value)
        action_low_param = self.get_parameter("action_low").value
        action_high_param = self.get_parameter("action_high").value
        timer_period = float(self.get_parameter("timer_period").value)
        self._max_steps_per_episode = int(self.get_parameter("max_steps_per_episode").value)
        self._seed = int(self.get_parameter("seed").value)

        self._adapter = create_gym_adapter(self._env_id)
        self._state_dim = self._adapter.get_state_dim()
        self._action_dim = self._adapter.get_action_dim()

        self._env = gym.make(self._env_id)
        self._obs, self._info = self._env.reset(seed=self._seed)

        env_action_low = np.asarray(self._env.action_space.low, dtype=np.float32)
        env_action_high = np.asarray(self._env.action_space.high, dtype=np.float32)

        if env_action_low.size != self._action_dim or env_action_high.size != self._action_dim:
            raise ValueError(
                "Environment action space shape does not match adapter action_dim: "
                f"low shape={env_action_low.shape}, "
                f"high shape={env_action_high.shape}, "
                f"adapter action_dim={self._action_dim}"
            )

        if action_low_param is None:
            if not np.allclose(env_action_low, env_action_low[0]):
                raise ValueError(
                    "Per-dimension action_low values are not all equal. "
                    "Current planner only supports scalar sampling bounds."
                )
            self._action_low = float(env_action_low[0])
        else:
            self._action_low = float(action_low_param)

        if action_high_param is None:
            if not np.allclose(env_action_high, env_action_high[0]):
                raise ValueError(
                    "Per-dimension action_high values are not all equal. "
                    "Current planner only supports scalar sampling bounds."
                )
            self._action_high = float(env_action_high[0])
        else:
            self._action_high = float(action_high_param)


        self._episode_step = 0
        self._episode_idx = 0
        self._episode_return = 0.0

        self._client = self.create_client(Imagine, "wm/imagine")
        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /wm/imagine service...")

        self._action_pub = self.create_publisher(Float32MultiArray, "wm/action_cmd", 10)
        self._state_pub = self.create_publisher(Float32MultiArray, "wm/state", 10)
        self._rollouts_pub = self.create_publisher(Float32MultiArray, "wm/rollouts", 10)
        self._best_rollout_pub = self.create_publisher(Float32MultiArray, "wm/best_rollout", 10)

        self._publish_state(self._adapter.obs_to_state(self._obs))

        self._timer = self.create_timer(timer_period, self._tick)

        self.get_logger().info(
            f"GymPlannerNode started: env_id={self._env_id}, "
            f"num_candidates={self._num_candidates}, horizon={self._horizon}, "
            f"action_range=[{self._action_low}, {self._action_high}], "
            f"timer_period={timer_period}, max_steps_per_episode={self._max_steps_per_episode}"
        )

    def _tick(self) -> None:
        current_state = self._adapter.obs_to_state(self._obs)

        action_sequences = self._sample_action_sequences()

        request = Imagine.Request()
        request.current_state = current_state.tolist()
        request.action_sequences = action_sequences.reshape(-1).tolist()
        request.num_candidates = self._num_candidates
        request.horizon = self._horizon
        request.state_dim = self._state_dim
        request.action_dim = self._action_dim

        future = self._client.call_async(request)
        future.add_done_callback(
            lambda fut: self._handle_imagine_response(fut, action_sequences)
        )

    def _sample_action_sequences(self) -> np.ndarray:
        return np.random.uniform(
            low=self._action_low,
            high=self._action_high,
            size=(self._num_candidates, self._horizon, self._action_dim),
        ).astype(np.float32)

    def _handle_imagine_response(self, future, action_sequences: np.ndarray) -> None:
        try:
            response = future.result()
            if response is None:
                self.get_logger().error("Imagination service returned no response.")
                return

            scores = np.asarray(response.scores, dtype=np.float32)
            if scores.shape != (self._num_candidates,):
                self.get_logger().error(
                    f"Invalid scores shape: got {scores.shape}, expected {(self._num_candidates,)}"
                )
                return

            predicted_states = np.asarray(response.predicted_states, dtype=np.float32)
            expected_size = self._num_candidates * (self._horizon + 1) * self._state_dim
            if predicted_states.size != expected_size:
                self.get_logger().error(
                    f"Invalid predicted_states size: got {predicted_states.size}, expected {expected_size}"
                )
                return

            predicted_states = predicted_states.reshape(
                self._num_candidates,
                self._horizon + 1,
                self._state_dim,
            )

            self._publish_rollouts(predicted_states)

            best_index = int(response.best_index)
            if not (0 <= best_index < self._num_candidates):
                self.get_logger().error(f"Invalid best_index: {best_index}")
                return

            best_rollout = predicted_states[best_index]
            self._publish_best_rollout(best_rollout)

            best_action = action_sequences[best_index, 0]
            env_action = self._adapter.action_to_env(best_action)

            self._publish_action(best_action)

            obs, reward, terminated, truncated, info = self._env.step(env_action)
            self._obs = obs
            self._info = info
            self._episode_step += 1
            self._episode_return += reward

            current_state = self._adapter.obs_to_state(self._obs)
            self._publish_state(current_state)

            self.get_logger().info(
                f"episode={self._episode_idx} "
                f"step={self._episode_step} "
                f"reward={float(reward):.3f} "
                f"return={self._episode_return:.3f} "
                f"best_score={float(scores[best_index]):.3f}"
            )

            if terminated or truncated or self._episode_step >= self._max_steps_per_episode:
                self._reset_env()

        except Exception as exc:
            self.get_logger().error(f"Failed to process imagination response: {exc}")

    def _reset_env(self) -> None:

        self.get_logger().info(
            f"Episode {self._episode_idx} finished | "
            f"length={self._episode_step} "
            f"return={self._episode_return:.3f}"
        )
            
        self._episode_idx += 1
        self._episode_step = 0
        self._episode_return = 0.0
        self._obs, self._info = self._env.reset()
        current_state = self._adapter.obs_to_state(self._obs)
        self._publish_state(current_state)
        self.get_logger().info(f"Reset environment. episode={self._episode_idx}")

    def _publish_action(self, action: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = action.astype(np.float32).tolist()
        self._action_pub.publish(msg)

    def _publish_state(self, state: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = state.astype(np.float32).tolist()
        self._state_pub.publish(msg)

    def _publish_rollouts(self, predicted_states: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = predicted_states.astype(np.float32).reshape(-1).tolist()
        self._rollouts_pub.publish(msg)

    def _publish_best_rollout(self, best_rollout: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = best_rollout.astype(np.float32).reshape(-1).tolist()
        self._best_rollout_pub.publish(msg)

    def destroy_node(self) -> None:
        self._env.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GymPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()