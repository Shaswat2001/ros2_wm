from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from wm_interfaces.srv import Imagine


class PlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("planner_node")

        self._client = self.create_client(Imagine, "wm/imagine")
        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /wm/imagine service...")

        self._action_pub = self.create_publisher(Float32MultiArray, "wm/action_cmd", 10)

        self._state_pub = self.create_publisher(Float32MultiArray, "wm/state", 10)

        self._rollouts_pub = self.create_publisher(Float32MultiArray, "wm/rollouts", 10)

        self._best_rollout_pub = self.create_publisher(Float32MultiArray, "wm/best_rollout", 10)

        # State format: [x, y, goal_x, goal_y]
        self._current_state = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        self._publish_state()

        self._num_candidates = 32
        self._horizon = 5
        self._state_dim = 4
        self._action_dim = 2
        self._action_low = -1.0
        self._action_high = 1.0

        self._step_count = 0
        self._max_steps = 30
        self._goal_tolerance = 0.5

        self._timer = self.create_timer(0.5, self._tick)

        self.get_logger().info("Planner node started.")

    def _tick(self) -> None:
        if self._step_count >= self._max_steps:
            self.get_logger().info("Reached max steps. Stopping planner loop.")
            self._timer.cancel()
            return

        if self._goal_reached():
            self.get_logger().info("Goal reached. Stopping planner loop.")
            self._timer.cancel()
            return

        action_sequences = self._sample_action_sequences()

        request = Imagine.Request()
        request.current_state = self._current_state.tolist()
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

            self._publish_action(best_action)
            self._apply_action(best_action)
            self._publish_state()

            self._step_count += 1

            dist_to_goal = self._distance_to_goal()
            best_score = float(scores[best_index])

            self.get_logger().info(
                f"step={self._step_count} "
                f"state={self._current_state.tolist()} "
                f"best_action={best_action.tolist()} "
                f"best_score={best_score:.4f} "
                f"dist_to_goal={dist_to_goal:.4f}"
            )

        except Exception as exc:
            self.get_logger().error(f"Failed to process imagination response: {exc}")

    def _publish_action(self, action: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = action.astype(np.float32).tolist()
        self._action_pub.publish(msg)

    def _publish_state(self) -> None:
        msg = Float32MultiArray()
        msg.data = self._current_state.astype(np.float32).tolist()
        self._state_pub.publish(msg)

    def _publish_rollouts(self, predicted_states: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = predicted_states.astype(np.float32).reshape(-1).tolist()
        self._rollouts_pub.publish(msg)

    def _publish_best_rollout(self, best_rollout: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = best_rollout.astype(np.float32).reshape(-1).tolist()
        self._best_rollout_pub.publish(msg)

    def _apply_action(self, action: np.ndarray) -> None:
        self._current_state[0:2] = self._current_state[0:2] + action

    def _distance_to_goal(self) -> float:
        pos = self._current_state[0:2]
        goal = self._current_state[2:4]
        return float(np.linalg.norm(pos - goal))

    def _goal_reached(self) -> bool:
        return self._distance_to_goal() < self._goal_tolerance


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()