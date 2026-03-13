from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class VisualizerNode(Node):
    def __init__(self) -> None:
        super().__init__("visualizer_node")

        self.declare_parameter("state_dim", 4)
        self.declare_parameter("num_candidates", 32)
        self.declare_parameter("plot_margin", 1.0)

        self._state_dim = int(self.get_parameter("state_dim").value)
        self._num_candidates = int(self.get_parameter("num_candidates").value)
        self._plot_margin = float(self.get_parameter("plot_margin").value)

        self._state: np.ndarray | None = None
        self._best_rollout: np.ndarray | None = None
        self._rollouts: np.ndarray | None = None

        self.create_subscription(
            Float32MultiArray,
            "wm/state",
            self._state_callback,
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            "wm/best_rollout",
            self._best_rollout_callback,
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            "wm/rollouts",
            self._rollouts_callback,
            10,
        )

        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._timer = self.create_timer(0.2, self._draw)

        self.get_logger().info(
            f"Visualizer node started with state_dim={self._state_dim}, "
            f"num_candidates={self._num_candidates}, plot_margin={self._plot_margin}"
        )

    def _state_callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != self._state_dim:
            self.get_logger().warn(
                f"Expected state size {self._state_dim}, got {data.size}"
            )
            return
        self._state = data

    def _best_rollout_callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)

        if data.size % self._state_dim != 0:
            self.get_logger().warn(
                f"best_rollout size {data.size} is not divisible by state_dim {self._state_dim}"
            )
            return

        horizon_plus_one = data.size // self._state_dim
        self._best_rollout = data.reshape(horizon_plus_one, self._state_dim)

    def _rollouts_callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)

        denom = self._num_candidates * self._state_dim
        if denom <= 0 or data.size % denom != 0:
            self.get_logger().warn(
                f"rollouts size {data.size} is not divisible by "
                f"num_candidates * state_dim = {denom}"
            )
            return

        horizon_plus_one = data.size // denom
        self._rollouts = data.reshape(
            self._num_candidates,
            horizon_plus_one,
            self._state_dim,
        )

    def _compute_plot_limits(self) -> tuple[float, float, float, float]:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        if self._state is not None:
            xs.append(np.array([self._state[0], self._state[2]], dtype=np.float32))
            ys.append(np.array([self._state[1], self._state[3]], dtype=np.float32))

        if self._rollouts is not None:
            xs.append(self._rollouts[:, :, 0].reshape(-1))
            ys.append(self._rollouts[:, :, 1].reshape(-1))

        if self._best_rollout is not None:
            xs.append(self._best_rollout[:, 0])
            ys.append(self._best_rollout[:, 1])

        if not xs or not ys:
            return -1.0, 6.0, -1.0, 6.0

        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)

        x_min = float(np.min(all_x)) - self._plot_margin
        x_max = float(np.max(all_x)) + self._plot_margin
        y_min = float(np.min(all_y)) - self._plot_margin
        y_max = float(np.max(all_y)) + self._plot_margin

        # Avoid degenerate axes
        if abs(x_max - x_min) < 1e-3:
            x_min -= 1.0
            x_max += 1.0
        if abs(y_max - y_min) < 1e-3:
            y_min -= 1.0
            y_max += 1.0

        return x_min, x_max, y_min, y_max

    def _draw(self) -> None:
        if self._state is None:
            return

        self._ax.clear()

        x, y, goal_x, goal_y = self._state

        # Draw all candidate rollouts faintly
        if self._rollouts is not None:
            for i in range(self._rollouts.shape[0]):
                rollout_xy = self._rollouts[i, :, 0:2]
                self._ax.plot(
                    rollout_xy[:, 0],
                    rollout_xy[:, 1],
                    alpha=0.15,
                    linewidth=1.0,
                )

        # Draw best rollout prominently
        if self._best_rollout is not None:
            rollout_xy = self._best_rollout[:, 0:2]
            self._ax.plot(
                rollout_xy[:, 0],
                rollout_xy[:, 1],
                "--",
                linewidth=2.5,
                label="best rollout",
            )

        # Draw robot and goal
        self._ax.scatter([x], [y], s=80, label="robot")
        self._ax.scatter([goal_x], [goal_y], marker="*", s=180, label="goal")

        x_min, x_max, y_min, y_max = self._compute_plot_limits()
        self._ax.set_xlim(x_min, x_max)
        self._ax.set_ylim(y_min, y_max)

        self._ax.set_title("wm_runtime 2D Visualizer")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.grid(True)
        self._ax.legend()

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def destroy_node(self) -> None:
        plt.close(self._fig)
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()