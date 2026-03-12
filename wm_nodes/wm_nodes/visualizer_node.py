from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class VisualizerNode(Node):
    def __init__(self) -> None:
        super().__init__("visualizer_node")

        self._state = None
        self._best_rollout = None

        self._state_dim = 4
        self._horizon = 5  # matches planner for now

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

        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._timer = self.create_timer(0.2, self._draw)

        self.get_logger().info("Visualizer node started.")

    def _state_callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != 4:
            self.get_logger().warn(f"Expected state size 4, got {data.size}")
            return
        self._state = data

    def _best_rollout_callback(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        expected_size = (self._horizon + 1) * self._state_dim
        if data.size != expected_size:
            self.get_logger().warn(
                f"Expected best_rollout size {expected_size}, got {data.size}"
            )
            return
        self._best_rollout = data.reshape(self._horizon + 1, self._state_dim)

    def _draw(self) -> None:
        if self._state is None:
            return

        self._ax.clear()

        x, y, goal_x, goal_y = self._state

        self._ax.scatter([x], [y], label="robot")
        self._ax.scatter([goal_x], [goal_y], marker="*", s=150, label="goal")

        if self._best_rollout is not None:
            rollout_xy = self._best_rollout[:, 0:2]
            self._ax.plot(
                rollout_xy[:, 0],
                rollout_xy[:, 1],
                "--",
                label="best rollout",
            )

        self._ax.set_title("wm_runtime 2D Visualizer")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_xlim(-1, 6)
        self._ax.set_ylim(-1, 6)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.grid(True)
        self._ax.legend()

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def destroy_node(self):
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