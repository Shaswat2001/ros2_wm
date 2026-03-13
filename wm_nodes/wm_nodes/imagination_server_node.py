from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from wm_core.backend_registry import create_backend
from wm_env_gym.adapter_registry import create_gym_adapter
from wm_interfaces.srv import Imagine

class ImaginationServerNode(Node):
    def __init__(self) -> None:
        super().__init__("imagination_server")

        self.declare_parameter("backend_type", "point_mass")
        self.declare_parameter("env_id", "Pendulum-v1")

        backend_type = str(self.get_parameter("backend_type").value)
        env_id = str(self.get_parameter("env_id").value)

        backend_kwargs = {}

        if backend_type == "oracle_gym":

            backend_kwargs["env_id"] = env_id
            backend_kwargs["adapter"] = create_gym_adapter(env_id)

        self._backend = create_backend(
            backend_type,
            **backend_kwargs
        )

        self._service = self.create_service(
            Imagine,
            "wm/imagine",
            self.handle_imagine,
        )

        self.get_logger().info(
            f"Imagination server ready on service /wm/imagine with backend_type={backend_type}"
        )

    def handle_imagine(self, request: Imagine.Request, response: Imagine.Response) -> Imagine.Response:
        try:
            current_state = np.asarray(request.current_state, dtype=np.float32)
            action_sequences = np.asarray(request.action_sequences, dtype=np.float32)

            num_candidates = int(request.num_candidates)
            horizon = int(request.horizon)
            state_dim = int(request.state_dim)
            action_dim = int(request.action_dim)

            if current_state.shape != (state_dim,):
                raise ValueError(
                    f"Expected current_state shape ({state_dim},), got {current_state.shape}"
                )

            expected_action_len = num_candidates * horizon * action_dim
            if action_sequences.size != expected_action_len:
                raise ValueError(
                    "Invalid packed action_sequences length: "
                    f"expected {expected_action_len}, got {action_sequences.size}"
                )

            action_sequences = action_sequences.reshape(
                num_candidates,
                horizon,
                action_dim,
            )

            result = self._backend.rollout(current_state, action_sequences)

            expected_predicted_shape = (num_candidates, horizon + 1, state_dim)
            if result.predicted_states.shape != expected_predicted_shape:
                raise ValueError(
                    f"Backend returned predicted_states shape {result.predicted_states.shape}, "
                    f"expected {expected_predicted_shape}"
                )

            if result.scores.shape != (num_candidates,):
                raise ValueError(
                    f"Backend returned scores shape {result.scores.shape}, "
                    f"expected ({num_candidates},)"
                )

            response.predicted_states = result.predicted_states.reshape(-1).tolist()
            response.scores = result.scores.tolist()
            response.best_index = int(np.argmax(result.scores))

            self.get_logger().info(
                f"Served imagination request: "
                f"{num_candidates=} {horizon=} {state_dim=} {action_dim=} "
                f"best_index={response.best_index}"
            )

            return response

        except Exception as exc:
            self.get_logger().error(f"Failed to handle imagine request: {exc}")
            response.predicted_states = []
            response.scores = []
            response.best_index = 0
            return response
    
    def destroy_node(self) -> None:
        if hasattr(self._backend, "close"):
            self._backend.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ImaginationServerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()