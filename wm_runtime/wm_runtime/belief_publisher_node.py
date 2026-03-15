from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, JointState

from wm_interfaces.msg import BeliefState, ModelStatus, WorldModelObservation
from wm_interfaces.srv import ForwardObservation

class BeliefPublisherNode(Node):
    """
    Sensor bridge: assembles observations and delegates forward() to
    the model_server_node via the /wm/forward_observation service.

    This node does NOT load its own world model — the model_server
    is the single owner of the model instance.  This avoids the
    duplicate-model-in-RAM problem.

    Supported observation modes:
      - 'dataset'  — model_server samples from its loaded dataset
                     (no sensors needed; good for demos)
      - 'image'    — subscribes to an image topic
      - 'vector'   — subscribes to a Float32MultiArray topic
    """

    def __init__(self):
        super().__init__('wm_belief_publisher')

        # ── Parameters ────────────────────────────────────────
        self.declare_parameter('obs_mode', 'dataset')      # image | vector | dataset
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('vector_topic', '/obs_vector')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('dataset_seed', 42)
        self.declare_parameter('dataset_advance', True)     # step seed each tick

        # ── Sensor state ──────────────────────────────────────
        self._latest_image: Optional[np.ndarray] = None
        self._latest_vector: Optional[np.ndarray] = None
        self._latest_joint_state: Optional[JointState] = None
        self._dataset_seed_counter = self.get_parameter('dataset_seed').value
        self._forward_count = 0

        # ── Track model_server readiness ──────────────────────
        self._model_server_ready = False

        # ── Callback group for async service calls ────────────
        self._cb_group = ReentrantCallbackGroup()

        # ── Service client (replaces local model loading) ─────
        self._forward_client = self.create_client(
            ForwardObservation,
            '/wm/forward_observation',
            callback_group=self._cb_group,
        )

        # ── Publishers ────────────────────────────────────────
        self.pub_obs = self.create_publisher(
            WorldModelObservation, '/wm/observation', 10,
        )

        # ── Subscriber: model server status ───────────────────
        self.create_subscription(
            ModelStatus, '/wm/status',
            self._on_model_status, 10,
        )

        # ── Sensor subscriptions ──────────────────────────────
        obs_mode = self.get_parameter('obs_mode').value
        best_effort = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        if obs_mode == 'image':
            topic = self.get_parameter('image_topic').value
            self.create_subscription(
                Image, topic, self._on_image, best_effort,
            )
            self.get_logger().info(f'Subscribed to image: {topic}')

        elif obs_mode == 'vector':
            topic = self.get_parameter('vector_topic').value
            self.create_subscription(
                Float32MultiArray, topic, self._on_vector, 10,
            )
            self.get_logger().info(f'Subscribed to vector: {topic}')

        elif obs_mode == 'dataset':
            self.get_logger().info(
                'Running in dataset mode — the model_server will '
                'sample from its loaded dataset on each forward call.'
            )
        else:
            self.get_logger().error(f'Unknown obs_mode: {obs_mode}')

        # Always subscribe to joint states (optional, used if available)
        jt_topic = self.get_parameter('joint_state_topic').value
        self.create_subscription(
            JointState, jt_topic, self._on_joint_state, 10,
        )

        # ── Publish timer ─────────────────────────────────────
        rate = self.get_parameter('publish_rate_hz').value
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'BeliefPublisherNode ready (mode={obs_mode}, rate={rate}Hz). '
            f'Waiting for model_server...'
        )

    def _on_image(self, msg: Image):
        """Convert sensor_msgs/Image to numpy array."""
        h, w = msg.height, msg.width
        encoding = msg.encoding

        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding in ('rgb8', 'bgr8'):
            img = raw.reshape(h, w, 3)
            if encoding == 'bgr8':
                img = img[:, :, ::-1]  # BGR -> RGB
        elif encoding == 'mono8':
            img = raw.reshape(h, w)
        elif encoding in ('rgba8', 'bgra8'):
            img = raw.reshape(h, w, 4)[:, :, :3]
            if encoding == 'bgra8':
                img = img[:, :, ::-1]
        else:
            img = raw.reshape(h, w, -1) if len(raw) > h * w else raw.reshape(h, w)

        self._latest_image = img

    def _on_vector(self, msg: Float32MultiArray):
        """Store latest observation vector."""
        self._latest_vector = np.array(msg.data, dtype=np.float32)

    def _on_joint_state(self, msg: JointState):
        """Store latest joint state."""
        self._latest_joint_state = msg

    def _on_model_status(self, msg: ModelStatus):
        """Track whether the model server has a model loaded."""
        was_ready = self._model_server_ready
        self._model_server_ready = msg.ready

        if msg.ready and not was_ready:
            self.get_logger().info(
                f'Model server ready: {msg.model_type}/{msg.environment} '
                f'on {msg.device}'
            )

    def _tick(self):
        """Called at publish_rate_hz. Assembles obs, sends to model_server."""
        # Wait until model_server has a model loaded
        if not self._model_server_ready:
            return

        # Wait until the service is available (non-blocking check)
        if not self._forward_client.service_is_ready():
            return

        obs_mode = self.get_parameter('obs_mode').value

        # ── Dataset mode: model_server owns the dataset too,
        #    so we just send a seed and an empty obs array.
        #    The model_server's _forward_observation will use
        #    its own dataset to sample the obs.
        # ── Sensor modes: we send real sensor data.
        obs = self._assemble_observation()

        # In dataset mode we always have something to send (the seed).
        # In sensor modes, obs may be None if no data has arrived yet.
        if obs is None and obs_mode != 'dataset':
            return

        # Build the service request
        request = ForwardObservation.Request()
        request.seed = self._dataset_seed_counter

        if obs is not None:
            request.obs_data = obs.flatten().astype(np.float32).tolist()
            request.obs_shape = list(obs.shape)
        else:
            # Dataset mode with no local obs — send empty arrays.
            # The model_server will sample from its dataset using the seed.
            request.obs_data = []
            request.obs_shape = []

        # Async call — fire and process result in callback
        future = self._forward_client.call_async(request)
        future.add_done_callback(self._on_forward_response)

        # Advance dataset seed for next tick
        if (obs_mode == 'dataset'
                and self.get_parameter('dataset_advance').value):
            self._dataset_seed_counter += 1

    def _on_forward_response(self, future):
        """Handle the ForwardObservation service response."""
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warn(
                    f'Forward failed: {response.message}'
                )
                return

            self._forward_count += 1

            if self._forward_count % 100 == 0:
                self.get_logger().info(
                    f'forward() #{self._forward_count} via model_server'
                )

        except Exception as e:
            self.get_logger().error(f'Forward service call failed: {e}')

    def _assemble_observation(self) -> Optional[np.ndarray]:
        """
        Build the numpy observation array from the latest sensor data.
        Returns None if no data is available yet.

        In dataset mode, returns None — the model_server handles
        dataset sampling internally.
        """
        obs_mode = self.get_parameter('obs_mode').value

        if obs_mode == 'dataset':
            # The model_server owns the dataset; nothing to assemble.
            return None

        elif obs_mode == 'image':
            if self._latest_image is None:
                return None
            return self._latest_image.copy()

        elif obs_mode == 'vector':
            if self._latest_vector is None:
                return None
            return self._latest_vector.copy()

        return None

def main(args=None):
    rclpy.init(args=args)
    node = BeliefPublisherNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()