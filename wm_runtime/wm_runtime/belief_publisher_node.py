from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, JointState

from wm_interfaces.msg import BeliefState, ModelStatus, WorldModelObservation
from wm_interfaces.srv import ForwardObservation, StepAction

class BeliefPublisherNode(Node):
    """
    Sensor bridge that drives the world model through its correct lifecycle:

        1. Wait for model_server to be ready
        2. Call /wm/forward_observation ONCE to initialize the model
        3. On every tick, call /wm/step_action with a sampled action
           to step the model and get back recon frames + rewards

    This node does NOT load its own world model. The model_server
    is the single owner. In dataset mode, the model_server samples
    the initial observation from its loaded dataset.
    """

    def __init__(self):
        super().__init__('wm_belief_publisher')

        # ── Parameters ────────────────────────────────────────
        self.declare_parameter('obs_mode', 'dataset')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('vector_topic', '/obs_vector')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('dataset_seed', 42)
        self.declare_parameter('dataset_advance', True)
        self.declare_parameter('default_action_space', 'discrete')
        self.declare_parameter('default_num_actions', 18)
        self.declare_parameter('continuous_predict', False)  # If true, step with random actions continuously

        # ── State ─────────────────────────────────────────────
        self._latest_image: Optional[np.ndarray] = None
        self._latest_vector: Optional[np.ndarray] = None
        self._latest_joint_state: Optional[JointState] = None
        self._dataset_seed_counter = self.get_parameter('dataset_seed').value
        self._step_count = 0

        self._model_server_ready = False
        self._model_initialized = False  # True after forward() call
        self._pending_call = False       # Prevent overlapping async calls

        # ── Callback group for async service calls ────────────
        self._cb_group = ReentrantCallbackGroup()

        # ── Service clients ───────────────────────────────────
        self._forward_client = self.create_client(
            ForwardObservation,
            '/wm/forward_observation',
            callback_group=self._cb_group,
        )
        self._step_client = self.create_client(
            StepAction,
            '/wm/step_action',
            callback_group=self._cb_group,
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
            self.create_subscription(Image, topic, self._on_image, best_effort)
            self.get_logger().info(f'Subscribed to image: {topic}')
        elif obs_mode == 'vector':
            topic = self.get_parameter('vector_topic').value
            self.create_subscription(Float32MultiArray, topic, self._on_vector, 10)
            self.get_logger().info(f'Subscribed to vector: {topic}')
        elif obs_mode == 'dataset':
            self.get_logger().info(
                'Running in dataset mode — model_server samples observations.'
            )
        else:
            self.get_logger().error(f'Unknown obs_mode: {obs_mode}')

        jt_topic = self.get_parameter('joint_state_topic').value
        self.create_subscription(JointState, jt_topic, self._on_joint_state, 10)

        # ── Tick timer ────────────────────────────────────────
        rate = self.get_parameter('publish_rate_hz').value
        self.create_timer(1.0 / rate, self._tick)

        continuous = self.get_parameter('continuous_predict').value
        self.get_logger().info(
            f'BeliefPublisherNode ready (mode={obs_mode}, rate={rate}Hz, '
            f'continuous_predict={continuous}). Waiting for model_server...'
        )

    def _on_image(self, msg: Image):
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding
        if enc in ('rgb8', 'bgr8'):
            img = raw.reshape(h, w, 3)
            if enc == 'bgr8':
                img = img[:, :, ::-1]
        elif enc == 'mono8':
            img = raw.reshape(h, w)
        elif enc in ('rgba8', 'bgra8'):
            img = raw.reshape(h, w, 4)[:, :, :3]
            if enc == 'bgra8':
                img = img[:, :, ::-1]
        else:
            img = raw.reshape(h, w, -1) if len(raw) > h * w else raw.reshape(h, w)
        self._latest_image = img

    def _on_vector(self, msg: Float32MultiArray):
        self._latest_vector = np.array(msg.data, dtype=np.float32)

    def _on_joint_state(self, msg: JointState):
        self._latest_joint_state = msg

    def _on_model_status(self, msg: ModelStatus):
        was_ready = self._model_server_ready
        self._model_server_ready = msg.ready
        if msg.ready and not was_ready:
            self.get_logger().info(
                f'Model server ready: {msg.model_type}/{msg.environment} on {msg.device}'
            )
            # Reset initialization so we re-forward on model swap
            self._model_initialized = False

    def _tick(self):
        if not self._model_server_ready:
            return
        if self._pending_call:
            return  # Previous async call still in flight

        if not self._model_initialized:
            self._call_forward()
        elif self.get_parameter('continuous_predict').value:
            self._call_step()

    def _call_forward(self):
        """Call /wm/forward_observation ONCE to initialize the model state."""
        if not self._forward_client.service_is_ready():
            self.get_logger().debug('Waiting for /wm/forward_observation service...')
            return
        if not self._step_client.service_is_ready():
            self.get_logger().debug('Waiting for /wm/step_action service...')
            return

        obs_mode = self.get_parameter('obs_mode').value
        request = ForwardObservation.Request()
        request.seed = self._dataset_seed_counter

        obs = self._assemble_observation()
        if obs is not None:
            request.obs_data = obs.flatten().astype(np.float32).tolist()
            request.obs_shape = list(obs.shape)
        else:
            # Dataset mode — model_server samples its own obs
            request.obs_data = []
            request.obs_shape = []

        self._pending_call = True
        future = self._forward_client.call_async(request)
        future.add_done_callback(self._on_forward_response)

    def _on_forward_response(self, future):
        self._pending_call = False
        try:
            response = future.result()
            if response.success:
                self._model_initialized = True
                self._step_count = 0
                if self.get_parameter('continuous_predict').value:
                    self.get_logger().info(
                        'Model initialized via forward(). Starting continuous predict loop.'
                    )
                else:
                    self.get_logger().info(
                        'Model initialized via forward(). Idle mode — '
                        'use /wm/plan_action, /wm/step_action, or set '
                        'continuous_predict:=true to stream frames.'
                    )
            else:
                self.get_logger().warn(f'Forward failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Forward service call failed: {e}')

    def _call_step(self):
        """Call /wm/step_action to step the model with a random action."""
        if not self._step_client.service_is_ready():
            return

        request = StepAction.Request()
        request.action_space = self.get_parameter('default_action_space').value
        num_actions = self.get_parameter('default_num_actions').value

        # Sample a random action
        rng = np.random.default_rng(self._dataset_seed_counter + self._step_count)
        if request.action_space == 'discrete':
            action = int(rng.integers(0, num_actions))
            request.action = [float(action)]
        else:
            request.action = rng.uniform(-1.0, 1.0, size=1).tolist()

        self._pending_call = True
        future = self._step_client.call_async(request)
        future.add_done_callback(self._on_step_response)

    def _on_step_response(self, future):
        self._pending_call = False
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warn(f'Step failed: {response.message}')
                # If step fails (e.g., model was reloaded), re-initialize
                self._model_initialized = False
                return

            self._step_count += 1

            if response.terminated:
                self.get_logger().info(
                    f'Episode terminated after {self._step_count} steps. Re-initializing.'
                )
                self._model_initialized = False
                # Advance seed for next episode
                if self.get_parameter('dataset_advance').value:
                    self._dataset_seed_counter += 1

            if self._step_count % 100 == 0:
                self.get_logger().info(
                    f'predict() #{self._step_count} via model_server '
                    f'(reward={response.reward:.3f})'
                )

        except Exception as e:
            self.get_logger().error(f'Step service call failed: {e}')

    def _assemble_observation(self) -> Optional[np.ndarray]:
        obs_mode = self.get_parameter('obs_mode').value
        if obs_mode == 'dataset':
            return None
        elif obs_mode == 'image':
            return self._latest_image.copy() if self._latest_image is not None else None
        elif obs_mode == 'vector':
            return self._latest_vector.copy() if self._latest_vector is not None else None
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