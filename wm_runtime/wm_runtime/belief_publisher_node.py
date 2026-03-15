import time
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, JointState

from wm_interfaces.msg import BeliefState, WorldModelObservation

class BeliefPublisherNode(Node):
    """
    Assembles observations from sensor topics and publishes belief states.
    """

    def __init__(self):
        super().__init__('wm_belief_publisher')

        self.declare_parameter('obs_mode', 'dataset')      # image | vector | dataset
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('vector_topic', '/obs_vector')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('dataset_seed', 42)
        self.declare_parameter('dataset_advance', True)     # step seed each tick

        self._wm = None
        self._dataset = None
        self._lock = threading.Lock()

        self._latest_image: Optional[np.ndarray] = None
        self._latest_vector: Optional[np.ndarray] = None
        self._latest_joint_state: Optional[JointState] = None
        self._dataset_seed_counter = self.get_parameter('dataset_seed').value
        self._forward_count = 0

        self.pub_belief = self.create_publisher(BeliefState, '/wm/belief_state', 10)
        self.pub_obs = self.create_publisher(WorldModelObservation, '/wm/observation', 10)

        obs_mode = self.get_parameter('obs_mode').value
        best_effort = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
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
                'Running in dataset mode — no sensor subscription. '
                'Will sample from the model\'s linked dataset.'
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
            f'BeliefPublisherNode ready (mode={obs_mode}, rate={rate}Hz).'
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
            # Fallback: try to reshape as-is
            img = raw.reshape(h, w, -1) if len(raw) > h * w else raw.reshape(h, w)

        self._latest_image = img

    def _on_vector(self, msg: Float32MultiArray):
        """Store latest observation vector."""
        self._latest_vector = np.array(msg.data, dtype=np.float32)

    def _on_joint_state(self, msg: JointState):
        """Store latest joint state."""
        self._latest_joint_state = msg

    def _tick(self):
        """Called at publish_rate_hz. Assembles obs, runs forward(), publishes belief"""
        # Lazy-connect to the world model on the model server
        if self._wm is None:
            self._connect_to_model()
            if self._wm is None:
                return
        
        obs = self._assemble_observation()
        if obs is None:
            return
        
        with self._lock:
            try:
                t0 = time.perf_counter()

                seed = self._dataset_seed_counter
                self._wm.reset(seed=seed)
                state = self._wm.forward(obs)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._forward_count += 1

                belief_msg = BeliefState()
                belief_msg.header.stamp = self.get_clock().now().to_msg()
                latent = state['latent_state']
                belief_msg.latent = latent.flatten().astype(np.float32).tolist()
                belief_msg.latent_shape = list(latent.shape)
                belief_msg.seed = seed
                belief_msg.model_id = self._get_model_id()
                self.pub_belief.publish(belief_msg)

                if self._forward_count % 100 == 0:
                    self.get_logger().info(
                        f'forward() #{self._forward_count} — {elapsed_ms:.1f}ms'
                    )

            except Exception as e:
                self.get_logger().error(f'forward() failed: {e}')

        # Advance dataset seed for next tick
        if (self.get_parameter('obs_mode').value == 'dataset'
                and self.get_parameter('dataset_advance').value):
            self._dataset_seed_counter += 1
    
    def _assemble_observation(self) -> Optional[np.ndarray]:
        """
        Build the numpy observation array from the latest sensor data
        or from the dataset, depending on obs_mode.
        """
        obs_mode = self.get_parameter('obs_mode').value

        if obs_mode == 'dataset':
            if self._dataset is None:
                return None
            return self._dataset.sample(seed=self._dataset_seed_counter)

        elif obs_mode == 'image':
            if self._latest_image is None:
                return None
            return self._latest_image.copy()

        elif obs_mode == 'vector':
            if self._latest_vector is None:
                return None
            return self._latest_vector.copy()

        return None

    def _connect_to_model(self):
        """
        Get a reference to the world model loaded by model_server_node.

        Strategy: import worldmodel_hub and try to access the model
        through a shared-memory approach. For now, we use a simple
        ROS service check — if the model server is running and has a
        model loaded, we load our own copy.

        NOTE: In a production system, the model would live in shared
        memory or a separate process with IPC. For the MVP, each node
        that needs the model loads its own copy via the same params.
        The model_server handles services; this node handles the
        continuous belief publishing loop.
        """
        try:
            from worldmodel_hub import AutoWorldModel

            # Check if model_server has advertised /wm/status
            # If so, we know the repo_id and can load our own copy
            topic_names = [name for name, _ in self.get_topic_names_and_types()]

            if '/wm/status' not in topic_names:
                return  # Model server not up yet

            # For MVP: read the same params as model_server
            # In practice these would come from a shared config
            repo_id = self.get_parameter('repo_id').value if self.has_parameter('repo_id') else ''
            if not repo_id:
                # Try to discover from model_server's status
                # For now, declare these params so the launch file can set them
                self.declare_parameter('repo_id', '')
                self.declare_parameter('subfolder', '')
                self.declare_parameter('device', 'cpu')
                self.declare_parameter('trust_remote_code', False)
                repo_id = self.get_parameter('repo_id').value

            if not repo_id:
                return

            subfolder = self.get_parameter('subfolder').value or None
            device = self.get_parameter('device').value

            self.get_logger().info(f'Loading model: {repo_id} / {subfolder}')
            self._wm = AutoWorldModel.from_pretrained(
                repo_id=repo_id,
                subfolder=subfolder,
                device=device,
                trust_remote_code=self.get_parameter('trust_remote_code').value,
            )

            # Load dataset if available
            try:
                self._dataset = self._wm.load_dataset()
                self.get_logger().info('Dataset loaded for belief publisher.')
            except Exception:
                self._dataset = None

            self.get_logger().info('Belief publisher connected to world model.')

        except ImportError:
            self.get_logger().warn(
                'worldmodel_hub not installed. '
                'Belief publisher will retry on next tick.'
            )
        except Exception as e:
            self.get_logger().warn(f'Model connection failed: {e}')

    def _get_model_id(self) -> str:
        """Extract model_id string from loaded model."""
        if self._wm is None:
            return ''
        if hasattr(self._wm, 'world_spec') and self._wm.world_spec is not None:
            spec = self._wm.world_spec
            return f'{spec.model_type}/{spec.env}'
        return 'unknown'

def main(args=None):
    rclpy.init(args=args)
    node = BeliefPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


