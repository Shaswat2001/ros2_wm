import time
import threading
import traceback
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from wm_interfaces.msg import (
    ActionSequence, 
    BeliefState,
    ModelStatus,
    RolloutTrajectory,
    RolloutSet
)

from wm_interfaces.srv import (
    Imagine, 
    LoadModel,
    WhatIf
)

class ModelServerNode(Node):
    """
    World model server - loads models and serves imagination requests.
    """

    def __init__(self):
        super().__init__("wm_model_server")

        self.declare_parameter('repo_id', '')
        self.declare_parameter('subfolder', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('trust_remote_code', False)
        self.declare_parameter('status_rate_hz', 1.0)

        self._wm = None
        self._dataset = None
        self._model_lock = threading.Lock()
        self._current_belief: Optional[dict] = None
        self._model_id = ''
        self._model_type = ''
        self._environment = ''

        # Runtime stats
        self._total_predictions = 0
        self._total_rollouts = 0
        self._avg_predict_ms = 0.0
        self._avg_forward_ms = 0.0

        # ── Callback group (allow concurrent service calls) ───
        self._cb_group = ReentrantCallbackGroup()

        # ── Services ──────────────────────────────────────────
        self.srv_load = self.create_service(
            LoadModel, '/wm/load_model',
            self._handle_load_model, 
            callback_group=self._cb_group
        )
        self.srv_imagine = self.create_service(
            Imagine, '/wm/imagine',
            self._handle_imagine,
            callback_group=self._cb_group,
        )
        self.srv_what_if = self.create_service(
            WhatIf, '/wm/what_if',
            self._handle_what_if,
            callback_group=self._cb_group,
        )

        # ── Publishers ────────────────────────────────────────
        self.pub_belief = self.create_publisher(
            BeliefState, '/wm/belief_state', 10,
        )
        self.pub_status = self.create_publisher(
            ModelStatus, '/wm/status', 10,
        )

        # ── Status timer ──────────────────────────────────────
        status_hz = self.get_parameter('status_rate_hz').value
        if status_hz > 0:
            self.create_timer(1.0 / status_hz, self._publish_status)

        # ── Auto-load if parameters are set ───────────────────
        repo_id = self.get_parameter('repo_id').value
        if repo_id:
            self.get_logger().info(f'Auto-loading model from parameter: {repo_id}')
            self._load_model(
                repo_id=repo_id,
                subfolder=self.get_parameter('subfolder').value,
                device=self.get_parameter('device').value,
                trust_remote_code=self.get_parameter('trust_remote_code').value,
            )

        self.get_logger().info('ModelServerNode ready.')

    # Service handlers

    def _handle_load_model(self, request, response):
        """LoadModel.srv handler — wraps AutoWorldModel.from_pretrained()."""
        self.get_logger().info(
            f'LoadModel request: {request.repo_id} / {request.subfolder}'
        )
        try:
            self._load_model(
                repo_id=request.repo_id,
                subfolder=request.subfolder or None,
                device=request.device or 'cpu',
                trust_remote_code=request.trust_remote_code,
            )
            response.success = True
            response.message = 'Model loaded successfully.'
            response.model_type = self._model_type
            response.environment = self._environment
        except Exception as e:
            response.success = False
            response.message = f'Failed to load model: {e}'
            self.get_logger().error(f'Load failed:\n{traceback.format_exc()}')

        return response
    
    def _handle_imagine(self, request, response):
        """Imagine.srv handler — rolls out candidate action sequences."""
        if self._wm is None:
            response.success = False
            response.message = 'No model loaded. Call /wm/load_model first.'
            return response
        
        try:
            belief_latent = np.array(request.belief_latent, dtype=np.float32)
            horizon = request.horizon

            trajectories = []
            best_reward = float('-inf')
            best_idx = 0

            with self._model_lock:
                for i, candidate in enumerate(request.candidates):
                    traj = self._rollout_candidate(
                        belief_latent=belief_latent,
                        candidate=candidate,
                        horizon=horizon,
                        seed=request.belief.seed,
                        record_observations=request.record_observations
                    )
                    trajectories.append(traj)

                    if traj.total_reward > best_reward:
                        best_reward = traj.total_reward
                        best_idx = i
            
            response.trajectories = trajectories
            response.best_index = best_idx
            response.success = True
            response.message = f'Imagined {len(trajectories)} candidates.'
            self._total_rollouts += len(trajectories)
        
        except Exception as e:
            response.success = False
            response.message = f'Imagination failed: {e}'
            self.get_logger().error(f'Imagine failed:\n{traceback.format_exc()}')

        return response

    def _handle_what_if(self, request, response):
        """WhatIf.srv handler — counterfactual comparison."""
        if self._wm is None:
            response.success = False
            response.message = 'No model loaded. Call /wm/load_model first.'
            return response

        try:
            belief_latent = np.array(request.belief.latent, dtype=np.float32)
            horizon = request.horizon

            trajectories = []
            with self._model_lock:
                for alt in request.alternatives:
                    traj = self._rollout_candidate(
                        belief_latent=belief_latent,
                        candidate=alt,
                        horizon=horizon,
                        seed=request.belief.seed,
                        record_observations=request.record_observations,
                    )
                    trajectories.append(traj)

            # Compute reward deltas relative to first alternative
            base_reward = trajectories[0].total_reward if trajectories else 0.0
            reward_deltas = [
                t.total_reward - base_reward for t in trajectories
            ]

            best_idx = int(np.argmax([t.total_reward for t in trajectories]))

            response.trajectories = trajectories
            response.reward_deltas = [float(d) for d in reward_deltas]
            response.best_index = best_idx
            response.success = True
            response.message = f'Compared {len(trajectories)} alternatives.'

        except Exception as e:
            response.success = False
            response.message = f'WhatIf failed: {e}'
            self.get_logger().error(f'WhatIf failed:\n{traceback.format_exc()}')

        return response

    # Core worldmodel_hub bridge methods

    def _load_model(self, repo_id, subfolder, device, trust_remote_code):
        """
        Load a world model via AutoWorldModel.from_pretrained().
        Thread-safe: acquires _model_lock before swapping.
        """

        # Import here so the node can start even if worldmodel_hub
        # isn't installed yet (e.g. during interface-only testing).
        from worldmodel_hub import AutoWorldModel

        self.get_logger().info(f'Loading: {repo_id} / {subfolder} on {device}')
        t0 = time.perf_counter()

        new_wm = AutoWorldModel.from_pretrained(
            repo_id=repo_id,
            subfolder=subfolder or None,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        elapsed = time.perf_counter() - t0

        with self._model_lock:
            self._wm = new_wm
            self._current_belief = None
            self._total_predictions = 0
            self._total_rollouts = 0

        # Extract metadata
        if hasattr(new_wm, 'world_spec') and new_wm.world_spec is not None:
            spec = new_wm.world_spec
            self._model_type = spec.model_type
            self._environment = spec.env
            self._model_id = f'{spec.model_type}/{spec.env}'
        else:
            self._model_type = 'unknown'
            self._environment = 'unknown'
            self._model_id = repo_id

        self.get_logger().info(
            f'Model loaded in {elapsed:.2f}s — '
            f'type={self._model_type}, env={self._environment}'
        )

        # Try to load the associated dataset
        try:
            self._dataset = new_wm.load_dataset()
            self.get_logger().info('Associated dataset loaded.')
        except Exception:
            self._dataset = None
            self.get_logger().info('No associated dataset (this is fine).')

    def _forward_observation(self, obs: np.ndarray, seed: int = 42):
        """
        Run wm.reset() + wm.forward(obs) and publish the belief state.
        Must be called _model_lock held.
        """
        t0 = time.perf_counter()

        self._wm.reset(seed=seed)
        state = self._wm.forward(obs)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._avg_forward_ms = (
            0.9 * self._avg_forward_ms + 0.1 * elapsed_ms
        )

        self._current_belief = state

        # Publish belief
        msg = BeliefState()
        msg.header.stamp = self.get_clock().now().to_msg()
        latent = state['latent_state']
        msg.latent = latent.flatten().astype(np.float32).tolist()
        msg.latent_shape = list(latent.shape)
        msg.seed = seed
        msg.model_id = self._model_id
        self.pub_belief.publish(msg)

        return state

    def _rollout_candidate(
        self,
        belief_latent: np.ndarray,
        candidate: ActionSequence,
        horizon: int,
        seed: int,
        record_observations: bool = False,
    ) -> RolloutTrajectory:
        """
        Roll out a single candidate action sequence through the world model.
        Returns a RolloutTrajectory ROS message.

        Must be called with _model_lock held.
        """
        # Parse actions from the flat message
        action_dim = candidate.action_dim or 1
        h = horizon or candidate.horizon
        is_discrete = (candidate.action_space == 'discrete')

        raw_actions = np.array(candidate.actions, dtype=np.float32)
        if action_dim > 1:
            actions = raw_actions.reshape(h, action_dim)
        else:
            actions = raw_actions[:h]

        # We need an observation to call forward(). If we have a dataset,
        # sample one. Otherwise, the caller should have provided belief
        # from a prior forward() call — for now we re-forward with a
        # dummy if needed.
        if self._current_belief is None and self._dataset is not None:
            obs = self._dataset.sample(seed=seed)
            self._forward_observation(obs, seed=seed)
        elif self._current_belief is None:
            # No dataset, no prior belief — reset with zero obs
            self.get_logger().warn(
                'No prior belief and no dataset. Using zero observation.'
            )
            # We can't know the obs shape, so this is a fallback.
            # In practice the user should call forward() first.
            self._wm.reset(seed=seed)

        # Run the prediction loop
        t0 = time.perf_counter()
        predicted_latents = []
        rewards = []
        terminated = []
        frames = []

        for step in range(h):
            if is_discrete:
                action = int(actions[step])
            elif action_dim > 1:
                action = actions[step]
            else:
                action = float(actions[step])

            output = self._wm.predict(action)
            self._total_predictions += 1

            latent = output['latent_state'].flatten().astype(np.float32)
            predicted_latents.append(latent)
            rewards.append(float(output.get('reward', 0.0) or 0.0))
            terminated.append(bool(output.get('terminated', False)))

            if record_observations and output.get('recon') is not None:
                frame = np.asarray(output['recon'], dtype=np.uint8)
                frames.append(frame)

            # Stop early if terminated
            if terminated[-1]:
                break

        wall_time = time.perf_counter() - t0
        actual_steps = len(rewards)

        # Update predict timing
        if actual_steps > 0:
            per_step = (wall_time * 1000.0) / actual_steps
            self._avg_predict_ms = (
                0.9 * self._avg_predict_ms + 0.1 * per_step
            )

        # Build ROS message
        traj = RolloutTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.horizon = actual_steps
        traj.latent_dim = len(predicted_latents[0]) if predicted_latents else 0

        traj.predicted_latents = []
        for lat in predicted_latents:
            traj.predicted_latents.extend(lat.tolist())

        traj.rewards = rewards
        traj.terminated = terminated
        traj.total_reward = float(sum(rewards))

        # Episode length = steps to first termination
        ep_len = actual_steps
        for i, t in enumerate(terminated):
            if t:
                ep_len = i + 1
                break
        traj.episode_length = ep_len

        traj.wall_time = wall_time
        traj.steps_per_second = (
            float(actual_steps / wall_time) if wall_time > 0 else 0.0
        )

        # Copy the candidate action sequence back
        traj.actions = candidate

        # Pack frames if recorded
        if frames:
            traj.frame_height = frames[0].shape[0]
            traj.frame_width = frames[0].shape[1]
            traj.frame_channels = (
                frames[0].shape[2] if frames[0].ndim == 3 else 1
            )
            traj.predicted_frames = []
            for f in frames:
                traj.predicted_frames.extend(f.flatten().tolist())
        else:
            traj.frame_height = 0
            traj.frame_width = 0
            traj.frame_channels = 0

        traj.seed = seed
        traj.model_id = self._model_id
        traj.label = candidate.label

        return traj

    def _publish_status(self):
        """Periodic status heartbeat."""
        msg = ModelStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ready = (self._wm is not None)
        msg.repo_id = self.get_parameter('repo_id').value or ''
        msg.subfolder = self.get_parameter('subfolder').value or ''
        msg.model_type = self._model_type
        msg.environment = self._environment
        msg.device = self.get_parameter('device').value or 'cpu'
        msg.avg_predict_ms = self._avg_predict_ms
        msg.avg_forward_ms = self._avg_forward_ms
        msg.total_predictions = self._total_predictions
        msg.total_rollouts = self._total_rollouts
        self.pub_status.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ModelServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
