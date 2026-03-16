import time
import threading
import traceback
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from wm_interfaces.msg import (
    ActionSequence,
    BeliefState,
    ModelStatus,
    RolloutTrajectory,
    RolloutSet,
)

from wm_interfaces.srv import (
    ForwardObservation,
    Imagine,
    LoadModel,
    PlanAction,
    StepAction,
    WhatIf,
)

class ModelServerNode(Node):
    """
    Central world-model server.

    * Owns the single model instance (no other node loads a copy).
    * Exposes five services: load_model, forward_observation,
      imagine, what_if, plan_action.
    * Publishes: /wm/belief_state, /wm/status, /wm/rollout_set,
      /wm/viz/forward_frames, and TF world → wm_latent_space.
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
            callback_group=self._cb_group,
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
        self.srv_forward = self.create_service(
            ForwardObservation, '/wm/forward_observation',
            self._handle_forward_observation,
            callback_group=self._cb_group,
        )
        self.srv_plan = self.create_service(
            PlanAction, '/wm/plan_action',
            self._handle_plan_action,
            callback_group=self._cb_group,
        )
        self.srv_step = self.create_service(
            StepAction, '/wm/step_action',
            self._handle_step_action,
            callback_group=self._cb_group,
        )

        # ── Publishers ────────────────────────────────────────
        self.pub_belief = self.create_publisher(
            BeliefState, '/wm/belief_state', 10,
        )
        self.pub_status = self.create_publisher(
            ModelStatus, '/wm/status', 10,
        )
        self.pub_rollout_set = self.create_publisher(
            RolloutSet, '/wm/rollout_set', 10,
        )
        # Continuous forward-pass reconstructions (separate from
        # the rollout_visualizer's /wm/viz/imagined_frames topic)
        self.pub_forward_frames = self.create_publisher(
            Image, '/wm/viz/forward_frames', 10,
        )

        # ── TF broadcaster for the latent-space frame ─────────
        self._tf_broadcaster = TransformBroadcaster(self)

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
            belief_latent = np.array(request.belief.latent, dtype=np.float32)
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
                        record_observations=request.record_observations,
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

            self._publish_rollout_set(
                trajectories=trajectories,
                best_index=best_idx,
                belief=request.belief,
            )

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

            base_reward = trajectories[0].total_reward if trajectories else 0.0
            reward_deltas = [t.total_reward - base_reward for t in trajectories]
            best_idx = int(np.argmax([t.total_reward for t in trajectories]))

            response.trajectories = trajectories
            response.reward_deltas = [float(d) for d in reward_deltas]
            response.best_index = best_idx
            response.success = True
            response.message = f'Compared {len(trajectories)} alternatives.'

            self._publish_rollout_set(
                trajectories=trajectories,
                best_index=best_idx,
                belief=request.belief,
            )

        except Exception as e:
            response.success = False
            response.message = f'WhatIf failed: {e}'
            self.get_logger().error(f'WhatIf failed:\n{traceback.format_exc()}')
        return response

    def _handle_forward_observation(self, request, response):
        """
        ForwardObservation.srv handler — encodes an observation via
        wm.reset() + wm.forward(obs) and returns the belief state.

        If obs_data is empty, samples from the model's linked dataset.
        """
        if self._wm is None:
            response.success = False
            response.message = 'No model loaded. Call /wm/load_model first.'
            return response

        try:
            seed = request.seed

            if len(request.obs_data) == 0:
                if self._dataset is None:
                    response.success = False
                    response.message = (
                        'Empty observation and no dataset loaded. '
                        'Either send observation data or load a model '
                        'with an associated dataset.'
                    )
                    return response
                obs = self._dataset.sample(seed=seed)
            else:
                obs_data = np.array(request.obs_data, dtype=np.float32)
                obs_shape = tuple(request.obs_shape)
                obs = obs_data.reshape(obs_shape) if obs_shape else obs_data

            with self._model_lock:
                state = self._forward_observation(obs, seed=seed)

            belief_msg = self._state_to_belief_msg(state, seed)
            response.belief = belief_msg
            response.success = True
            response.message = 'Forward pass complete.'

        except Exception as e:
            response.success = False
            response.message = f'Forward failed: {e}'
            self.get_logger().error(
                f'ForwardObservation failed:\n{traceback.format_exc()}'
            )
        return response

    def _handle_plan_action(self, request, response):
        """
        PlanAction.srv handler — the "easy mode" planning API.

        Given a raw observation: forward → sample candidates → rollout
        all → (optionally CEM-refine) → return best first action.
        """
        if self._wm is None:
            response.success = False
            response.message = 'No model loaded. Call /wm/load_model first.'
            return response

        try:
            seed = request.seed
            horizon = request.horizon or 10
            num_candidates = request.num_candidates or 16
            method = request.sampling_method or 'random'

            # 1. Extract observation
            obs = self._extract_obs_from_request(request.observation, seed)
            if obs is None:
                response.success = False
                response.message = (
                    'Could not extract observation. Send obs_vector in '
                    'the WorldModelObservation, or ensure a dataset is loaded.'
                )
                return response

            # 2. Forward pass
            with self._model_lock:
                state = self._forward_observation(obs, seed=seed)

            belief_msg = self._state_to_belief_msg(state, seed)
            latent = state['latent_state'].flatten().astype(np.float32)

            # 3. Sample candidates
            candidates = self._sample_candidates(
                num_candidates=num_candidates,
                horizon=horizon,
                method=method,
                seed=seed,
            )

            # 4. Roll out each candidate (with frames for visualizer)
            trajectories = []
            best_reward = float('-inf')
            best_idx = 0

            with self._model_lock:
                for i, candidate in enumerate(candidates):
                    self._forward_observation(obs, seed=seed)
                    traj = self._rollout_candidate(
                        belief_latent=latent,
                        candidate=candidate,
                        horizon=horizon,
                        seed=seed,
                        record_observations=True,
                    )
                    trajectories.append(traj)
                    if traj.total_reward > best_reward:
                        best_reward = traj.total_reward
                        best_idx = i

            self._total_rollouts += len(trajectories)

            # 5. CEM refinement (if requested)
            if method == 'cem' and len(trajectories) > 0:
                trajectories, candidates, best_idx = self._cem_refine(
                    obs=obs,
                    seed=seed,
                    horizon=horizon,
                    initial_trajectories=trajectories,
                    initial_candidates=candidates,
                    num_iterations=3,
                    elite_fraction=0.2,
                )
                best_reward = trajectories[best_idx].total_reward

            # 6. Extract best first action from the REFINED candidates
            best_candidate = candidates[best_idx]
            action_dim = best_candidate.action_dim or 1
            raw_actions = np.array(best_candidate.actions, dtype=np.float32)
            if action_dim > 1:
                best_first_action = raw_actions[:action_dim].tolist()
            else:
                best_first_action = [float(raw_actions[0])]

            # 7. Build response
            response.best_action = best_first_action
            response.best_rollout = trajectories[best_idx]

            rollout_set = RolloutSet()
            rollout_set.header.stamp = self.get_clock().now().to_msg()
            rollout_set.trajectories = trajectories
            rollout_set.best_index = best_idx
            rollout_set.belief = belief_msg
            response.all_rollouts = rollout_set

            response.success = True
            response.message = (
                f'Planned with {len(trajectories)} candidates '
                f'({method}), best reward={best_reward:.3f}'
            )

            self._publish_rollout_set(
                trajectories=trajectories,
                best_index=best_idx,
                belief=belief_msg,
            )

        except Exception as e:
            response.success = False
            response.message = f'Planning failed: {e}'
            self.get_logger().error(
                f'PlanAction failed:\n{traceback.format_exc()}'
            )
        return response

    def _handle_step_action(self, request, response):
        """
        StepAction.srv handler — wraps wm.predict(action).

        Steps the world model forward by one action. If the model state
        hasn't been initialized yet (no prior forward() call), this will
        auto-initialize from the dataset if available.
        """
        if self._wm is None:
            response.success = False
            response.message = 'No model loaded. Call /wm/load_model first.'
            return response

        try:
            # Auto-initialize if no belief state exists
            if self._current_belief is None:
                if self._dataset is not None:
                    self.get_logger().info(
                        'No model state — auto-initializing from dataset.'
                    )
                    with self._model_lock:
                        obs = self._dataset.sample(seed=42)
                        self._forward_observation(obs, seed=42)
                else:
                    response.success = False
                    response.message = (
                        'No model state and no dataset. '
                        'Call /wm/forward_observation first.'
                    )
                    return response
            is_discrete = (request.action_space == 'discrete')
            raw_action = np.array(request.action, dtype=np.float32)

            if is_discrete:
                action = int(raw_action[0])
            elif len(raw_action) == 1:
                action = float(raw_action[0])
            else:
                action = raw_action

            with self._model_lock:
                t0 = time.perf_counter()
                output = self._wm.predict(action)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._total_predictions += 1

                if self._total_predictions > 0:
                    self._avg_predict_ms = (
                        0.9 * self._avg_predict_ms + 0.1 * elapsed_ms
                    )

                # Update internal belief state
                self._current_belief = output

            # Build belief message
            belief_msg = self._state_to_belief_msg(output, seed=0)
            self.pub_belief.publish(belief_msg)

            response.belief = belief_msg
            response.reward = float(output.get('reward', 0.0) or 0.0)
            response.terminated = bool(output.get('terminated', False))

            # Pack reconstructed frame if available
            recon = output.get('recon')
            if recon is not None:
                frame = np.asarray(recon)
                # Squeeze batch dims
                while frame.ndim > 3:
                    frame = frame.squeeze(0)
                # CHW → HWC
                if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] < frame.shape[1]:
                    frame = np.transpose(frame, (1, 2, 0))
                # Float → uint8
                if frame.dtype != np.uint8:
                    fdata = np.asarray(frame, dtype=np.float64)
                    if fdata.max() <= 1.0 and fdata.min() >= -0.01:
                        fdata = fdata.clip(0.0, 1.0) * 255.0
                    frame = fdata.clip(0, 255).astype(np.uint8)
                # Squeeze single-channel (H,W,1) → (H,W)
                if frame.ndim == 3 and frame.shape[2] == 1:
                    frame = frame[:, :, 0]
                frame = np.ascontiguousarray(frame)

                response.recon_height = frame.shape[0]
                response.recon_width = frame.shape[1]
                response.recon_channels = frame.shape[2] if frame.ndim == 3 else 1
                response.recon_data = frame.tobytes()

                # Also publish to the forward_frames topic
                self._publish_frame(recon)
            else:
                response.recon_height = 0
                response.recon_width = 0
                response.recon_channels = 0

            response.success = True
            response.message = (
                f'Step complete: reward={response.reward:.3f}, '
                f'terminated={response.terminated}'
            )

        except Exception as e:
            response.success = False
            response.message = f'Step failed: {e}'
            self.get_logger().error(
                f'StepAction failed:\n{traceback.format_exc()}'
            )
        return response

    def _extract_obs_from_request(self, obs_msg, seed: int) -> Optional[np.ndarray]:
        """Extract numpy observation from WorldModelObservation. Priority: vector > image > dataset."""
        if len(obs_msg.obs_vector) > 0:
            obs = np.array(obs_msg.obs_vector, dtype=np.float32)
            if len(obs_msg.obs_shape) > 0:
                obs = obs.reshape(tuple(obs_msg.obs_shape))
            return obs

        if obs_msg.image.height > 0 and obs_msg.image.width > 0:
            h, w = obs_msg.image.height, obs_msg.image.width
            raw = np.frombuffer(obs_msg.image.data, dtype=np.uint8)
            enc = obs_msg.image.encoding
            if enc in ('rgb8', 'bgr8'):
                img = raw.reshape(h, w, 3)
                if enc == 'bgr8':
                    img = img[:, :, ::-1]
            elif enc == 'mono8':
                img = raw.reshape(h, w)
            else:
                img = raw.reshape(h, w, -1) if len(raw) > h * w else raw.reshape(h, w)
            return img.astype(np.float32)

        if self._dataset is not None:
            return self._dataset.sample(seed=seed)
        return None

    def _sample_candidates(self, num_candidates, horizon, method, seed) -> list:
        """Generate candidate ActionSequence messages."""
        rng = np.random.default_rng(seed)
        is_discrete, num_actions, action_dim = False, 4, 1
        action_low, action_high = -1.0, 1.0

        if self._dataset is not None:
            try:
                act_space = self._dataset.action_space
                if hasattr(act_space, 'n'):
                    is_discrete, num_actions, action_dim = True, act_space.n, 1
                else:
                    is_discrete = False
                    action_dim = act_space.shape[0] if act_space.shape else 1
                    action_low = float(act_space.low.flat[0])
                    action_high = float(act_space.high.flat[0])
            except Exception:
                pass

        candidates = []
        for i in range(num_candidates):
            msg = ActionSequence()
            msg.horizon = horizon
            msg.action_dim = action_dim
            msg.action_space = 'discrete' if is_discrete else 'continuous'
            msg.label = f'{method}_{i}'

            if is_discrete:
                actions = rng.integers(0, num_actions, size=horizon)
                msg.actions = actions.astype(np.float32).tolist()
            elif method == 'mppi':
                noise_std = (action_high - action_low) * 0.3
                actions = rng.normal(0.0, noise_std, size=(horizon, action_dim))
                actions = np.clip(actions, action_low, action_high)
                msg.actions = actions.flatten().astype(np.float32).tolist()
            else:
                actions = rng.uniform(action_low, action_high, size=(horizon, action_dim))
                msg.actions = actions.flatten().astype(np.float32).tolist()
            candidates.append(msg)
        return candidates

    def _cem_refine(self, obs, seed, horizon, initial_trajectories,
                    initial_candidates, num_iterations=3, elite_fraction=0.2):
        """
        Cross-entropy method: fit Gaussian to elite actions, resample, re-rollout.
        Returns (trajectories, candidates, best_idx) — note: candidates are returned.
        """
        rng = np.random.default_rng(seed + 1000)
        trajectories = list(initial_trajectories)
        candidates = list(initial_candidates)
        num_elite = max(2, int(len(candidates) * elite_fraction))
        num_cand = len(candidates)
        action_dim = candidates[0].action_dim or 1
        is_discrete = (candidates[0].action_space == 'discrete')

        if is_discrete:
            rewards = [t.total_reward for t in trajectories]
            return trajectories, candidates, int(np.argmax(rewards))

        for cem_iter in range(num_iterations):
            rewards = np.array([t.total_reward for t in trajectories])
            elite_indices = np.argsort(rewards)[-num_elite:]

            elite_actions = []
            for idx in elite_indices:
                a = np.array(candidates[idx].actions, dtype=np.float32)
                a = a.reshape(horizon, action_dim) if action_dim > 1 else a[:horizon].reshape(horizon, 1)
                elite_actions.append(a)

            elite_stack = np.stack(elite_actions, axis=0)
            mean = elite_stack.mean(axis=0)
            std = elite_stack.std(axis=0) + 1e-6

            new_candidates = []
            for i in range(num_cand):
                if i < num_elite:
                    actions = elite_actions[i]
                else:
                    actions = mean + std * rng.normal(0.0, 1.0, size=(horizon, action_dim))
                msg = ActionSequence()
                msg.horizon = horizon
                msg.action_dim = action_dim
                msg.action_space = 'continuous'
                msg.label = f'cem_{cem_iter}_{i}'
                msg.actions = actions.flatten().astype(np.float32).tolist()
                new_candidates.append(msg)

            new_trajectories = []
            with self._model_lock:
                for candidate in new_candidates:
                    self._forward_observation(obs, seed=seed)
                    traj = self._rollout_candidate(
                        belief_latent=np.zeros(1),
                        candidate=candidate,
                        horizon=horizon,
                        seed=seed,
                        record_observations=True,
                    )
                    new_trajectories.append(traj)

            self._total_rollouts += len(new_trajectories)
            trajectories = new_trajectories
            candidates = new_candidates

        rewards = [t.total_reward for t in trajectories]
        best_idx = int(np.argmax(rewards))
        return trajectories, candidates, best_idx

    def _load_model(self, repo_id, subfolder, device, trust_remote_code):
        """Load a world model via AutoWorldModel.from_pretrained(). Thread-safe."""
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

        try:
            self._dataset = new_wm.load_dataset()
            self.get_logger().info('Associated dataset loaded.')
        except Exception:
            self._dataset = None
            self.get_logger().info('No associated dataset (this is fine).')

    def _forward_observation(self, obs: np.ndarray, seed: int = 42):
        """Run wm.reset() + wm.forward(obs), publish belief + input frame. Lock must be held."""
        t0 = time.perf_counter()
        self._wm.reset(seed=seed)
        state = self._wm.forward(obs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._avg_forward_ms = 0.9 * self._avg_forward_ms + 0.1 * elapsed_ms
        self._current_belief = state

        msg = self._state_to_belief_msg(state, seed)
        self.pub_belief.publish(msg)

        # Publish the frame: prefer the model's reconstruction if available,
        # otherwise publish the raw input observation (IRIS's forward() does
        # not produce a 'recon' key — only predict() does).
        recon = state.get('recon')
        if recon is not None:
            self.get_logger().debug(
                f'forward() returned recon: type={type(recon).__name__}, '
                f'shape={np.asarray(recon).shape}, dtype={np.asarray(recon).dtype}'
            )
            self._publish_frame(recon)
        else:
            self.get_logger().debug(
                f'forward() has no recon, publishing input obs: '
                f'shape={obs.shape}, dtype={obs.dtype}'
            )
            self._publish_frame(obs)

        return state

    def _rollout_candidate(self, belief_latent, candidate, horizon, seed,
                           record_observations=False) -> RolloutTrajectory:
        """Roll out a single candidate. Lock must be held."""
        action_dim = candidate.action_dim or 1
        h = horizon or candidate.horizon
        is_discrete = (candidate.action_space == 'discrete')

        raw_actions = np.array(candidate.actions, dtype=np.float32)
        actions = raw_actions.reshape(h, action_dim) if action_dim > 1 else raw_actions[:h]

        if self._current_belief is None and self._dataset is not None:
            obs = self._dataset.sample(seed=seed)
            self._forward_observation(obs, seed=seed)
        elif self._current_belief is None:
            self.get_logger().warn('No prior belief and no dataset. Using reset only.')
            self._wm.reset(seed=seed)

        t0 = time.perf_counter()
        predicted_latents, rewards, terminated, frames = [], [], [], []

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
                frame = np.asarray(output['recon'], dtype=np.float64)
                # Squeeze batch dim if present
                while frame.ndim > 3:
                    frame = frame.squeeze(0)
                # CHW → HWC
                if frame.ndim == 3 and frame.shape[0] in (1, 3):
                    frame = np.transpose(frame, (1, 2, 0))
                # Squeeze single channel: (H,W,1) → (H,W)
                if frame.ndim == 3 and frame.shape[2] == 1:
                    frame = frame.squeeze(2)
                # float → uint8
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255.0).clip(0, 255)
                    frame = frame.astype(np.uint8)
                frames.append(frame)

            if terminated[-1]:
                break

        wall_time = time.perf_counter() - t0
        actual_steps = len(rewards)

        if actual_steps > 0:
            per_step = (wall_time * 1000.0) / actual_steps
            self._avg_predict_ms = 0.9 * self._avg_predict_ms + 0.1 * per_step

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

        ep_len = actual_steps
        for i, t in enumerate(terminated):
            if t:
                ep_len = i + 1
                break
        traj.episode_length = ep_len
        traj.wall_time = wall_time
        traj.steps_per_second = float(actual_steps / wall_time) if wall_time > 0 else 0.0
        traj.actions = candidate

        if frames:
            traj.frame_height = frames[0].shape[0]
            traj.frame_width = frames[0].shape[1]
            traj.frame_channels = frames[0].shape[2] if frames[0].ndim == 3 else 1
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

    def _state_to_belief_msg(self, state: dict, seed: int) -> BeliefState:
        """Convert a forward() output dict to a BeliefState message."""
        msg = BeliefState()
        msg.header.stamp = self.get_clock().now().to_msg()
        latent = state['latent_state']
        msg.latent = latent.flatten().astype(np.float32).tolist()
        msg.latent_shape = list(latent.shape)
        msg.seed = seed
        msg.model_id = self._model_id
        return msg

    def _publish_rollout_set(self, trajectories, best_index, belief):
        """Publish RolloutSet to /wm/rollout_set for the visualizer."""
        msg = RolloutSet()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.trajectories = trajectories
        msg.best_index = best_index
        msg.belief = belief
        self.pub_rollout_set.publish(msg)
        self.get_logger().debug(
            f'Published RolloutSet: {len(trajectories)} trajectories, '
            f'best_index={best_index}'
        )

    def _publish_status(self):
        """Periodic status heartbeat + TF frame broadcast."""
        now = self.get_clock().now().to_msg()

        msg = ModelStatus()
        msg.header.stamp = now
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

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = 'world'
        tf.child_frame_id = 'wm_latent_space'
        tf.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf)

    def _publish_frame(self, frame_data):
        """
        Publish an observation or reconstruction image to /wm/viz/forward_frames.

        Matches the PygameRenderer._prepare_frame() logic from worldmodel_hub
        exactly, then converts to a ROS Image message with correct encoding
        and validated byte count.
        """
        if frame_data is None:
            return

        frame = np.asarray(frame_data)

        self.get_logger().debug(
            f'[frame_debug] raw input: shape={frame.shape}, dtype={frame.dtype}'
        )

        # Squeeze any leftover batch dimensions: (1, C, H, W) → (C, H, W)
        while frame.ndim > 3:
            frame = frame.squeeze(0)

        if frame.ndim < 2:
            self.get_logger().warn(
                f'_publish_frame: unexpected shape {frame.shape}, skipping'
            )
            return

        # 2D array = grayscale HW
        if frame.ndim == 2:
            pass  # already HW, handle below

        # 3D: detect CHW vs HWC
        elif frame.ndim == 3:
            # CHW → HWC (C is 1 or 3, and is the smallest dim for typical images)
            if frame.shape[0] in (1, 3) and frame.shape[0] < frame.shape[1]:
                frame = np.transpose(frame, (1, 2, 0))

        # Float → uint8
        if frame.dtype != np.uint8:
            frame = np.asarray(frame, dtype=np.float64)
            if frame.max() <= 1.0 and frame.min() >= -0.01:
                frame = (frame.clip(0.0, 1.0) * 255.0)
            frame = frame.clip(0, 255).astype(np.uint8)

        frame = np.ascontiguousarray(frame)

        self.get_logger().debug(
            f'[frame_debug] after processing: shape={frame.shape}, dtype={frame.dtype}'
        )

        # Determine encoding
        if frame.ndim == 2:
            h, w = frame.shape
            encoding = 'mono8'
            step = w
        elif frame.ndim == 3 and frame.shape[2] == 3:
            h, w = frame.shape[0], frame.shape[1]
            encoding = 'rgb8'
            step = w * 3
        elif frame.ndim == 3 and frame.shape[2] == 1:
            h, w = frame.shape[0], frame.shape[1]
            frame = frame[:, :, 0]  # squeeze channel dim for mono8
            encoding = 'mono8'
            step = w
        elif frame.ndim == 3 and frame.shape[2] == 4:
            h, w = frame.shape[0], frame.shape[1]
            frame = frame[:, :, :3]  # drop alpha
            encoding = 'rgb8'
            step = w * 3
            frame = np.ascontiguousarray(frame)
        else:
            self.get_logger().warn(
                f'_publish_frame: unhandled shape {frame.shape}, skipping'
            )
            return

        raw = frame.tobytes()
        expected_size = h * step
        if len(raw) != expected_size:
            self.get_logger().error(
                f'Frame size mismatch: {h}x{w} {encoding} expects '
                f'{expected_size} bytes but got {len(raw)}. '
                f'Frame shape={frame.shape}, dtype={frame.dtype}'
            )
            return

        self.get_logger().debug(
            f'[frame_debug] publishing: {w}x{h} {encoding} step={step} '
            f'data={len(raw)} bytes (expected={expected_size})'
        )

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'wm_imagined'
        msg.height = h
        msg.width = w
        msg.encoding = encoding
        msg.step = step
        msg.is_bigendian = False
        msg.data = raw
        self.pub_forward_frames.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ModelServerNode()
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