from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image

from wm_interfaces.msg import RolloutSet, RolloutTrajectory

# ── Color palettes ────────────────────────────────────────────

def _viridis_color(t: float, alpha: float = 0.7) -> ColorRGBA:
    """Simple viridis-ish color ramp for t in [0, 1]."""
    r = max(0.0, min(1.0, 0.267 + 0.004 * t + 2.7 * t * t - 1.9 * t * t * t))
    g = max(0.0, min(1.0, 0.004 + 1.4 * t - 0.5 * t * t))
    b = max(0.0, min(1.0, 0.329 + 1.4 * t - 1.7 * t * t + 0.2 * t * t * t))
    return ColorRGBA(r=r, g=g, b=b, a=alpha)

def _rank_color(rank: int, total: int, alpha: float = 0.5) -> ColorRGBA:
    """Color by rank: best = green, worst = red, middle = yellow."""
    if total <= 1:
        return ColorRGBA(r=0.2, g=0.8, b=0.2, a=alpha)
    t = rank / (total - 1)  # 0 = best, 1 = worst
    r = min(1.0, 2.0 * t)
    g = min(1.0, 2.0 * (1.0 - t))
    return ColorRGBA(r=r, g=g, b=0.1, a=alpha)

def _best_color(alpha: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=0.1, g=0.95, b=0.3, a=alpha)

def _uncertainty_color(alpha: float = 0.15) -> ColorRGBA:
    return ColorRGBA(r=0.3, g=0.5, b=1.0, a=alpha)

# ── Latent → 3D mapping ──────────────────────────────────────

def _latent_to_points(
    flat_latents: List[float],
    latent_dim: int,
    horizon: int,
    scale: float = 1.0,
    dims: Tuple[int, int, int] = (0, 1, 2),
) -> List[Point]:
    """
    Convert a flat latent trajectory into 3D points for RViz.

    Takes the first 3 latent dimensions (configurable via `dims`)
    and maps them to XYZ. This gives a "latent space" trajectory
    view that's meaningful even for non-spatial models.
    """
    points = []
    arr = np.array(flat_latents, dtype=np.float32)

    # Total latents = (horizon + 1) * latent_dim  (includes initial state)
    # But we might have fewer if rollout terminated early
    num_states = len(arr) // latent_dim if latent_dim > 0 else 0

    for i in range(num_states):
        offset = i * latent_dim
        state = arr[offset:offset + latent_dim]

        x = float(state[dims[0]]) * scale if dims[0] < len(state) else 0.0
        y = float(state[dims[1]]) * scale if dims[1] < len(state) else 0.0
        z = float(state[dims[2]]) * scale if dims[2] < len(state) else 0.0

        points.append(Point(x=x, y=y, z=z))

    return points

class RolloutVisualizerNode(Node):
    """Publishes RViz markers for imagined trajectories."""

    def __init__(self):
        super().__init__('wm_rollout_visualizer')

        # ── Parameters ────────────────────────────────────────
        self.declare_parameter('frame_id', 'wm_latent_space')
        self.declare_parameter('scale', 0.1)           # spatial scale for latent dims
        self.declare_parameter('line_width', 0.005)     # marker line width
        self.declare_parameter('best_line_width', 0.012)
        self.declare_parameter('sphere_size', 0.015)    # endpoint spheres
        self.declare_parameter('show_all_candidates', True)
        self.declare_parameter('show_uncertainty', True)
        self.declare_parameter('show_frames', True)
        self.declare_parameter('latent_dims', [0, 1, 2])  # which 3 dims to plot
        self.declare_parameter('max_display_candidates', 64)

        # ── Publishers ────────────────────────────────────────
        self.pub_rollouts = self.create_publisher(
            MarkerArray, '/wm/viz/rollouts', 10,
        )
        self.pub_best = self.create_publisher(
            MarkerArray, '/wm/viz/best_rollout', 10,
        )
        self.pub_uncertainty = self.create_publisher(
            MarkerArray, '/wm/viz/uncertainty', 10,
        )
        self.pub_frames = self.create_publisher(
            Image, '/wm/viz/imagined_frames', 10,
        )

        # ── Subscriber ────────────────────────────────────────
        self.create_subscription(
            RolloutSet, '/wm/rollout_set',
            self._on_rollout_set, 10,
        )

        # Track previous marker count for cleanup
        self._prev_marker_count = 0

        self.get_logger().info('RolloutVisualizerNode ready.')

    def _on_rollout_set(self, msg: RolloutSet):
        """Handle a new batch of imagined trajectories."""
        if not msg.trajectories:
            return

        frame_id = self.get_parameter('frame_id').value
        scale = self.get_parameter('scale').value
        dims_param = self.get_parameter('latent_dims').value
        dims = (dims_param[0], dims_param[1], dims_param[2])
        max_display = self.get_parameter('max_display_candidates').value
        now = self.get_clock().now().to_msg()

        header = Header(stamp=now, frame_id=frame_id)

        trajs = msg.trajectories
        best_idx = msg.best_index

        # Sort by reward to assign rank colors
        ranked_indices = sorted(
            range(len(trajs)),
            key=lambda i: trajs[i].total_reward,
            reverse=True,
        )
        rank_map = {idx: rank for rank, idx in enumerate(ranked_indices)}

        # ── All candidates markers ────────────────────────────
        if self.get_parameter('show_all_candidates').value:
            all_markers = MarkerArray()

            # First: delete old markers
            delete_marker = Marker()
            delete_marker.header = header
            delete_marker.action = Marker.DELETEALL
            all_markers.markers.append(delete_marker)

            display_count = min(len(trajs), max_display)
            for i in range(display_count):
                traj = trajs[i]
                rank = rank_map.get(i, i)

                points = _latent_to_points(
                    traj.predicted_latents, traj.latent_dim,
                    traj.horizon, scale, dims,
                )
                if len(points) < 2:
                    continue

                # Line strip for trajectory path
                line = Marker()
                line.header = header
                line.ns = 'rollout_candidates'
                line.id = i * 2
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.points = points
                line.scale = Vector3(
                    x=self.get_parameter('line_width').value,
                    y=0.0, z=0.0,
                )
                color = _rank_color(rank, len(trajs), alpha=0.4)
                line.color = color
                line.colors = [
                    _viridis_color(t / max(1, len(points) - 1), alpha=0.4)
                    for t in range(len(points))
                ]
                all_markers.markers.append(line)

                # Endpoint sphere
                endpoint = Marker()
                endpoint.header = header
                endpoint.ns = 'rollout_endpoints'
                endpoint.id = i * 2 + 1
                endpoint.type = Marker.SPHERE
                endpoint.action = Marker.ADD
                endpoint.pose.position = points[-1]
                ss = self.get_parameter('sphere_size').value
                endpoint.scale = Vector3(x=ss, y=ss, z=ss)
                endpoint.color = color
                endpoint.color.a = 0.6
                all_markers.markers.append(endpoint)

            self.pub_rollouts.publish(all_markers)

        # ── Best trajectory markers ───────────────────────────
        if best_idx < len(trajs):
            best_traj = trajs[best_idx]
            best_markers = MarkerArray()

            delete_marker = Marker()
            delete_marker.header = header
            delete_marker.action = Marker.DELETEALL
            best_markers.markers.append(delete_marker)

            best_points = _latent_to_points(
                best_traj.predicted_latents, best_traj.latent_dim,
                best_traj.horizon, scale, dims,
            )

            if len(best_points) >= 2:
                # Thick green line for best path
                best_line = Marker()
                best_line.header = header
                best_line.ns = 'best_rollout'
                best_line.id = 0
                best_line.type = Marker.LINE_STRIP
                best_line.action = Marker.ADD
                best_line.points = best_points
                best_line.scale = Vector3(
                    x=self.get_parameter('best_line_width').value,
                    y=0.0, z=0.0,
                )
                best_line.color = _best_color(alpha=1.0)
                # Gradient along the path
                best_line.colors = [
                    _best_color(alpha=max(0.4, 1.0 - 0.5 * t / max(1, len(best_points) - 1)))
                    for t in range(len(best_points))
                ]
                best_markers.markers.append(best_line)

                # Start sphere (green)
                start = Marker()
                start.header = header
                start.ns = 'best_rollout'
                start.id = 1
                start.type = Marker.SPHERE
                start.action = Marker.ADD
                start.pose.position = best_points[0]
                bs = self.get_parameter('sphere_size').value * 2.0
                start.scale = Vector3(x=bs, y=bs, z=bs)
                start.color = ColorRGBA(r=0.1, g=0.95, b=0.3, a=1.0)
                best_markers.markers.append(start)

                # End sphere (bright green)
                end = Marker()
                end.header = header
                end.ns = 'best_rollout'
                end.id = 2
                end.type = Marker.SPHERE
                end.action = Marker.ADD
                end.pose.position = best_points[-1]
                end.scale = Vector3(x=bs, y=bs, z=bs)
                end.color = ColorRGBA(r=0.95, g=1.0, b=0.2, a=1.0)
                best_markers.markers.append(end)

                # Reward text
                txt = Marker()
                txt.header = header
                txt.ns = 'best_rollout'
                txt.id = 3
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position = Point(
                    x=best_points[-1].x,
                    y=best_points[-1].y,
                    z=best_points[-1].z + 0.05,
                )
                txt.scale = Vector3(x=0.0, y=0.0, z=0.02)
                txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                txt.text = (
                    f'R={best_traj.total_reward:.2f} '
                    f'({best_traj.steps_per_second:.0f} step/s)'
                )
                best_markers.markers.append(txt)

            self.pub_best.publish(best_markers)

        # ── Uncertainty visualization ─────────────────────────
        if self.get_parameter('show_uncertainty').value and len(trajs) > 1:
            self._publish_uncertainty(trajs, header, scale, dims)

        # ── Imagined frames from best trajectory ──────────────
        if (self.get_parameter('show_frames').value
                and best_idx < len(trajs)):
            self._publish_frames(trajs[best_idx], now)

    def _publish_uncertainty(
        self,
        trajs: List[RolloutTrajectory],
        header: Header,
        scale: float,
        dims: Tuple[int, int, int],
    ):
        """
        Compute per-timestep variance across candidates and show
        as translucent spheres (uncertainty cones).
        """
        markers = MarkerArray()

        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        # Find the common horizon length
        min_horizon = min(t.horizon for t in trajs)
        if min_horizon < 1:
            self.pub_uncertainty.publish(markers)
            return

        latent_dim = trajs[0].latent_dim
        if latent_dim < 1:
            self.pub_uncertainty.publish(markers)
            return

        # Collect all latent trajectories into array
        # Shape: (num_trajs, num_states, latent_dim)
        all_latents = []
        for traj in trajs:
            arr = np.array(traj.predicted_latents, dtype=np.float32)
            num_states = len(arr) // latent_dim
            if num_states < min_horizon:
                continue
            reshaped = arr[:min_horizon * latent_dim].reshape(min_horizon, latent_dim)
            all_latents.append(reshaped)

        if len(all_latents) < 2:
            self.pub_uncertainty.publish(markers)
            return

        stacked = np.stack(all_latents, axis=0)  # (N, T, D)
        means = stacked.mean(axis=0)              # (T, D)
        stds = stacked.std(axis=0)                # (T, D)

        for t in range(min_horizon):
            # Position from mean latent
            x = float(means[t, dims[0]]) * scale if dims[0] < latent_dim else 0.0
            y = float(means[t, dims[1]]) * scale if dims[1] < latent_dim else 0.0
            z = float(means[t, dims[2]]) * scale if dims[2] < latent_dim else 0.0

            # Size from std
            sx = float(stds[t, dims[0]]) * scale * 2.0 if dims[0] < latent_dim else 0.01
            sy = float(stds[t, dims[1]]) * scale * 2.0 if dims[1] < latent_dim else 0.01
            sz = float(stds[t, dims[2]]) * scale * 2.0 if dims[2] < latent_dim else 0.01

            # Clamp minimum size
            min_s = 0.003
            sx = max(sx, min_s)
            sy = max(sy, min_s)
            sz = max(sz, min_s)

            sphere = Marker()
            sphere.header = header
            sphere.ns = 'uncertainty'
            sphere.id = t
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(x=x, y=y, z=z)
            sphere.scale = Vector3(x=sx, y=sy, z=sz)

            # Fade alpha over time
            alpha = max(0.05, 0.2 - 0.15 * (t / max(1, min_horizon - 1)))
            sphere.color = _uncertainty_color(alpha=alpha)
            markers.markers.append(sphere)

        self.pub_uncertainty.publish(markers)

    def _publish_frames(self, traj: RolloutTrajectory, stamp):
        """
        Publish the last decoded frame from the best rollout as a
        sensor_msgs/Image for display in RViz or rqt_image_view.
        """
        print(traj)
        if not traj.predicted_frames:
            return
        if traj.frame_height == 0 or traj.frame_width == 0:
            return

        h = traj.frame_height
        w = traj.frame_width
        c = traj.frame_channels or 3

        frame_size = h * w * c
        total = len(traj.predicted_frames)

        if total < frame_size:
            return

        # Take the last frame
        last_offset = total - frame_size
        frame_data = traj.predicted_frames[last_offset:]
        frame_bytes = bytes(int(v) & 0xFF for v in frame_data)

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'wm_imagined'
        msg.height = h
        msg.width = w
        msg.encoding = 'rgb8' if c == 3 else 'mono8'
        msg.is_bigendian = False
        msg.step = w * c
        msg.data = frame_bytes

        self.pub_frames.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RolloutVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()