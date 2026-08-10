"""Goal-conditioned velocity command for object-front navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def sample_localization_position_noise(
  num_samples: int,
  device: str | torch.device,
  nominal_std: float,
  outlier_probability: float,
  outlier_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample planar zero-mean Gaussian noise with rare large outliers."""
  is_outlier = torch.rand(num_samples, device=device) < outlier_probability
  sample_std = torch.full((num_samples,), nominal_std, device=device)
  sample_std[is_outlier] = outlier_std
  position_noise_w = torch.randn(num_samples, 2, device=device)
  position_noise_w *= sample_std.unsqueeze(-1)
  return position_noise_w, is_outlier


class GoalPoseCommand(CommandTerm):
  """Sample an object pose and generate a velocity command toward its front.

  The marker is placed on the side of the object facing the robot.  The desired
  robot pose is ``stand_off_distance`` in front of that marker, facing the
  object.  ``command`` intentionally remains a body-frame ``(vx, vy, wz)``
  command so the original velocity-task rewards and gait shaping can be reused.
  """

  cfg: GoalPoseCommandCfg

  def __init__(self, cfg: GoalPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    camera_site_ids, camera_site_names = self.robot.find_sites(cfg.camera_site_name)
    if len(camera_site_ids) != 1:
      raise ValueError(
        f"Expected one camera site matching '{cfg.camera_site_name}', "
        f"found {camera_site_names}."
      )
    self.camera_site_id = camera_site_ids[0]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.target_heading_w = torch.zeros(self.num_envs, device=self.device)
    self.object_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.object_heading_w = torch.zeros(self.num_envs, device=self.device)
    self.object_height_w = torch.zeros(self.num_envs, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_position_reached = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_goal_reached = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.localization_position_noise_w = torch.zeros(
      self.num_envs, 2, device=self.device
    )
    self.is_localization_outlier = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["position_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["heading_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["goal_reached"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Goal-directed body-frame velocity command used by the velocity task."""
    return self.vel_command_b

  def _root_pose_from_qpos(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the current root pose, including during reset before ``forward``.

    Command reset runs after reset events but before MuJoCo forward kinematics.
    Reading qpos here prevents a newly sampled goal from being anchored to the
    previous episode's stale derived body pose.
    """
    root_pose = self.robot.data.data.qpos[
      :, self.robot.data.indexing.free_joint_q_adr
    ]
    root_pos_w = root_pose[:, :3]
    root_quat_w = root_pose[:, 3:7]
    forward_b = torch.zeros(self.num_envs, 3, device=self.device)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(root_quat_w, forward_b)
    heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
    return root_pos_w, root_quat_w, heading_w

  def _pose_error_b(
    self, *, use_localization_estimate: bool = False
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root_pos_w, _, heading_w = self._root_pose_from_qpos()
    root_pos_xy_w = root_pos_w[:, :2]
    if use_localization_estimate:
      root_pos_xy_w = root_pos_xy_w + self.localization_position_noise_w
    delta_w = self.target_pos_w - root_pos_xy_w
    cos_yaw = torch.cos(heading_w)
    sin_yaw = torch.sin(heading_w)
    delta_x_b = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
    delta_y_b = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
    delta_b = torch.stack((delta_x_b, delta_y_b), dim=-1)
    distance = torch.linalg.vector_norm(delta_b, dim=-1)
    heading_error = wrap_to_pi(self.target_heading_w - heading_w)
    return delta_b, distance, heading_error

  @property
  def target_pose_b(self) -> torch.Tensor:
    """Desired robot pose relative to its base.

    Returns ``(x, y, distance, sin(yaw_error), cos(yaw_error))``.  This is the
    processed SE(2) target obtained after applying the known marker-to-goal
    stand-off transform.
    """
    delta_b, distance, heading_error = self._pose_error_b()
    return torch.cat(
      (
        delta_b,
        distance.unsqueeze(-1),
        torch.sin(heading_error).unsqueeze(-1),
        torch.cos(heading_error).unsqueeze(-1),
      ),
      dim=-1,
    )

  @property
  def marker_pose_camera(self) -> torch.Tensor:
    """Ground-truth ArUco-like object pose in the head-camera frame."""
    return self._marker_pose_camera(use_localization_estimate=False)

  @property
  def localized_marker_pose_camera(self) -> torch.Tensor:
    """Actor marker pose using the same noisy localization as the command."""
    return self._marker_pose_camera(use_localization_estimate=True)

  def _marker_pose_camera(
    self, *, use_localization_estimate: bool
  ) -> torch.Tensor:
    """Return ``(x, y, z, distance, sin(yaw_error), cos(yaw_error))``."""
    camera_pos_w = self.robot.data.site_pos_w[:, self.camera_site_id, :]
    camera_quat_w = self.robot.data.site_quat_w[:, self.camera_site_id, :]
    marker_pos_w = torch.cat(
      (self.object_pos_w, self.object_height_w.unsqueeze(-1)), dim=-1
    )
    marker_delta_w = marker_pos_w - camera_pos_w
    if use_localization_estimate:
      marker_delta_w = marker_delta_w.clone()
      marker_delta_w[:, :2] -= self.localization_position_noise_w
    marker_pos_camera = quat_apply_inverse(
      camera_quat_w, marker_delta_w
    )
    distance = torch.linalg.vector_norm(marker_pos_camera, dim=-1)

    camera_forward_b = torch.zeros_like(marker_pos_camera)
    camera_forward_b[:, 0] = 1.0
    camera_forward_w = quat_apply(camera_quat_w, camera_forward_b)
    camera_heading_w = torch.atan2(
      camera_forward_w[:, 1], camera_forward_w[:, 0]
    )
    marker_heading_error = wrap_to_pi(
      self.object_heading_w - camera_heading_w
    )
    return torch.cat(
      (
        marker_pos_camera,
        distance.unsqueeze(-1),
        torch.sin(marker_heading_error).unsqueeze(-1),
        torch.cos(marker_heading_error).unsqueeze(-1),
      ),
      dim=-1,
    )

  def _update_metrics(self) -> None:
    _, distance, heading_error = self._pose_error_b()
    normalization_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["position_error"] += distance / normalization_steps
    self.metrics["heading_error"] += torch.abs(heading_error) / normalization_steps
    self.metrics["goal_reached"] += (
      self.is_goal_reached.float() / normalization_steps
    )

  def _sample_localization_noise(
    self, env_ids: torch.Tensor | None = None
  ) -> None:
    indices = slice(None) if env_ids is None else env_ids
    num_samples = self.num_envs if env_ids is None else len(env_ids)
    if not self.cfg.localization_noise_enabled:
      self.localization_position_noise_w[indices] = 0.0
      self.is_localization_outlier[indices] = False
      return

    position_noise_w, is_outlier = sample_localization_position_noise(
      num_samples,
      self.device,
      self.cfg.localization_position_noise_std,
      self.cfg.localization_outlier_probability,
      self.cfg.localization_outlier_position_std,
    )
    self.localization_position_noise_w[indices] = position_noise_w
    self.is_localization_outlier[indices] = is_outlier

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # A new object invalidates the previous arrival latch.
    self.is_position_reached[env_ids] = False
    self.is_goal_reached[env_ids] = False
    root_pos_w, _, root_heading_w = self._root_pose_from_qpos()
    object_distance = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.ranges.object_distance
    )
    object_bearing_b = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.ranges.object_bearing
    )
    standing = (
      torch.rand(len(env_ids), device=self.device) < self.cfg.rel_standing_envs
    )
    object_distance = torch.where(
      standing,
      torch.full_like(object_distance, self.cfg.stand_off_distance),
      object_distance,
    )
    object_bearing_b = torch.where(
      standing, torch.zeros_like(object_bearing_b), object_bearing_b
    )
    self.is_standing_env[env_ids] = standing

    approach_heading_w = wrap_to_pi(
      root_heading_w[env_ids] + object_bearing_b
    )
    approach_direction_w = torch.stack(
      (torch.cos(approach_heading_w), torch.sin(approach_heading_w)), dim=-1
    )
    self.object_pos_w[env_ids] = (
      root_pos_w[env_ids, :2]
      + object_distance.unsqueeze(-1) * approach_direction_w
    )
    self.object_heading_w[env_ids] = wrap_to_pi(
      approach_heading_w + torch.pi
    )
    self.object_height_w[env_ids] = (
      root_pos_w[env_ids, 2] + self.cfg.marker_height_offset
    )

    self.target_pos_w[env_ids] = (
      self.object_pos_w[env_ids]
      - self.cfg.stand_off_distance * approach_direction_w
    )
    self.target_heading_w[env_ids] = approach_heading_w
    self._sample_localization_noise(env_ids)
    self._update_velocity_command(env_ids)

  def _update_velocity_command(
    self,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)

    delta_b, distance, heading_error = self._pose_error_b(
      use_localization_estimate=True
    )
    ranges = self.cfg.ranges
    self.vel_command_b[env_ids, 0] = torch.clamp(
      self.cfg.position_control_stiffness * delta_b[env_ids, 0],
      min=ranges.lin_vel_x[0],
      max=ranges.lin_vel_x[1],
    )
    self.vel_command_b[env_ids, 1] = torch.clamp(
      self.cfg.position_control_stiffness * delta_b[env_ids, 1],
      min=ranges.lin_vel_y[0],
      max=ranges.lin_vel_y[1],
    )
    self.vel_command_b[env_ids, 2] = torch.clamp(
      self.cfg.heading_control_stiffness * heading_error[env_ids],
      min=ranges.ang_vel_z[0],
      max=ranges.ang_vel_z[1],
    )

    position_reached = distance[env_ids] < self.cfg.position_tolerance
    heading_reached = (
      torch.abs(heading_error[env_ids]) < self.cfg.heading_tolerance
    )

    # Preserve the stop latch through small post-arrival motion, but recover
    # from a genuine overshoot.  The wider release radius prevents the old
    # threshold chatter/creep while still allowing a robot that drifts away to
    # approach the same goal again.
    needs_recovery = self.is_position_reached[env_ids] & (
      distance[env_ids] > self.cfg.position_recovery_tolerance
    )
    self.is_position_reached[env_ids] &= ~needs_recovery
    self.is_goal_reached[env_ids] &= ~needs_recovery

    # Keep a minimum translational speed outside the target sphere.  A pure
    # proportional controller becomes arbitrarily slow near the goal and can
    # leave the robot creeping just outside the arrival threshold.
    linear_command = self.vel_command_b[env_ids, :2]
    linear_speed = torch.linalg.vector_norm(linear_command, dim=-1)
    keep_approach_speed = (
      (~position_reached)
      & (linear_speed > 1.0e-6)
      & (linear_speed < self.cfg.min_approach_speed)
    )
    speed_scale = torch.where(
      keep_approach_speed,
      self.cfg.min_approach_speed / torch.clamp(linear_speed, min=1.0e-6),
      torch.ones_like(linear_speed),
    )
    self.vel_command_b[env_ids, :2] *= speed_scale.unsqueeze(-1)

    # Do not brake before reaching the sphere.  Translation switches directly
    # from the minimum approach speed to zero on the first step inside it.
    self.is_position_reached[env_ids] |= position_reached
    self.is_goal_reached[env_ids] |= (
      self.is_position_reached[env_ids] & heading_reached
    )
    self.vel_command_b[env_ids, :2] *= (
      ~self.is_position_reached[env_ids]
    ).unsqueeze(-1)
    self.vel_command_b[env_ids, 2] *= ~heading_reached

    # Keep the full stop latched while the base remains inside the recovery
    # radius.  This prevents tiny position/heading threshold crossings from
    # re-enabling commands and causing slow creep.
    latched = self.is_goal_reached[env_ids]
    self.vel_command_b[env_ids] *= (~latched).unsqueeze(-1)

  def _update_command(self) -> None:
    self._sample_localization_noise()
    self._update_velocity_command()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    root_z = self.robot.data.root_link_pos_w[:, 2].cpu().numpy()
    target_pos = self.target_pos_w.cpu().numpy()
    target_heading = self.target_heading_w.cpu().numpy()
    object_pos = self.object_pos_w.cpu().numpy()
    object_height = self.object_height_w.cpu().numpy()
    object_heading = self.object_heading_w.cpu().numpy()
    camera_pos = self.robot.data.site_pos_w[:, self.camera_site_id, :].cpu().numpy()

    for env_idx in env_indices:
      target_center = np.array(
        [target_pos[env_idx, 0], target_pos[env_idx, 1], root_z[env_idx]]
      )
      marker_center = np.array(
        [
          object_pos[env_idx, 0],
          object_pos[env_idx, 1],
          object_height[env_idx],
        ]
      )
      target_yaw = target_heading[env_idx]
      target_matrix = np.array(
        [
          [np.cos(target_yaw), -np.sin(target_yaw), 0.0],
          [np.sin(target_yaw), np.cos(target_yaw), 0.0],
          [0.0, 0.0, 1.0],
        ]
      )
      marker_yaw = object_heading[env_idx]
      marker_matrix = np.array(
        [
          [np.cos(marker_yaw), -np.sin(marker_yaw), 0.0],
          [np.sin(marker_yaw), np.cos(marker_yaw), 0.0],
          [0.0, 0.0, 1.0],
        ]
      )
      visualizer.add_sphere(
        target_center,
        radius=self.cfg.position_tolerance,
        color=(0.1, 0.9, 0.2, 0.7),
        label="navigation target",
      )
      visualizer.add_frame(
        target_center, target_matrix, scale=0.25, label="target pose"
      )
      visualizer.add_frame(
        marker_center, marker_matrix, scale=0.18, label="ArUco marker"
      )
      visualizer.add_arrow(
        camera_pos[env_idx],
        marker_center,
        color=(0.2, 0.6, 1.0, 0.7),
        width=0.01,
        label="camera to marker",
      )


@dataclass(kw_only=True)
class GoalPoseCommandCfg(CommandTermCfg):
  """Configuration for :class:`GoalPoseCommand`."""

  entity_name: str
  camera_site_name: str
  stand_off_distance: float = 0.7
  marker_height_offset: float = 0.42
  rel_standing_envs: float = 0.1
  position_control_stiffness: float = 1.0
  heading_control_stiffness: float = 1.0
  min_approach_speed: float = 0.35
  localization_noise_enabled: bool = True
  localization_position_noise_std: float = 0.01
  localization_outlier_probability: float = 0.01
  localization_outlier_position_std: float = 1.0
  position_tolerance: float = 0.08
  position_recovery_tolerance: float = 0.10
  heading_tolerance: float = 0.20

  @dataclass
  class Ranges:
    object_distance: tuple[float, float]
    object_bearing: tuple[float, float]
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    marker_size: float = 0.18

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> GoalPoseCommand:
    return GoalPoseCommand(self, env)

  def __post_init__(self) -> None:
    if self.stand_off_distance <= 0.0:
      raise ValueError("stand_off_distance must be positive.")
    if self.ranges.object_distance[0] < self.stand_off_distance:
      raise ValueError(
        "The minimum object distance must be at least stand_off_distance."
      )
    if not 0.0 <= self.rel_standing_envs <= 1.0:
      raise ValueError("rel_standing_envs must be in [0, 1].")
    if self.min_approach_speed < 0.0:
      raise ValueError("min_approach_speed must be non-negative.")
    if self.localization_position_noise_std < 0.0:
      raise ValueError(
        "localization_position_noise_std must be non-negative."
      )
    if not 0.0 <= self.localization_outlier_probability <= 1.0:
      raise ValueError(
        "localization_outlier_probability must be in [0, 1]."
      )
    if self.localization_outlier_position_std < 0.0:
      raise ValueError(
        "localization_outlier_position_std must be non-negative."
      )
    if self.position_recovery_tolerance <= self.position_tolerance:
      raise ValueError(
        "position_recovery_tolerance must be greater than position_tolerance."
      )
