"""Position-indexed piecewise-arc commands for path-following navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, wrap_to_pi

from .velocity_command import sample_localization_position_noise

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _arc_poses(curvature: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
  """Return ``(x, y, theta)`` on origin-tangent constant-curvature arcs."""
  phi = curvature * progress
  x = progress * torch.sinc(phi / torch.pi)
  y = 0.5 * progress * phi * torch.sinc(phi / (2.0 * torch.pi)).square()
  return torch.stack((x, y, phi), dim=-1)


def _smoothstep(ratio: torch.Tensor) -> torch.Tensor:
  ratio = torch.clamp(ratio, 0.0, 1.0)
  return ratio.square() * (3.0 - 2.0 * ratio)


class TrajectoryCommand(CommandTerm):
  """Generate a path and reference it from the robot's closest sampled pose."""

  cfg: TrajectoryCommandCfg

  def __init__(self, cfg: TrajectoryCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self.motion_steps = round(cfg.motion_duration / env.step_dt)
    self.episode_steps = round(cfg.episode_duration / env.step_dt)
    self.start_ramp_steps = round(cfg.start_ramp_duration / env.step_dt)
    self.min_segment_steps = int(np.ceil(cfg.min_segment_duration / env.step_dt))
    self.reference_step_offsets = torch.tensor(
      [round(offset / env.step_dt) for offset in cfg.reference_times],
      device=self.device,
      dtype=torch.long,
    )
    if not np.isclose(self.motion_steps * env.step_dt, cfg.motion_duration):
      raise ValueError("motion_duration must be an integer number of control steps.")
    if not np.isclose(self.episode_steps * env.step_dt, cfg.episode_duration):
      raise ValueError("episode_duration must be an integer number of control steps.")
    if cfg.num_segments_range[1] * self.min_segment_steps > self.motion_steps:
      raise ValueError("Maximum segment count does not fit in motion_duration.")

    num_poses = self.episode_steps + 1
    num_path_poses = 2 * self.motion_steps + 1
    self.trajectory_poses_w = torch.zeros(
      self.num_envs, num_poses, 3, device=self.device
    )
    self.trajectory_progresses = torch.zeros(
      self.num_envs, num_poses, device=self.device
    )
    self.trajectory_linear_velocities = torch.zeros(
      self.num_envs, self.episode_steps, device=self.device
    )
    self.trajectory_angular_velocities = torch.zeros_like(
      self.trajectory_linear_velocities
    )
    self.path_poses_w = torch.zeros(
      self.num_envs, num_path_poses, 3, device=self.device
    )
    self.path_progresses = torch.zeros(
      self.num_envs, num_path_poses, device=self.device
    )
    self.trajectory_step = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.trajectory_step_ground_truth = torch.zeros_like(self.trajectory_step)
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.localization_position_noise_w = torch.zeros(
      self.num_envs, 2, device=self.device
    )
    self.is_localization_outlier = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.previous_closest_progress = torch.zeros(
      self.num_envs, device=self.device
    )
    self.closest_progress = torch.zeros(self.num_envs, device=self.device)
    self.step_progress = torch.zeros(self.num_envs, device=self.device)
    self.closest_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.closest_heading_w = torch.zeros(self.num_envs, device=self.device)
    self.distance_to_path = torch.zeros(self.num_envs, device=self.device)
    self.path_heading_error = torch.zeros(self.num_envs, device=self.device)

    self.metrics["path_progress"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["distance_to_path"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["path_heading_error"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    """Body-frame reference ``(vx, vy, wz)`` at the closest path pose."""
    return self.vel_command_b

  @property
  def future_pose_b(self) -> torch.Tensor:
    """Noisy future path poses as ``(x, y, cos(theta), sin(theta))`` blocks."""
    return self._future_pose_b(use_localization_estimate=True)

  @property
  def future_pose_b_ground_truth(self) -> torch.Tensor:
    """Ground-truth future path poses for the privileged critic."""
    return self._future_pose_b(use_localization_estimate=False)

  def _root_pose_from_qpos(self) -> tuple[torch.Tensor, torch.Tensor]:
    root_pose = self.robot.data.data.qpos[
      :, self.robot.data.indexing.free_joint_q_adr
    ]
    root_pos_w = root_pose[:, :3]
    root_quat_w = root_pose[:, 3:7]
    forward_b = torch.zeros(self.num_envs, 3, device=self.device)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(root_quat_w, forward_b)
    heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
    return root_pos_w, heading_w

  def _future_pose_b(self, *, use_localization_estimate: bool) -> torch.Tensor:
    env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(-1)
    reference_step = (
      self.trajectory_step
      if use_localization_estimate
      else self.trajectory_step_ground_truth
    )
    pose_indices = torch.clamp(
      reference_step.unsqueeze(-1) + self.reference_step_offsets,
      max=self.episode_steps,
    )
    future_poses_w = self.trajectory_poses_w[env_ids, pose_indices]

    root_pos_w, root_heading_w = self._root_pose_from_qpos()
    root_pos_xy_w = root_pos_w[:, :2]
    if use_localization_estimate:
      root_pos_xy_w = root_pos_xy_w + self.localization_position_noise_w
    delta_w = future_poses_w[:, :, :2] - root_pos_xy_w.unsqueeze(1)
    cos_yaw = torch.cos(root_heading_w).unsqueeze(-1)
    sin_yaw = torch.sin(root_heading_w).unsqueeze(-1)
    delta_x_b = cos_yaw * delta_w[:, :, 0] + sin_yaw * delta_w[:, :, 1]
    delta_y_b = -sin_yaw * delta_w[:, :, 0] + cos_yaw * delta_w[:, :, 1]
    heading_error = wrap_to_pi(
      future_poses_w[:, :, 2] - root_heading_w.unsqueeze(-1)
    )
    future_pose = torch.stack(
      (
        delta_x_b,
        delta_y_b,
        torch.cos(heading_error),
        torch.sin(heading_error),
      ),
      dim=-1,
    )
    return future_pose.flatten(start_dim=1)

  def _closest_trajectory_steps(self, positions_w: torch.Tensor) -> torch.Tensor:
    """Return the nearest sampled forward-path index for each position."""
    forward_path = self.trajectory_poses_w[:, : self.motion_steps + 1, :2]
    distance_sq = torch.sum(
      (forward_path - positions_w.unsqueeze(1)).square(), dim=-1
    )
    return torch.argmin(distance_sq, dim=-1)

  def _sample_segment_steps(self, num_samples: int) -> torch.Tensor:
    min_segments, max_segments = self.cfg.num_segments_range
    segment_counts = torch.randint(
      min_segments,
      max_segments + 1,
      (num_samples,),
      device=self.device,
    )
    segment_steps = torch.zeros(
      num_samples, max_segments, device=self.device, dtype=torch.long
    )
    for count in range(min_segments, max_segments + 1):
      mask = segment_counts == count
      sample_count = int(mask.sum().item())
      if sample_count == 0:
        continue
      extra_steps = self.motion_steps - count * self.min_segment_steps
      if extra_steps == 0:
        extras = torch.zeros(
          sample_count, count, device=self.device, dtype=torch.long
        )
      else:
        probabilities = torch.full(
          (count,), 1.0 / count, device=self.device
        )
        extras = torch.distributions.Multinomial(
          total_count=extra_steps, probs=probabilities
        ).sample((sample_count,)).to(torch.long)
      segment_steps[mask, :count] = extras + self.min_segment_steps
    return segment_steps

  def _sample_curvatures(self, shape: tuple[int, int]) -> torch.Tensor:
    magnitude = torch.rand(shape, device=self.device).pow(
      self.cfg.curvature_exponent
    ) / self.cfg.min_radius
    sign = torch.where(
      torch.rand(shape, device=self.device) < 0.5,
      -torch.ones_like(magnitude),
      torch.ones_like(magnitude),
    )
    curvature = sign * magnitude
    straight = (
      torch.rand(shape, device=self.device) < self.cfg.straight_probability
    )
    return torch.where(straight, torch.zeros_like(curvature), curvature)

  def _speed_scale(self, midpoint_time: torch.Tensor) -> torch.Tensor:
    scale = torch.ones_like(midpoint_time)
    scale = torch.where(
      midpoint_time < self.cfg.start_ramp_duration,
      _smoothstep(midpoint_time / self.cfg.start_ramp_duration),
      scale,
    )
    stop_start = self.cfg.motion_duration - self.cfg.stop_ramp_duration
    scale = torch.where(
      midpoint_time > stop_start,
      _smoothstep(
        (self.cfg.motion_duration - midpoint_time)
        / self.cfg.stop_ramp_duration
      ),
      scale,
    )
    return scale

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
    num_samples = len(env_ids)
    root_pos_w, root_heading_w = self._root_pose_from_qpos()
    anchor_pos_w = root_pos_w[env_ids, :2]
    anchor_heading_w = root_heading_w[env_ids]

    standing = torch.rand(num_samples, device=self.device) < self.cfg.rel_standing_envs
    self.is_standing_env[env_ids] = standing
    segment_steps = self._sample_segment_steps(num_samples)
    segment_ends = torch.cumsum(segment_steps, dim=-1)
    curvatures = self._sample_curvatures(segment_steps.shape)
    curvatures[segment_steps == 0] = 0.0
    se2_speed = torch.empty(num_samples, device=self.device).uniform_(
      *self.cfg.se2_speed_range
    )
    se2_speed[standing] = 0.0

    local_poses = torch.zeros(
      num_samples, self.episode_steps + 1, 3, device=self.device
    )
    progresses = torch.zeros(
      num_samples, self.episode_steps + 1, device=self.device
    )
    linear_velocities = torch.zeros(
      num_samples, self.episode_steps, device=self.device
    )
    angular_velocities = torch.zeros_like(linear_velocities)

    for step in range(self.motion_steps):
      segment_index = torch.sum(step >= segment_ends, dim=-1)
      segment_index = torch.clamp(segment_index, max=curvatures.shape[1] - 1)
      curvature = torch.gather(
        curvatures, 1, segment_index.unsqueeze(-1)
      ).squeeze(-1)
      midpoint_time = torch.full(
        (num_samples,),
        (step + 0.5) * self._env.step_dt,
        device=self.device,
      )
      base_v = se2_speed / torch.sqrt(
        1.0 + (self.cfg.characteristic_length * curvature).square()
      )
      linear_velocity = base_v * self._speed_scale(midpoint_time)
      angular_velocity = curvature * linear_velocity
      linear_velocities[:, step] = linear_velocity
      angular_velocities[:, step] = angular_velocity

      ds = linear_velocity * self._env.step_dt
      local_delta = _arc_poses(curvature, ds)
      heading = local_poses[:, step, 2]
      cos_heading = torch.cos(heading)
      sin_heading = torch.sin(heading)
      local_poses[:, step + 1, 0] = (
        local_poses[:, step, 0]
        + cos_heading * local_delta[:, 0]
        - sin_heading * local_delta[:, 1]
      )
      local_poses[:, step + 1, 1] = (
        local_poses[:, step, 1]
        + sin_heading * local_delta[:, 0]
        + cos_heading * local_delta[:, 1]
      )
      local_poses[:, step + 1, 2] = heading + local_delta[:, 2]
      progresses[:, step + 1] = progresses[:, step] + ds

    local_poses[:, self.motion_steps + 1 :] = local_poses[
      :, self.motion_steps : self.motion_steps + 1
    ]
    progresses[:, self.motion_steps + 1 :] = progresses[
      :, self.motion_steps : self.motion_steps + 1
    ]
    accumulated_rotation = torch.sum(
      torch.abs(angular_velocities), dim=-1
    ) * self._env.step_dt
    if bool(torch.any(accumulated_rotation >= 2.0 * torch.pi).item()):
      raise ValueError("A sampled trajectory accumulates one or more full turns.")

    cos_anchor = torch.cos(anchor_heading_w).unsqueeze(-1)
    sin_anchor = torch.sin(anchor_heading_w).unsqueeze(-1)
    poses_w = torch.empty_like(local_poses)
    poses_w[:, :, 0] = (
      anchor_pos_w[:, 0].unsqueeze(-1)
      + cos_anchor * local_poses[:, :, 0]
      - sin_anchor * local_poses[:, :, 1]
    )
    poses_w[:, :, 1] = (
      anchor_pos_w[:, 1].unsqueeze(-1)
      + sin_anchor * local_poses[:, :, 0]
      + cos_anchor * local_poses[:, :, 1]
    )
    poses_w[:, :, 2] = anchor_heading_w.unsqueeze(-1) + local_poses[:, :, 2]

    total_progress = progresses[:, self.motion_steps]
    backward_ratio = torch.linspace(
      -1.0, 0.0, self.motion_steps + 1, device=self.device
    )
    backward_progress = total_progress.unsqueeze(-1) * backward_ratio
    backward_local = _arc_poses(
      curvatures[:, :1], backward_progress
    )
    backward_w = torch.empty_like(backward_local)
    backward_w[:, :, 0] = (
      anchor_pos_w[:, 0].unsqueeze(-1)
      + cos_anchor * backward_local[:, :, 0]
      - sin_anchor * backward_local[:, :, 1]
    )
    backward_w[:, :, 1] = (
      anchor_pos_w[:, 1].unsqueeze(-1)
      + sin_anchor * backward_local[:, :, 0]
      + cos_anchor * backward_local[:, :, 1]
    )
    backward_w[:, :, 2] = (
      anchor_heading_w.unsqueeze(-1) + backward_local[:, :, 2]
    )

    self.trajectory_poses_w[env_ids] = poses_w
    self.trajectory_progresses[env_ids] = progresses
    self.trajectory_linear_velocities[env_ids] = linear_velocities
    self.trajectory_angular_velocities[env_ids] = angular_velocities
    self.path_poses_w[env_ids] = torch.cat(
      (backward_w[:, :-1], poses_w[:, : self.motion_steps + 1]), dim=1
    )
    self.path_progresses[env_ids] = torch.cat(
      (backward_progress[:, :-1], progresses[:, : self.motion_steps + 1]),
      dim=1,
    )
    self.trajectory_step[env_ids] = 0
    self.trajectory_step_ground_truth[env_ids] = 0
    self.previous_closest_progress[env_ids] = 0.0
    self.closest_progress[env_ids] = 0.0
    self.step_progress[env_ids] = 0.0
    self.closest_pos_w[env_ids] = anchor_pos_w
    self.closest_heading_w[env_ids] = anchor_heading_w
    self.distance_to_path[env_ids] = 0.0
    self.path_heading_error[env_ids] = 0.0
    self._sample_localization_noise(env_ids)
    self._update_reference_command(env_ids)

  def _update_reference_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    root_pos_w, root_heading_w = self._root_pose_from_qpos()
    estimated_pos_w = (
      root_pos_w[:, :2] + self.localization_position_noise_w
    )
    estimated_step = self._closest_trajectory_steps(estimated_pos_w)
    ground_truth_step = self._closest_trajectory_steps(root_pos_w[:, :2])
    self.trajectory_step[env_ids] = estimated_step[env_ids]
    self.trajectory_step_ground_truth[env_ids] = ground_truth_step[env_ids]

    step = estimated_step[env_ids]
    # A purely position-indexed speed profile has an almost-zero velocity at
    # index zero and can leave a learned policy standing there indefinitely.
    # Advance only the startup speed ramp by wall time; path pose references
    # and all lookaheads remain based on the closest position index.
    elapsed_time = self.cfg.episode_duration - self.time_left[env_ids]
    startup_step = torch.round(elapsed_time / self._env.step_dt).to(torch.long)
    startup_step = torch.clamp(startup_step, 0, self.start_ramp_steps)
    velocity_step = torch.maximum(step, startup_step)
    velocity_step = torch.clamp(velocity_step, max=self.episode_steps - 1)
    linear_velocity = self.trajectory_linear_velocities[env_ids, velocity_step]
    angular_velocity = self.trajectory_angular_velocities[env_ids, velocity_step]
    reference_pose = self.trajectory_poses_w[env_ids, step]
    reference_heading = reference_pose[:, 2]
    position_error_w = reference_pose[:, :2] - estimated_pos_w[env_ids]
    reference_velocity_w = linear_velocity.unsqueeze(-1) * torch.stack(
      (torch.cos(reference_heading), torch.sin(reference_heading)), dim=-1
    )
    velocity_w = (
      reference_velocity_w + self.cfg.tracking_gain * position_error_w
    )
    heading_error = wrap_to_pi(reference_heading - root_heading_w[env_ids])
    cos_heading = torch.cos(root_heading_w[env_ids])
    sin_heading = torch.sin(root_heading_w[env_ids])
    self.vel_command_b[env_ids, 0] = (
      cos_heading * velocity_w[:, 0] + sin_heading * velocity_w[:, 1]
    )
    self.vel_command_b[env_ids, 1] = (
      -sin_heading * velocity_w[:, 0] + cos_heading * velocity_w[:, 1]
    )
    command_speed = torch.linalg.vector_norm(
      self.vel_command_b[env_ids, :2], dim=-1
    )
    speed_scale = torch.clamp(
      self.cfg.max_linear_speed / torch.clamp(command_speed, min=1.0e-6),
      max=1.0,
    )
    self.vel_command_b[env_ids, :2] *= speed_scale.unsqueeze(-1)
    self.vel_command_b[env_ids, 2] = torch.clamp(
      angular_velocity + self.cfg.tracking_gain * heading_error,
      min=-self.cfg.max_angular_speed,
      max=self.cfg.max_angular_speed,
    )

  def update_path_tracking(self) -> None:
    """Project the current robot pose onto the sampled path and update progress."""
    root_pos_w, root_heading_w = self._root_pose_from_qpos()
    robot_pos = root_pos_w[:, :2]
    starts = self.path_poses_w[:, :-1, :2]
    vectors = self.path_poses_w[:, 1:, :2] - starts
    lengths_sq = torch.sum(vectors.square(), dim=-1)
    valid = lengths_sq > 1.0e-16
    safe_lengths_sq = torch.where(valid, lengths_sq, torch.ones_like(lengths_sq))
    fractions = torch.sum(
      (robot_pos.unsqueeze(1) - starts) * vectors, dim=-1
    ) / safe_lengths_sq
    fractions = torch.clamp(fractions, 0.0, 1.0)
    points = starts + fractions.unsqueeze(-1) * vectors
    distance_sq = torch.sum(
      (points - robot_pos.unsqueeze(1)).square(), dim=-1
    )
    distance_sq = torch.where(
      valid, distance_sq, torch.full_like(distance_sq, torch.inf)
    )

    path_progress_delta = self.path_progresses[:, 1:] - self.path_progresses[:, :-1]
    candidate_progress = (
      self.path_progresses[:, :-1] + fractions * path_progress_delta
    )
    min_distance_sq = torch.min(distance_sq, dim=-1, keepdim=True).values
    tied = valid & (distance_sq <= min_distance_sq + 1.0e-12)
    tie_distance = torch.abs(
      candidate_progress - self.previous_closest_progress.unsqueeze(-1)
    )
    tie_distance = torch.where(
      tied, tie_distance, torch.full_like(tie_distance, torch.inf)
    )
    chosen = torch.argmin(tie_distance, dim=-1)
    has_valid_segment = torch.any(valid, dim=-1)

    gather_xy = chosen[:, None, None].expand(-1, 1, 2)
    closest_pos = torch.gather(points, 1, gather_xy).squeeze(1)
    chosen_fraction = torch.gather(fractions, 1, chosen.unsqueeze(-1)).squeeze(-1)
    closest_progress = torch.gather(
      candidate_progress, 1, chosen.unsqueeze(-1)
    ).squeeze(-1)
    heading_start = torch.gather(
      self.path_poses_w[:, :-1, 2], 1, chosen.unsqueeze(-1)
    ).squeeze(-1)
    heading_delta = torch.gather(
      self.path_poses_w[:, 1:, 2] - self.path_poses_w[:, :-1, 2],
      1,
      chosen.unsqueeze(-1),
    ).squeeze(-1)
    closest_heading = heading_start + chosen_fraction * heading_delta

    closest_pos = torch.where(
      has_valid_segment.unsqueeze(-1),
      closest_pos,
      self.path_poses_w[:, 0, :2],
    )
    closest_progress = torch.where(
      has_valid_segment, closest_progress, torch.zeros_like(closest_progress)
    )
    closest_heading = torch.where(
      has_valid_segment, closest_heading, self.path_poses_w[:, 0, 2]
    )

    self.step_progress = closest_progress - self.previous_closest_progress
    self.previous_closest_progress.copy_(closest_progress)
    self.closest_progress.copy_(closest_progress)
    self.closest_pos_w.copy_(closest_pos)
    self.closest_heading_w.copy_(closest_heading)
    self.distance_to_path.copy_(
      torch.linalg.vector_norm(robot_pos - closest_pos, dim=-1)
    )
    self.path_heading_error.copy_(
      wrap_to_pi(root_heading_w - closest_heading)
    )

  def _update_metrics(self) -> None:
    normalization_steps = self._env.max_episode_length
    self.metrics["path_progress"] += self.step_progress / normalization_steps
    self.metrics["distance_to_path"] += self.distance_to_path / normalization_steps
    self.metrics["path_heading_error"] += (
      torch.abs(self.path_heading_error) / normalization_steps
    )

  def _update_command(self) -> None:
    self._sample_localization_noise()
    self._update_reference_command()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    root_z = self.robot.data.root_link_pos_w[:, 2].cpu().numpy()
    path_poses = self.path_poses_w.cpu().numpy()
    future_steps = torch.clamp(
      self.trajectory_step.unsqueeze(-1) + self.reference_step_offsets,
      max=self.episode_steps,
    )
    env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(-1)
    future_poses = self.trajectory_poses_w[env_ids, future_steps].cpu().numpy()

    stride = max(1, self.motion_steps // 15)
    for env_idx in env_indices:
      for start_idx in range(self.motion_steps, 2 * self.motion_steps, stride):
        start = np.array(
          [
            path_poses[env_idx, start_idx, 0],
            path_poses[env_idx, start_idx, 1],
            root_z[env_idx],
          ]
        )
        end_idx = min(start_idx + stride, 2 * self.motion_steps)
        end = np.array(
          [
            path_poses[env_idx, end_idx, 0],
            path_poses[env_idx, end_idx, 1],
            root_z[env_idx],
          ]
        )
        visualizer.add_arrow(
          start,
          end,
          color=(0.2, 0.6, 1.0, 0.65),
          width=self.cfg.viz.path_width,
        )
      for reference_index, pose in enumerate(future_poses[env_idx]):
        yaw = pose[2]
        rotation = np.array(
          [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
          ]
        )
        center = np.array([pose[0], pose[1], root_z[env_idx]])
        visualizer.add_frame(
          center,
          rotation,
          scale=0.2,
          label=f"path +{self.cfg.reference_times[reference_index]:g}s",
        )


@dataclass(kw_only=True)
class TrajectoryCommandCfg(CommandTermCfg):
  """Configuration for :class:`TrajectoryCommand`."""

  entity_name: str
  reference_times: tuple[float, ...] = (1.0, 2.0)
  motion_duration: float = 3.0
  stop_hold_duration: float = 1.0
  start_ramp_duration: float = 0.8
  stop_ramp_duration: float = 0.8
  num_segments_range: tuple[int, int] = (1, 3)
  min_segment_duration: float = 0.7
  min_radius: float = 0.15
  straight_probability: float = 0.05
  curvature_exponent: float = 1.0
  se2_speed_range: tuple[float, float] = (0.20, 0.70)
  characteristic_length: float = 0.45
  tracking_gain: float = 3.0
  max_linear_speed: float = 1.0
  max_angular_speed: float = 1.6
  rel_standing_envs: float = 0.1
  localization_noise_enabled: bool = True
  localization_position_noise_std: float = 0.01
  localization_outlier_probability: float = 0.01
  localization_outlier_position_std: float = 1.0

  @property
  def episode_duration(self) -> float:
    return self.motion_duration + self.stop_hold_duration

  @dataclass
  class VizCfg:
    path_width: float = 0.008

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> TrajectoryCommand:
    return TrajectoryCommand(self, env)

  def __post_init__(self) -> None:
    if not self.reference_times or any(t <= 0.0 for t in self.reference_times):
      raise ValueError("reference_times must contain positive times.")
    if self.motion_duration <= 0.0 or self.stop_hold_duration < 0.0:
      raise ValueError("motion_duration must be positive and hold non-negative.")
    if self.start_ramp_duration < 0.0 or self.stop_ramp_duration < 0.0:
      raise ValueError("Ramp durations must be non-negative.")
    if self.start_ramp_duration + self.stop_ramp_duration > self.motion_duration:
      raise ValueError("Start and stop ramps do not fit in motion_duration.")
    min_segments, max_segments = self.num_segments_range
    if not 1 <= min_segments <= max_segments:
      raise ValueError("num_segments_range must satisfy 1 <= min <= max.")
    if self.min_segment_duration <= 0.0:
      raise ValueError("min_segment_duration must be positive.")
    if max_segments * self.min_segment_duration > self.motion_duration:
      raise ValueError("Maximum segment count does not fit in motion_duration.")
    if max(self.reference_times) > self.episode_duration:
      raise ValueError("Reference time exceeds the trajectory duration.")
    if self.min_radius <= 0.0:
      raise ValueError("min_radius must be positive.")
    if not 0.0 <= self.straight_probability <= 1.0:
      raise ValueError("straight_probability must be in [0, 1].")
    if self.curvature_exponent <= 0.0:
      raise ValueError("curvature_exponent must be positive.")
    speed_min, speed_max = self.se2_speed_range
    if not 0.0 < speed_min <= speed_max:
      raise ValueError("se2_speed_range must satisfy 0 < min <= max.")
    if self.characteristic_length <= 0.0:
      raise ValueError("characteristic_length must be positive.")
    if self.tracking_gain < 0.0:
      raise ValueError("tracking_gain must be non-negative.")
    if speed_max > self.max_linear_speed:
      raise ValueError("se2_speed_range can exceed max_linear_speed.")
    if speed_max / self.characteristic_length > self.max_angular_speed:
      raise ValueError("se2_speed_range can exceed max_angular_speed.")
    if not 0.0 <= self.rel_standing_envs <= 1.0:
      raise ValueError("rel_standing_envs must be in [0, 1].")
    if self.localization_position_noise_std < 0.0:
      raise ValueError("localization_position_noise_std must be non-negative.")
    if not 0.0 <= self.localization_outlier_probability <= 1.0:
      raise ValueError("localization_outlier_probability must be in [0, 1].")
    if self.localization_outlier_position_std < 0.0:
      raise ValueError("localization_outlier_position_std must be non-negative.")
    resampling_matches_duration = np.isclose(
      self.resampling_time_range[0], self.episode_duration
    ) and np.isclose(
      self.resampling_time_range[1], self.episode_duration
    )
    if not resampling_matches_duration:
      raise ValueError(
        "resampling_time_range must equal motion_duration + stop_hold_duration."
      )
