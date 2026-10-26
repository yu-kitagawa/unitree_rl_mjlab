from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _hold_unstable_mask(
  command: MotionCommand,
  tilt_threshold: float,
  angular_velocity_threshold: float,
) -> torch.Tensor:
  """Identify terminal holds that need a balance recovery step."""
  tilt = torch.linalg.vector_norm(command.robot.data.projected_gravity_b[:, :2], dim=-1)
  angular_velocity = torch.linalg.vector_norm(
    command.robot.data.root_link_ang_vel_b[:, :2], dim=-1
  )
  return command.is_holding & (
    (tilt > tilt_threshold) | (angular_velocity > angular_velocity_threshold)
  )


def _hold_tracking_scale(
  command: MotionCommand,
  hold_unstable_scale: float,
  tilt_threshold: float,
  angular_velocity_threshold: float,
) -> torch.Tensor:
  unstable = _hold_unstable_mask(command, tilt_threshold, angular_velocity_threshold)
  return torch.where(
    unstable,
    torch.full_like(unstable, hold_unstable_scale, dtype=torch.float32),
    torch.ones_like(unstable, dtype=torch.float32),
  )


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  hold_unstable_scale: float = 1.0,
  tilt_threshold: float = 0.18,
  angular_velocity_threshold: float = 0.8,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  reward = torch.exp(-error / std**2)
  return reward * _hold_tracking_scale(
    command,
    hold_unstable_scale,
    tilt_threshold,
    angular_velocity_threshold,
  )


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  hold_unstable_scale: float = 1.0,
  tilt_threshold: float = 0.18,
  angular_velocity_threshold: float = 0.8,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  reward = torch.exp(-error.mean(-1) / std**2)
  return reward * _hold_tracking_scale(
    command,
    hold_unstable_scale,
    tilt_threshold,
    angular_velocity_threshold,
  )


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  hold_unstable_scale: float = 1.0,
  tilt_threshold: float = 0.18,
  angular_velocity_threshold: float = 0.8,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  reward = torch.exp(-error.mean(-1) / std**2)
  return reward * _hold_tracking_scale(
    command,
    hold_unstable_scale,
    tilt_threshold,
    angular_velocity_threshold,
  )


def hold_stability_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  tilt_std: float = 0.15,
  angular_velocity_std: float = 0.6,
  linear_velocity_std: float = 0.25,
) -> torch.Tensor:
  """Reward a quiet upright base while the final motion pose is held."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset = command.robot
  tilt_error = (
    torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1) / tilt_std**2
  )
  angular_error = (
    torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=-1)
    / angular_velocity_std**2
  )
  linear_error = (
    torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=-1)
    / linear_velocity_std**2
  )
  stability = (
    1.0 / (1.0 + tilt_error) + 1.0 / (1.0 + angular_error) + 1.0 / (1.0 + linear_error)
  ) / 3.0
  return command.is_holding.float() * stability


def hold_joint_pose_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Keep selected joints, normally the arms, on the final reference pose."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]
  current = asset.data.joint_pos[:, asset_cfg.joint_ids]
  target = command.joint_pos[:, asset_cfg.joint_ids]
  error = torch.mean(torch.square((current - target) / std), dim=-1)
  return command.is_holding.float() * torch.exp(-error)


def hold_joint_velocity_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  unstable_scale: float = 0.05,
  tilt_threshold: float = 0.18,
  angular_velocity_threshold: float = 0.8,
) -> torch.Tensor:
  """Penalize lower-body motion while stable but permit recovery stepping."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]
  penalty = torch.sum(
    torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1
  )
  unstable = _hold_unstable_mask(command, tilt_threshold, angular_velocity_threshold)
  scale = torch.where(
    unstable,
    torch.full_like(penalty, unstable_scale),
    torch.ones_like(penalty),
  )
  return command.is_holding.float() * scale * penalty


def stability_gated_action_rate_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  unstable_scale: float = 0.1,
  tilt_threshold: float = 0.18,
  angular_velocity_threshold: float = 0.8,
) -> torch.Tensor:
  """Relax action smoothing only while a held pose needs fast recovery."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  penalty = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1
  )
  unstable = _hold_unstable_mask(command, tilt_threshold, angular_velocity_threshold)
  return torch.where(
    unstable,
    unstable_scale * penalty,
    penalty,
  )


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def feet_clearance(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = cast(
    MotionCommand,
    env.command_manager.get_term(command_name),
  )

  foot_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  body_indexes = _get_body_indexes(
    command,
    foot_names,
  )

  foot_vel = command.robot_body_lin_vel_w[:, body_indexes, :2]

  foot_height = command.robot_body_pos_w[:, body_indexes, 2]
  horizontal_speed = torch.norm(foot_vel, dim=-1)

  foot_max = command.motion_body_max_height_w[:, body_indexes].amax(dim=1)
  selected_foot_max = foot_max[command.motion_ids]
  selected_foot_max = selected_foot_max[:, None]

  swing_mask = horizontal_speed > 0.15

  error = (foot_height - selected_foot_max) ** 2

  reward = torch.exp(-error / (std**2))

  reward *= swing_mask.float()

  return reward.mean(dim=1)


def foot_slip_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  hold_unstable_scale: float = 1.0,
  tilt_threshold: float = 0.18,
  angular_velocity_threshold: float = 0.8,
) -> torch.Tensor:
  command = cast(
    MotionCommand,
    env.command_manager.get_term(command_name),
  )

  foot_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  body_indexes = _get_body_indexes(
    command,
    foot_names,
  )

  foot_vel = command.robot_body_lin_vel_w[:, body_indexes, :2]

  slip = torch.norm(
    foot_vel,
    dim=-1,
  )

  contact = command.foot_contact.float()

  penalty = (slip * contact).mean(-1)
  return penalty * _hold_tracking_scale(
    command,
    hold_unstable_scale,
    tilt_threshold,
    angular_velocity_threshold,
  )


def foot_contact_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = cast(
    MotionCommand,
    env.command_manager.get_term(command_name),
  )

  sensor = env.scene.sensors["feet_ground_contact"]

  actual_contact = sensor.data.found > 0

  ref_contact = command.foot_contact > 0

  match = (actual_contact == ref_contact).float()

  return match.mean(dim=-1)


def soft_landing(
  env: ManagerBasedRlEnv,
  command_name: str | None = None,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene["feet_ground_contact"]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  return torch.sum(landing_impact, dim=1)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
