from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class MotionDerivedJointPoseCommand(CommandTerm):
  """Sample a fixed joint pose extracted from a motion window.

  One pose is sampled for each environment when the command manager is reset.
  A robust median over the configured frame window removes the gait oscillation
  while retaining the representative pose from the reference motion.
  """

  cfg: MotionDerivedJointPoseCommandCfg

  def __init__(
    self,
    cfg: MotionDerivedJointPoseCommandCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.joint_ids, resolved_joint_names = self.robot.find_joints(
      cfg.joint_names, preserve_order=True
    )
    if tuple(resolved_joint_names) != cfg.joint_names:
      raise ValueError(
        "The controlled joints must resolve in the same order as `joint_names`."
      )

    self.pose_targets = self._load_pose_targets()
    self.pose_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.pose_command = torch.zeros(
      self.num_envs,
      len(cfg.joint_names),
      dtype=torch.float32,
      device=self.device,
    )

    if cfg.sampling_weights is None:
      self.sampling_weights = torch.ones(
        len(cfg.motion_files), dtype=torch.float32, device=self.device
      )
    else:
      if len(cfg.sampling_weights) != len(cfg.motion_files):
        raise ValueError(
          "`sampling_weights` must contain one value for each motion file."
        )
      self.sampling_weights = torch.tensor(
        cfg.sampling_weights, dtype=torch.float32, device=self.device
      )
      if torch.any(self.sampling_weights < 0) or not torch.any(
        self.sampling_weights > 0
      ):
        raise ValueError("`sampling_weights` must be non-negative and not all zero.")

    self.metrics["mean_joint_pose_error"] = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.pose_command

  def _load_pose_targets(self) -> torch.Tensor:
    if len(self.cfg.motion_files) != len(self.cfg.motion_frame_ranges):
      raise ValueError(
        "`motion_frame_ranges` must contain one range for each motion file."
      )
    if len(self.cfg.motion_files) == 0:
      raise ValueError("At least one motion file is required.")

    try:
      motion_joint_ids = [
        self.cfg.motion_joint_names.index(name) for name in self.cfg.joint_names
      ]
    except ValueError as exc:
      raise ValueError(
        "Every controlled joint must be present in `motion_joint_names`."
      ) from exc

    pose_targets: list[np.ndarray] = []
    for motion_file, (start, stop) in zip(
      self.cfg.motion_files,
      self.cfg.motion_frame_ranges,
      strict=True,
    ):
      path = Path(motion_file)
      if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")

      with np.load(path, allow_pickle=False) as data:
        if "joint_pos" not in data:
          raise KeyError(f"Motion file has no `joint_pos` array: {path}")
        joint_pos = data["joint_pos"]

      if joint_pos.ndim != 2:
        raise ValueError(
          f"`joint_pos` in {path} must be two-dimensional, got {joint_pos.shape}."
        )
      if joint_pos.shape[1] != len(self.cfg.motion_joint_names):
        raise ValueError(
          f"`joint_pos` in {path} has {joint_pos.shape[1]} joints, expected "
          f"{len(self.cfg.motion_joint_names)}."
        )
      if not 0 <= start < stop <= joint_pos.shape[0]:
        raise ValueError(
          f"Invalid frame range [{start}, {stop}) for {path} with "
          f"{joint_pos.shape[0]} frames."
        )

      motion_window = joint_pos[start:stop, motion_joint_ids]
      pose_targets.append(np.median(motion_window, axis=0))

    return torch.as_tensor(
      np.stack(pose_targets),
      dtype=torch.float32,
      device=self.device,
    )

  def _update_metrics(self) -> None:
    joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
    mean_error = torch.mean(torch.abs(joint_pos - self.pose_command), dim=1)
    self.metrics["mean_joint_pose_error"] += mean_error / self._env.max_episode_length

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pose_ids = torch.multinomial(
      self.sampling_weights,
      num_samples=len(env_ids),
      replacement=True,
    )
    self.pose_ids[env_ids] = pose_ids
    self.pose_command[env_ids] = self.pose_targets[pose_ids]

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class MotionDerivedJointPoseCommandCfg(CommandTermCfg):
  entity_name: str
  joint_names: tuple[str, ...]
  motion_joint_names: tuple[str, ...]
  motion_files: tuple[str, ...]
  motion_frame_ranges: tuple[tuple[int, int], ...]
  sampling_weights: tuple[float, ...] | None = None

  def build(self, env: ManagerBasedRlEnv) -> MotionDerivedJointPoseCommand:
    return MotionDerivedJointPoseCommand(self, env)
