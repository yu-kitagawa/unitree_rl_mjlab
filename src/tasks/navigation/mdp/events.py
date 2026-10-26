from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .joint_pose_command import JointPoseLibraryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class apply_joint_pose_body_load:
  """Apply independently randomized hand loads for raised carry poses."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._env = env
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    if not isinstance(self._body_ids, list) or not self._body_ids:
      raise ValueError("apply_joint_pose_body_load requires selected body_names.")
    self._num_envs = env.num_envs
    self._num_bodies = len(self._body_ids)
    self._device = env.device
    self._command_name = cfg.params["command_name"]

    force_magnitude = cfg.params["force_magnitude"]
    if isinstance(force_magnitude, (int, float)):
      self._force_magnitude_range = (float(force_magnitude),) * 2
    elif isinstance(force_magnitude, tuple) and len(force_magnitude) == 2:
      self._force_magnitude_range = tuple(map(float, force_magnitude))
    else:
      raise TypeError("force_magnitude must be a float or a (min, max) tuple.")
    lower, upper = self._force_magnitude_range
    if lower < 0.0 or lower > upper:
      raise ValueError("force_magnitude must satisfy 0 <= min <= max.")

    self._force_magnitudes = torch.empty(
      (self._num_envs, self._num_bodies),
      device=self._device,
      dtype=torch.float32,
    )
    self._randomize_force_magnitudes()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    force_magnitude: float | tuple[float, float],
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    del asset_cfg, force_magnitude
    command = cast(JointPoseLibraryCommand, env.command_manager.get_term(command_name))
    active = command.is_raised.float().unsqueeze(-1)
    forces = torch.zeros(
      (self._num_envs, self._num_bodies, 3),
      device=self._device,
      dtype=torch.float32,
    )
    forces[..., 2] = -self._force_magnitudes * active
    torques = torch.zeros_like(forces)
    if env_ids is not None:
      forces = forces[env_ids]
      torques = torques[env_ids]
    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      env_ids=env_ids,
      body_ids=self._body_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._randomize_force_magnitudes(env_ids)

  def _randomize_force_magnitudes(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    lower, upper = self._force_magnitude_range
    if env_ids is None:
      self._force_magnitudes.uniform_(lower, upper)
      return
    values = torch.empty_like(self._force_magnitudes[env_ids]).uniform_(lower, upper)
    self._force_magnitudes[env_ids] = values
