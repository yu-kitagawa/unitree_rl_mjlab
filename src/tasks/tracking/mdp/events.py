from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class apply_motion_body_load:
  """Apply a downward payload force when selected motion bodies are raised."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._env = env
    self._asset: Entity = env.scene[asset_cfg.name]
    self._asset_cfg = asset_cfg
    self._body_ids = asset_cfg.body_ids
    self._body_names = tuple(asset_cfg.body_names or ())
    self._num_envs = env.num_envs
    self._device = env.device
    self._command_body_ids: torch.Tensor | None = None
    self._command_name = cfg.params["command_name"]
    force_magnitude = cfg.params["force_magnitude"]
    if isinstance(force_magnitude, (int, float)):
      self._force_magnitude_range = (float(force_magnitude),) * 2
    elif isinstance(force_magnitude, tuple) and len(force_magnitude) == 2:
      self._force_magnitude_range = tuple(map(float, force_magnitude))
    else:
      raise TypeError("force_magnitude must be a float or a (min, max) tuple.")
    if self._force_magnitude_range[0] < 0.0:
      raise ValueError("force_magnitude must be non-negative.")
    if self._force_magnitude_range[0] > self._force_magnitude_range[1]:
      raise ValueError("force_magnitude min must not exceed max.")

    self._force_magnitudes = torch.empty(
      (self._num_envs, 1), device=self._device, dtype=torch.float32
    )
    self._randomize_force_magnitudes()
    self._height_threshold = cfg.params["height_threshold"]
    self._transition_width = cfg.params.get("transition_width", 0.0)

    if isinstance(self._body_ids, list):
      self._num_bodies = len(self._body_ids)
    else:
      self._num_bodies = self._asset.num_bodies

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    force_magnitude: float | tuple[float, float],
    height_threshold: float,
    transition_width: float = 0.0,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    del asset_cfg, force_magnitude  # Resolved and cached at initialization.
    self._write_load(
      env=env,
      env_ids=env_ids,
      command_name=command_name,
      height_threshold=height_threshold,
      transition_width=transition_width,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._randomize_force_magnitudes(env_ids)
    if not hasattr(self._env, "command_manager"):
      self._clear_load(env_ids)
      return
    self._write_load(
      env=self._env,
      env_ids=env_ids,
      command_name=self._command_name,
      height_threshold=self._height_threshold,
      transition_width=self._transition_width,
    )

  def _write_load(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    height_threshold: float,
    transition_width: float,
  ) -> None:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    command_body_ids = self._resolve_command_body_ids(command)

    body_heights = (
      command.body_pos_w[:, command_body_ids, 2] - command.anchor_pos_w[:, None, 2]
    )
    if transition_width > 0.0:
      active_scale = torch.clamp(
        (body_heights - height_threshold) / transition_width,
        min=0.0,
        max=1.0,
      )
    else:
      active_scale = (body_heights >= height_threshold).float()

    forces = torch.zeros(
      (self._num_envs, self._num_bodies, 3),
      device=self._device,
      dtype=torch.float32,
    )
    torques = torch.zeros_like(forces)
    forces[..., 2] = -self._force_magnitudes * active_scale

    if env_ids is not None:
      forces = forces[env_ids]
      torques = torques[env_ids]

    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      env_ids=env_ids,
      body_ids=self._body_ids,
    )

  def _randomize_force_magnitudes(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    lower, upper = self._force_magnitude_range
    if env_ids is None:
      self._force_magnitudes.uniform_(lower, upper)
      return
    samples = torch.empty_like(self._force_magnitudes[env_ids]).uniform_(lower, upper)
    self._force_magnitudes[env_ids] = samples

  def _clear_load(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      num_envs = self._num_envs
    else:
      num_envs = len(env_ids)
    zeros = torch.zeros(
      (num_envs, self._num_bodies, 3),
      device=self._device,
      dtype=torch.float32,
    )
    self._asset.write_external_wrench_to_sim(
      zeros,
      zeros,
      env_ids=env_ids,
      body_ids=self._body_ids,
    )

  def _resolve_command_body_ids(self, command: MotionCommand) -> torch.Tensor:
    if self._command_body_ids is not None:
      return self._command_body_ids
    if not self._body_names:
      raise ValueError(
        "apply_motion_body_load requires asset_cfg.body_names so the selected "
        "bodies can be matched to the motion command."
      )

    missing = [name for name in self._body_names if name not in command.cfg.body_names]
    if missing:
      raise ValueError(
        "Payload body names must also exist in the motion command body_names. "
        f"Missing: {missing}"
      )

    self._command_body_ids = torch.tensor(
      [command.cfg.body_names.index(name) for name in self._body_names],
      device=self._device,
      dtype=torch.long,
    )
    return self._command_body_ids
