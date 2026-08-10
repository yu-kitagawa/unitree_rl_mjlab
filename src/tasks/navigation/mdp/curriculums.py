from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import GoalPoseCommand, GoalPoseCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class GoalRangeStage(TypedDict):
  step: int
  object_distance: tuple[float, float] | None
  object_bearing: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_nav(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = cast(
    GoalPoseCommand, env.command_manager.get_term(command_name)
  )
  position_error = torch.linalg.vector_norm(
    asset.data.root_link_pos_w[env_ids, :2] - command.target_pos_w[env_ids],
    dim=-1,
  )
  heading_error = torch.abs(
    torch.atan2(
      torch.sin(asset.data.heading_w[env_ids] - command.target_heading_w[env_ids]),
      torch.cos(asset.data.heading_w[env_ids] - command.target_heading_w[env_ids]),
    )
  )
  move_up = (position_error < 0.2) & (heading_error < 0.3)
  move_down = ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def goal_ranges(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  goal_stages: list[GoalRangeStage],
) -> dict[str, torch.Tensor]:
  """Expand object distance and bearing ranges as training progresses."""
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(GoalPoseCommandCfg, command_term.cfg)
  for stage in goal_stages:
    if env.common_step_counter > stage["step"]:
      if stage.get("object_distance") is not None:
        cfg.ranges.object_distance = stage["object_distance"]
      if stage.get("object_bearing") is not None:
        cfg.ranges.object_bearing = stage["object_bearing"]
  return {}


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
