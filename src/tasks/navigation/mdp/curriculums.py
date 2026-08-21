from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from .trajectory_command import TrajectoryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_nav(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
) -> torch.Tensor:
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = cast(
    TrajectoryCommand, env.command_manager.get_term(command_name)
  )
  position_error = command.distance_to_path[env_ids]
  heading_error = torch.abs(command.path_heading_error[env_ids])
  move_up = (position_error < 0.2) & (heading_error < 0.3)
  move_down = ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


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
