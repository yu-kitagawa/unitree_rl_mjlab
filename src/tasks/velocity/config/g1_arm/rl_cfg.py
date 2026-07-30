"""RL configuration for the G1 arm-pose velocity task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_arm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the PPO runner configuration with a separate experiment name."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_arm_velocity"
  return cfg
