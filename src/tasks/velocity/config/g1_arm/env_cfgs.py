"""G1 velocity tasks conditioned on an episode-level arm pose."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.velocity.mdp as mdp
from src import SRC_PATH
from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from src.tasks.velocity.mdp import MotionDerivedJointPoseCommandCfg

_G1_MOTION_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

_G1_ARM_JOINT_NAMES = tuple(
  name
  for name in _G1_MOTION_JOINT_NAMES
  if any(part in name for part in ("shoulder", "elbow", "wrist"))
)

_G1_NON_ARM_JOINT_NAMES = tuple(
  name for name in _G1_MOTION_JOINT_NAMES if name not in _G1_ARM_JOINT_NAMES
)

_ARM_MOTION_DIR = SRC_PATH / "assets" / "motions" / "g1" / "arm_vel"


def _add_episode_arm_pose_task(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Add an arm-pose command sampled once per episode."""
  cfg.commands["arm_pose"] = MotionDerivedJointPoseCommandCfg(
    entity_name="robot",
    joint_names=_G1_ARM_JOINT_NAMES,
    motion_joint_names=_G1_MOTION_JOINT_NAMES,
    motion_files=(
      str(_ARM_MOTION_DIR / "arm_down.npz"),
      str(_ARM_MOTION_DIR / "arm_up.npz"),
    ),
    # arm_down walks in frames [0, 80). arm_up raises its arms before frame 150,
    # walks with them held horizontally in [170, 260), and lowers them after 350.
    motion_frame_ranges=((0, 80), (170, 260)),
    sampling_weights=(1.0, 1.0),
    # The command manager also samples on reset. Keep interval resampling outside
    # any practical episode so the selected arm pose remains episode-constant.
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=False,
  )

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["arm_pose"] = ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "arm_pose"},
    )

  arm_asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=_G1_ARM_JOINT_NAMES,
    preserve_order=True,
  )
  cfg.rewards["arm_pose"] = RewardTermCfg(
    func=mdp.track_joint_pose_exp,
    weight=1.0,
    params={
      "std": 0.15,
      "command_name": "arm_pose",
      "asset_cfg": arm_asset_cfg,
    },
  )

  # The base velocity posture terms target the robot's default arm pose. Restrict
  # them to the lower body and waist to avoid opposing the arm-pose command.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=_G1_NON_ARM_JOINT_NAMES,
    preserve_order=True,
  )
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=_G1_NON_ARM_JOINT_NAMES,
    preserve_order=True,
  )
  for std_name in ("std_walking", "std_running"):
    cfg.rewards["pose"].params[std_name] = {
      pattern: value
      for pattern, value in cfg.rewards["pose"].params[std_name].items()
      if not any(part in pattern for part in ("shoulder", "elbow", "wrist"))
    }

  return cfg


def unitree_g1_arm_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 rough-terrain velocity task with episode arm poses."""
  return _add_episode_arm_pose_task(unitree_g1_rough_env_cfg(play=play))


def unitree_g1_arm_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 flat-terrain velocity task with episode arm poses."""
  return _add_episode_arm_pose_task(unitree_g1_flat_env_cfg(play=play))
