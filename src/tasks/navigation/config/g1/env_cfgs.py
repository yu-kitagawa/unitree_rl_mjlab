"""Unitree G1 29-DoF trajectory navigation configurations."""

from src import SRC_PATH
from src.assets.robots import (
  G1_ACTION_SCALE,
  G1_ARM_JOINT_NAMES,
  G1_JOINT_NAMES,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from src.tasks.navigation import mdp
from src.tasks.navigation.mdp import (
  JointPoseLibraryCommandCfg,
  TrajectoryCommandCfg,
)
from src.tasks.navigation.velocity_env_cfg import make_navigation_env_cfg


_G1_NON_ARM_JOINT_NAMES = tuple(
  name for name in G1_JOINT_NAMES if name not in G1_ARM_JOINT_NAMES
)

_ARM_MOTION_DIR = SRC_PATH / "assets" / "motions" / "g1" / "arm_vel"
_ARM_POSE_LIBRARY = _ARM_MOTION_DIR / "carry_poses.npz"

# The generic G1 wrist-pitch scale is sized for small locomotion corrections.
# The arm-up reference changes both wrist-pitch joints by about 0.68 rad, which
# would require policy actions near -9 with that scale. Besides being far
# outside the usual action range, the first transition alone is then dominated
# by the action-rate penalty. Give wrist pitch the same command range as the
# other arm joints for this conditioned task only.
_G1_NAVIGATION_ACTION_SCALE = dict(G1_ACTION_SCALE)
_G1_NAVIGATION_ACTION_SCALE[r".*_wrist_pitch_joint"] = G1_ACTION_SCALE[
  r".*_wrist_roll_joint"
]


def _add_episode_arm_pose_task(cfg: ManagerBasedRlEnvCfg) -> None:
  """Condition navigation on down and collision-checked carrying poses."""
  cfg.commands["arm_pose"] = JointPoseLibraryCommandCfg(
    entity_name="robot",
    joint_names=G1_ARM_JOINT_NAMES,
    pose_file=str(_ARM_POSE_LIBRARY),
    # Keep the sampled pose fixed for the whole episode.
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
    joint_names=G1_ARM_JOINT_NAMES,
    preserve_order=True,
  )
  cfg.rewards["arm_pose"] = RewardTermCfg(
    func=mdp.track_joint_pose_exp,
    weight=2.0,
    params={
      # At std=0.15 the arm-up reward starts near zero and its gradient is
      # easily overwhelmed by locomotion/path rewards. This width keeps a
      # useful gradient across the full down-to-carrying target range.
      "std": 0.35,
      "command_name": "arm_pose",
      "asset_cfg": arm_asset_cfg,
    },
  )

  # The generic posture rewards target the default arm pose, so limit them to
  # the lower body and waist and let the arm command own the arm joints.
  # SceneEntityCfg is resolved in place by each manager term. Do not share one
  # instance between rewards: resolving it twice makes mjlab validate the
  # original tuple of names against its already-resolved list of joint IDs.
  for reward_name in ("pose", "stand_still"):
    cfg.rewards[reward_name].params["asset_cfg"] = SceneEntityCfg(
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


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough-terrain navigation configuration."""
  cfg = make_navigation_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = _G1_NAVIGATION_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, TrajectoryCommandCfg)

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  _add_episode_arm_pose_task(cfg)

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    future_path_obs = cfg.observations["actor"].terms["future_path_poses"]
    future_path_obs.delay_min_lag = 0
    future_path_obs.delay_max_lag = 0
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat-terrain navigation configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
