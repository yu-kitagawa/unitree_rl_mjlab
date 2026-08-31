"""Generate collision-free G1 arm targets for the navigation task."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import Entity
from src import SRC_PATH
from src.assets.robots import (
  G1_ARM_JOINT_NAMES,
  G1_JOINT_NAMES,
  get_g1_robot_cfg,
)


DEFAULT_MOTION_DIR = SRC_PATH / "assets" / "motions" / "g1" / "arm_vel"
DEFAULT_OUTPUT = DEFAULT_MOTION_DIR / "carry_poses.npz"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Generate reproducible carrying poses and reject poses that violate G1 "
      "joint margins, carrying geometry, or self-collide."
    )
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=DEFAULT_OUTPUT,
    help=f"Output pose library (default: {DEFAULT_OUTPUT})",
  )
  parser.add_argument(
    "--arm-down-motion",
    type=Path,
    default=DEFAULT_MOTION_DIR / "arm_down.npz",
  )
  parser.add_argument(
    "--arm-up-motion",
    type=Path,
    default=DEFAULT_MOTION_DIR / "arm_up.npz",
  )
  parser.add_argument(
    "--num-raised-poses",
    type=int,
    default=32,
    help="Number of raised targets, including the original arm-up reference.",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--joint-limit-margin-rad",
    type=float,
    default=0.08,
  )
  parser.add_argument(
    "--transition-steps",
    type=int,
    default=9,
    help="Collision checks along interpolation from the original raised pose.",
  )
  parser.add_argument(
    "--minimum-rms-distance-rad",
    type=float,
    default=0.10,
    help="Minimum RMS joint-space separation between raised poses.",
  )
  parser.add_argument("--max-attempts", type=int, default=50_000)
  return parser.parse_args()


def _load_motion_pose(
  path: Path,
  frame_range: tuple[int, int],
) -> np.ndarray:
  if not path.is_file():
    raise FileNotFoundError(f"Motion file not found: {path}")
  with np.load(path, allow_pickle=False) as data:
    if "joint_pos" not in data:
      raise KeyError(f"Motion file has no `joint_pos` array: {path}")
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)

  start, stop = frame_range
  if joint_pos.ndim != 2 or joint_pos.shape[1] != len(G1_JOINT_NAMES):
    raise ValueError(
      f"`joint_pos` in {path} must have shape (N, {len(G1_JOINT_NAMES)}), "
      f"got {joint_pos.shape}."
    )
  if not 0 <= start < stop <= joint_pos.shape[0]:
    raise ValueError(
      f"Invalid frame range [{start}, {stop}) for {path} with "
      f"{joint_pos.shape[0]} frames."
    )

  motion_indices = [G1_JOINT_NAMES.index(name) for name in G1_ARM_JOINT_NAMES]
  return np.median(joint_pos[start:stop, motion_indices], axis=0)


class PoseValidator:
  """Validate joint margins, carrying geometry, and MuJoCo self-collisions."""

  def __init__(
    self,
    reference_pose: np.ndarray,
    joint_limit_margin_rad: float,
    transition_steps: int,
  ):
    robot = Entity(get_g1_robot_cfg())
    self.model = robot.spec.compile()
    self.data = mujoco.MjData(self.model)
    self.reference_pose = reference_pose
    self.joint_limit_margin_rad = joint_limit_margin_rad
    self.transition_steps = transition_steps

    self.qpos_addresses = np.asarray(
      [
        self.model.jnt_qposadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in G1_ARM_JOINT_NAMES
      ],
      dtype=np.int32,
    )
    self.joint_ranges = np.asarray(
      [
        self.model.jnt_range[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in G1_ARM_JOINT_NAMES
      ]
    )
    self.body_ids = {
      name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
      for name in (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
      )
    }

  def reset_to_pose(self, pose: np.ndarray) -> None:
    mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
    self.data.qpos[self.qpos_addresses] = pose
    mujoco.mj_forward(self.model, self.data)

  def within_joint_margins(self, pose: np.ndarray) -> bool:
    low = self.joint_ranges[:, 0] + self.joint_limit_margin_rad
    high = self.joint_ranges[:, 1] - self.joint_limit_margin_rad
    return bool(np.all((pose >= low) & (pose <= high)))

  def _has_collision_free_transition(self, pose: np.ndarray) -> bool:
    for alpha in np.linspace(0.0, 1.0, self.transition_steps):
      interpolated = (1.0 - alpha) * self.reference_pose + alpha * pose
      self.reset_to_pose(interpolated)
      if self.data.ncon != 0:
        return False
    return True

  def _has_carrying_geometry(self, pose: np.ndarray) -> bool:
    self.reset_to_pose(pose)
    left = self.data.xpos[self.body_ids["left_wrist_yaw_link"]]
    right = self.data.xpos[self.body_ids["right_wrist_yaw_link"]]
    torso = self.data.xpos[self.body_ids["torso_link"]]
    midpoint = 0.5 * (left + right)

    return bool(
      0.08 <= midpoint[0] - torso[0] <= 0.42
      and -0.15 <= midpoint[2] - torso[2] <= 0.28
      and 0.20 <= left[1] - right[1] <= 0.62
      and abs(left[0] - right[0]) <= 0.12
      and abs(left[2] - right[2]) <= 0.10
      and left[1] > torso[1]
      and right[1] < torso[1]
    )

  def rejection_reason(self, pose: np.ndarray) -> str | None:
    if not np.all(np.isfinite(pose)):
      return "non_finite"
    if not self.within_joint_margins(pose):
      return "joint_limit"
    if not self._has_carrying_geometry(pose):
      return "carrying_geometry"
    if not self._has_collision_free_transition(pose):
      return "collision"
    return None


def _sample_bilateral_pose(rng: np.random.Generator) -> np.ndarray:
  """Sample a near-symmetric pose suitable for holding an object in front."""
  common = np.asarray(
    [
      rng.uniform(-0.30, 0.60),  # shoulder pitch
      rng.uniform(0.03, 0.42),  # outward shoulder roll
      rng.uniform(-0.35, 0.35),  # shoulder yaw
      rng.uniform(0.30, 1.15),  # elbow
      rng.uniform(-0.35, 0.35),  # wrist roll
      rng.uniform(-1.05, -0.25),  # wrist pitch
      rng.uniform(-0.12, 0.12),  # wrist yaw
    ]
  )
  jitter = rng.normal(
    loc=0.0,
    scale=np.asarray([0.05, 0.035, 0.05, 0.06, 0.05, 0.05, 0.025]),
    size=(2, 7),
  )

  left = common + jitter[0]
  right = common * np.asarray([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
  right += jitter[1]
  return np.concatenate((left, right))


def _is_diverse(
  candidate: np.ndarray,
  accepted: list[np.ndarray],
  minimum_rms_distance_rad: float,
) -> bool:
  distances = [np.sqrt(np.mean(np.square(candidate - pose))) for pose in accepted]
  return bool(min(distances) >= minimum_rms_distance_rad)


def generate_pose_library(args: argparse.Namespace) -> tuple[int, Counter[str]]:
  if args.num_raised_poses < 1:
    raise ValueError("`--num-raised-poses` must be at least one.")
  if args.transition_steps < 2:
    raise ValueError("`--transition-steps` must be at least two.")
  if args.joint_limit_margin_rad < 0.0:
    raise ValueError("`--joint-limit-margin-rad` must be non-negative.")
  if args.minimum_rms_distance_rad < 0.0:
    raise ValueError("`--minimum-rms-distance-rad` must be non-negative.")

  arm_down = _load_motion_pose(args.arm_down_motion, (0, 80))
  arm_up = _load_motion_pose(args.arm_up_motion, (170, 260))
  validator = PoseValidator(
    reference_pose=arm_up,
    joint_limit_margin_rad=args.joint_limit_margin_rad,
    transition_steps=args.transition_steps,
  )
  if not validator.within_joint_margins(arm_down):
    raise ValueError("The original arm-down pose violates the joint limit margin.")
  reference_rejection = validator.rejection_reason(arm_up)
  if reference_rejection is not None:
    raise ValueError(
      f"The original arm-up reference was rejected: {reference_rejection}."
    )

  rng = np.random.default_rng(args.seed)
  raised_poses = [arm_up]
  rejections: Counter[str] = Counter()
  attempts = 0
  while len(raised_poses) < args.num_raised_poses and attempts < args.max_attempts:
    attempts += 1
    candidate = _sample_bilateral_pose(rng)
    reason = validator.rejection_reason(candidate)
    if reason is not None:
      rejections[reason] += 1
      continue
    if not _is_diverse(
      candidate,
      raised_poses,
      args.minimum_rms_distance_rad,
    ):
      rejections["too_similar"] += 1
      continue
    raised_poses.append(candidate)

  if len(raised_poses) != args.num_raised_poses:
    raise RuntimeError(
      f"Generated only {len(raised_poses)}/{args.num_raised_poses} raised poses "
      f"after {attempts} attempts. Rejections: {dict(rejections)}"
    )

  poses = np.asarray([arm_down, *raised_poses], dtype=np.float32)
  pose_names = np.asarray(
    ["arm_down", "carry_reference"]
    + [f"carry_random_{index:03d}" for index in range(1, len(raised_poses))]
  )
  # Give down and raised conditions equal total probability. The raised half is
  # divided uniformly among all generated carrying poses.
  sampling_weights = np.asarray(
    [1.0] + [1.0 / len(raised_poses)] * len(raised_poses),
    dtype=np.float32,
  )
  is_raised = np.asarray([False] + [True] * len(raised_poses))

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    joint_names=np.asarray(G1_ARM_JOINT_NAMES),
    pose_names=pose_names,
    poses=poses,
    sampling_weights=sampling_weights,
    is_raised=is_raised,
    seed=np.asarray(args.seed, dtype=np.int64),
    joint_limit_margin_rad=np.asarray(
      args.joint_limit_margin_rad, dtype=np.float32
    ),
    transition_steps=np.asarray(args.transition_steps, dtype=np.int64),
    minimum_rms_distance_rad=np.asarray(
      args.minimum_rms_distance_rad, dtype=np.float32
    ),
  )
  return attempts, rejections


def main() -> None:
  args = _parse_args()
  attempts, rejections = generate_pose_library(args)
  print(f"Saved: {args.output}")
  print(f"Poses: 1 down + {args.num_raised_poses} raised")
  print(f"Accepted random samples: {args.num_raised_poses - 1}/{attempts}")
  print(f"Rejections: {dict(rejections)}")


if __name__ == "__main__":
  main()
