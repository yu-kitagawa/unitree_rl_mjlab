"""List or interactively view poses in a G1 arm pose library."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import Entity
from src import SRC_PATH
from src.assets.robots import (
  G1_ARM_JOINT_NAMES,
  get_g1_robot_cfg,
)


DEFAULT_POSE_FILE = (
  SRC_PATH / "assets" / "motions" / "g1" / "arm_vel" / "carry_poses.npz"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pose-file", type=Path, default=DEFAULT_POSE_FILE)
  parser.add_argument(
    "--list",
    action="store_true",
    help="Print the library without opening the MuJoCo viewer.",
  )
  parser.add_argument(
    "--pose-index",
    type=int,
    help="Hold one pose. By default the viewer cycles through all poses.",
  )
  parser.add_argument(
    "--period-s",
    type=float,
    default=2.0,
    help="Seconds per pose when cycling.",
  )
  return parser.parse_args()


def _load_library(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  if not path.is_file():
    raise FileNotFoundError(f"Pose library not found: {path}")
  with np.load(path, allow_pickle=False) as data:
    required = {"joint_names", "pose_names", "poses"}
    missing = required - set(data.files)
    if missing:
      raise KeyError(f"Pose library is missing: {', '.join(sorted(missing))}.")
    joint_names = tuple(str(name) for name in data["joint_names"])
    pose_names = np.asarray(data["pose_names"])
    poses = np.asarray(data["poses"], dtype=np.float64)

  if joint_names != G1_ARM_JOINT_NAMES:
    raise ValueError(
      f"Pose joint order {joint_names} does not match {G1_ARM_JOINT_NAMES}."
    )
  if poses.shape != (len(pose_names), len(G1_ARM_JOINT_NAMES)):
    raise ValueError(
      f"Invalid pose shape {poses.shape}; expected "
      f"({len(pose_names)}, {len(G1_ARM_JOINT_NAMES)})."
    )
  return np.asarray(joint_names), pose_names, poses


def _print_pose(
  index: int,
  pose_name: str,
  pose: np.ndarray,
  joint_names: np.ndarray,
  *,
  include_joints: bool,
) -> None:
  print(f"[{index:02d}] {pose_name}")
  if include_joints:
    for joint_name, angle in zip(joint_names, pose, strict=True):
      print(f"  {str(joint_name):32s} {angle:+.4f} rad")


def _view(
  joint_names: np.ndarray,
  pose_names: np.ndarray,
  poses: np.ndarray,
  pose_index: int | None,
  period_s: float,
) -> None:
  if period_s <= 0.0:
    raise ValueError("`--period-s` must be positive.")
  if pose_index is not None and not 0 <= pose_index < len(poses):
    raise ValueError(
      f"`--pose-index` must be in [0, {len(poses) - 1}], got {pose_index}."
    )

  import mujoco.viewer

  robot = Entity(get_g1_robot_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)
  qpos_addresses = np.asarray(
    [
      model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
      ]
      for name in joint_names
    ],
    dtype=np.int32,
  )

  shown_index = -1
  start_time = time.monotonic()
  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = (0.10, 0.0, 0.80)
    viewer.cam.azimuth = 145.0
    viewer.cam.elevation = -12.0
    viewer.cam.distance = 2.2
    while viewer.is_running():
      if pose_index is None:
        next_index = int((time.monotonic() - start_time) / period_s) % len(poses)
      else:
        next_index = pose_index

      if next_index != shown_index:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[qpos_addresses] = poses[next_index]
        mujoco.mj_forward(model, data)
        _print_pose(
          next_index,
          str(pose_names[next_index]),
          poses[next_index],
          joint_names,
          include_joints=False,
        )
        shown_index = next_index

      viewer.sync()
      time.sleep(0.02)


def main() -> None:
  args = _parse_args()
  joint_names, pose_names, poses = _load_library(args.pose_file)
  if args.list:
    for index, (pose_name, pose) in enumerate(
      zip(pose_names, poses, strict=True)
    ):
      _print_pose(
        index,
        str(pose_name),
        pose,
        joint_names,
        include_joints=True,
      )
    return
  _view(joint_names, pose_names, poses, args.pose_index, args.period_s)


if __name__ == "__main__":
  main()
