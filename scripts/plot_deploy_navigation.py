#!/usr/bin/env python3
"""Live plots for the G1 deploy Navigation CSV log."""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import deque
from pathlib import Path


G1_JOINT_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
G1_ARM_JOINT_INDICES = tuple(range(15, 29))

SCALAR_FIELDS = (
    "time_s",
    "localization_available",
    "slam_pose_x_m",
    "slam_pose_y_m",
    "slam_pose_theta_rad",
    "goal_x_m",
    "goal_y_m",
    "goal_yaw_rad",
    "command_rel_pos_x_m",
    "command_rel_pos_y_m",
    "command_rel_yaw_rad",
    "position_error_m",
    "closest_progress_m",
    "goal_reached",
)
JOINT_PREFIXES = (
    "command_joint_",
    "encoder_joint_",
    "action_joint_",
)
REQUIRED_FIELDS = set(SCALAR_FIELDS) | {"source"}


class NavigationLogTail:
    """Incrementally consume a CSV file that is truncated on Navigation entry."""

    def __init__(self, path: Path, max_points: int):
        self.path = path
        self.max_points = max_points
        self.offset = 0
        self.inode: int | None = None
        self.pending = ""
        self.fieldnames: list[str] | None = None
        self.numeric_fields: list[str] = []
        self.joint_fields: dict[str, list[str]] = {}
        self.values: dict[str, deque[float] | deque[str]] = {}
        self.navigation_origin: tuple[float, float, float] | None = None
        self.error = "Waiting for Navigation CSV log"
        self._clear_values()

    def _clear_values(self) -> None:
        self.values = {
            field: deque(maxlen=self.max_points) for field in self.numeric_fields
        }
        self.values["source"] = deque(maxlen=self.max_points)

    def _reset(self) -> None:
        self.offset = 0
        self.pending = ""
        self.fieldnames = None
        self.numeric_fields = []
        self.joint_fields = {}
        self.navigation_origin = None
        self._clear_values()

    @staticmethod
    def _joint_sort_key(field: str) -> int:
        try:
            return int(field.rsplit("_", 1)[-1])
        except ValueError:
            return 1_000_000

    def _configure_header(self, fieldnames: list[str]) -> bool:
        missing = REQUIRED_FIELDS - set(fieldnames)
        if missing:
            self.error = "CSV is missing fields: " + ", ".join(sorted(missing))
            return False

        self.joint_fields = {
            prefix: sorted(
                (field for field in fieldnames if field.startswith(prefix)),
                key=self._joint_sort_key,
            )
            for prefix in JOINT_PREFIXES
        }
        missing_groups = [
            prefix.rstrip("_")
            for prefix, fields in self.joint_fields.items()
            if prefix != "command_joint_" and not fields
        ]
        if missing_groups:
            self.error = "CSV is missing joint groups: " + ", ".join(missing_groups)
            return False

        self.fieldnames = fieldnames
        self.numeric_fields = [field for field in fieldnames if field != "source"]
        self._clear_values()
        return True

    def poll(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.error = f"Waiting for {self.path}"
            return

        if self.inode != stat.st_ino or stat.st_size < self.offset:
            self._reset()
            self.inode = stat.st_ino

        try:
            with self.path.open("r", encoding="utf-8", newline="") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except OSError as exc:
            self.error = f"Cannot read Navigation CSV: {exc}"
            return

        if not chunk:
            return

        text = self.pending + chunk
        if text.endswith("\n"):
            complete_text = text
            self.pending = ""
        else:
            lines = text.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                self.pending = lines.pop()
            complete_text = "".join(lines)

        if not complete_text:
            return

        rows = list(csv.reader(io.StringIO(complete_text)))
        if self.fieldnames is None:
            if not rows:
                return
            if not self._configure_header(rows.pop(0)):
                # Retry from byte zero so an old-schema file can be replaced
                # in place (same inode) when Navigation is entered again.
                self.offset = 0
                self.pending = ""
                return

        assert self.fieldnames is not None
        for row_values in rows:
            if len(row_values) != len(self.fieldnames):
                continue
            row = dict(zip(self.fieldnames, row_values))
            try:
                parsed = {
                    field: float(row[field]) for field in self.numeric_fields
                }
            except (KeyError, ValueError):
                continue
            if (
                self.values["time_s"]
                and parsed["time_s"] < self.values["time_s"][-1]
            ):
                # The writer truncated and rewrote the file between polls,
                # but its new size already passed the previous byte offset.
                self._reset()
                self.inode = stat.st_ino
                self.poll()
                return
            for field, value in parsed.items():
                self.values[field].append(value)
            self.values["source"].append(row["source"])
            if self.navigation_origin is None and bool(
                parsed["localization_available"]
            ):
                candidate_origin = (
                    parsed["slam_pose_x_m"],
                    parsed["slam_pose_y_m"],
                    parsed["slam_pose_theta_rad"],
                )
                if all(math.isfinite(value) for value in candidate_origin):
                    self.navigation_origin = candidate_origin

        if self.values["time_s"]:
            self.error = ""


class TargetPathLog:
    """Reload the generated target path whenever Navigation rewrites it."""

    REQUIRED_FIELDS = ("time_s", "x_m", "y_m", "yaw_rad", "is_goal")

    def __init__(self, path: Path):
        self.path = path
        self.signature: tuple[int, int, int] | None = None
        self.values: dict[str, list[float]] = {
            field: [] for field in self.REQUIRED_FIELDS
        }
        self.error = f"Waiting for {path}"

    def poll(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.error = f"Waiting for {self.path}"
            return
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature == self.signature:
            return
        try:
            with self.path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or not set(self.REQUIRED_FIELDS).issubset(
                    reader.fieldnames
                ):
                    self.error = "Target path CSV has an incompatible header"
                    return
                values = {field: [] for field in self.REQUIRED_FIELDS}
                for row in reader:
                    for field in self.REQUIRED_FIELDS:
                        values[field].append(float(row[field]))
        except (OSError, ValueError) as exc:
            self.error = f"Cannot read Navigation target path CSV: {exc}"
            return
        self.values = values
        self.signature = signature
        self.error = "" if values["time_s"] else "Target path CSV is empty"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_log = repo_root / "deploy/robots/g1/log/navigation_pose.csv"
    default_target_path = (
        repo_root / "deploy/robots/g1/log/navigation_target_path.csv"
    )
    parser = argparse.ArgumentParser(
        description="Live plots of G1 Navigation commands, joints, and pose."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=default_log,
        help=f"CSV emitted by g1_ctrl (default: {default_log})",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=200,
        help="Plot refresh interval in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--target-path-file",
        type=Path,
        default=default_target_path,
        help=f"Generated target path CSV (default: {default_target_path})",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10000,
        help="Maximum displayed samples (default: 10000)",
    )
    parser.add_argument(
        "--joints",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help=(
            "Encoder/action joint indices to plot (default: all 29); "
            "command shows only matching arm joints"
        ),
    )
    args = parser.parse_args()
    if args.interval_ms <= 0 or args.max_points <= 1:
        parser.error("--interval-ms must be positive and --max-points must exceed 1")
    if args.joints is not None:
        invalid = [index for index in args.joints if not 0 <= index < len(G1_JOINT_NAMES)]
        if invalid:
            parser.error(f"joint indices must be in [0, {len(G1_JOINT_NAMES) - 1}]")
        args.joints = list(dict.fromkeys(args.joints))
    return args


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:
        raise SystemExit(
            "matplotlib and numpy are required: python3 -m pip install matplotlib numpy"
        ) from exc

    measured_joint_indices = args.joints or list(range(len(G1_JOINT_NAMES)))
    command_joint_indices = [
        index for index in measured_joint_indices if index in G1_ARM_JOINT_INDICES
    ]
    tail = NavigationLogTail(args.log_file.expanduser().resolve(), args.max_points)
    target_path = TargetPathLog(args.target_path_file.expanduser().resolve())

    pose_figure, (relative_axis, progress_axis, trajectory_axis) = plt.subplots(
        1, 3, figsize=(20, 6), constrained_layout=True
    )
    relative_yaw_axis = relative_axis.twinx()
    (relative_x_line,) = relative_axis.plot([], [], label="rel x", color="tab:blue")
    (relative_y_line,) = relative_axis.plot([], [], label="rel y", color="tab:orange")
    (distance_line,) = relative_axis.plot(
        [], [], label="rel distance", color="tab:green", linewidth=2.2
    )
    (relative_yaw_line,) = relative_yaw_axis.plot(
        [], [], label="rel yaw", color="tab:red", alpha=0.8
    )
    relative_axis.axhline(0.0, color="black", linewidth=0.7)
    relative_axis.set_title("command_rel_pos")
    relative_axis.set_xlabel("time [s]")
    relative_axis.set_ylabel("position / distance [m]")
    relative_yaw_axis.set_ylabel("yaw [rad]")
    relative_axis.grid(True)
    relative_lines = [relative_x_line, relative_y_line, distance_line, relative_yaw_line]
    relative_axis.legend(relative_lines, [line.get_label() for line in relative_lines])

    (closest_progress_line,) = progress_axis.plot(
        [], [], label="closest progress", color="tab:purple", linewidth=2.0
    )
    progress_axis.set_title("Closest path progress")
    progress_axis.set_xlabel("time [s]")
    progress_axis.set_ylabel("path progress [m]")
    progress_axis.grid(True)
    progress_axis.legend(loc="best")

    (trajectory_line,) = trajectory_axis.plot(
        [], [], label="robot path", color="tab:blue"
    )
    (target_path_line,) = trajectory_axis.plot(
        [],
        [],
        label="target path",
        color="tab:green",
        linestyle="--",
        linewidth=2.0,
    )
    MAX_ARROWS = 20

    theta_quiver = trajectory_axis.quiver(
        np.zeros(MAX_ARROWS),
        np.zeros(MAX_ARROWS),
        np.zeros(MAX_ARROWS),
        np.zeros(MAX_ARROWS),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="tab:orange",
        width=0.004,
    )
    (start_marker,) = trajectory_axis.plot([], [], "ko", markersize=5, label="start")
    (goal_marker,) = trajectory_axis.plot([], [], "r*", markersize=13, label="goal")
    (goal_heading,) = trajectory_axis.plot([], [], color="tab:red", linewidth=2)
    trajectory_axis.set_title("SLAM / simulator pose projected to XYθ")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.grid(True)
    trajectory_axis.legend(loc="best")
    pose_status = pose_figure.suptitle(f"Waiting for {tail.path}")

    joint_figure, joint_axes = plt.subplots(
        1,
        3,
        figsize=(20, 6),
        sharex=True,
        constrained_layout=True,
    )

    command_axis = joint_axes[0]
    encoder_axis = joint_axes[1]
    action_axis = joint_axes[2]

    command_axis.set_title("Command joints (arm pose target)")
    encoder_axis.set_title("Encoder joints")
    action_axis.set_title("Action joints")

    for ax in joint_axes:
        ax.grid(True)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Joint position [rad]")

    command_lines = []
    encoder_lines = []
    action_lines = []

    for joint_index in command_joint_indices:
        command_lines.append(
            (
                joint_index,
                command_axis.plot(
                    [],
                    [],
                    linewidth=1,
                    label=G1_JOINT_NAMES[joint_index],
                )[0],
            )
        )

    for joint_index in measured_joint_indices:
        encoder_lines.append(
            (
                joint_index,
                encoder_axis.plot(
                    [],
                    [],
                    linewidth=1,
                    label=G1_JOINT_NAMES[joint_index],
                )[0],
            )
        )

        action_lines.append(
            (
                joint_index,
                action_axis.plot(
                    [],
                    [],
                    linewidth=1,
                    label=G1_JOINT_NAMES[joint_index],
                )[0],
            )
        )

    for axis, lines in (
        (command_axis, command_lines),
        (encoder_axis, encoder_lines),
        (action_axis, action_lines),
    ):
        if lines:
            axis.legend(ncol=4, fontsize=6, loc="upper right")

    joint_status = joint_figure.suptitle(f"Waiting for {tail.path}")

    def update(_frame: int):
        tail.poll()
        target_path.poll()
        if not tail.values.get("time_s"):
            message = tail.error or f"Waiting for samples in {tail.path}"
            pose_status.set_text(message)
            joint_status.set_text(message)
            return ()

        data = {
            field: np.asarray(values, dtype=float)
            for field, values in tail.values.items()
            if field != "source"
        }
        time_s = data["time_s"]
        relative_x_line.set_data(time_s, data["command_rel_pos_x_m"])
        relative_y_line.set_data(time_s, data["command_rel_pos_y_m"])
        distance_line.set_data(time_s, data["position_error_m"])
        relative_yaw_line.set_data(time_s, data["command_rel_yaw_rad"])
        closest_progress_line.set_data(time_s, data["closest_progress_m"])

        slam_x = data["slam_pose_x_m"]
        slam_y = data["slam_pose_y_m"]
        slam_theta = data["slam_pose_theta_rad"]
        trajectory_line.set_data(slam_x, slam_y)
        valid_pose = np.flatnonzero(
            np.isfinite(slam_x) & np.isfinite(slam_y) & np.isfinite(slam_theta)
        )
        target_world_x = np.asarray([], dtype=float)
        target_world_y = np.asarray([], dtype=float)
        if valid_pose.size:
            first = valid_pose[0]
            origin_x, origin_y, origin_yaw = tail.navigation_origin or (
                slam_x[first],
                slam_y[first],
                slam_theta[first],
            )
            start_marker.set_data([origin_x], [origin_y])

            target_local_x = np.asarray(target_path.values["x_m"], dtype=float)
            target_local_y = np.asarray(target_path.values["y_m"], dtype=float)
            if target_local_x.size:
                cos_start = math.cos(origin_yaw)
                sin_start = math.sin(origin_yaw)
                target_world_x = (
                    origin_x
                    + cos_start * target_local_x
                    - sin_start * target_local_y
                )
                target_world_y = (
                    origin_y
                    + sin_start * target_local_x
                    + cos_start * target_local_y
                )
                target_path_line.set_data(target_world_x, target_world_y)
            else:
                target_path_line.set_data([], [])

            arrow_count = min(MAX_ARROWS, valid_pose.size)

            arrow_indices = valid_pose[
                np.linspace(
                    0,
                    valid_pose.size - 1,
                    arrow_count,
                ).astype(int)
            ]

            arrow_x = np.zeros(MAX_ARROWS)
            arrow_y = np.zeros(MAX_ARROWS)
            arrow_u = np.zeros(MAX_ARROWS)
            arrow_v = np.zeros(MAX_ARROWS)

            arrow_length = 0.12

            arrow_x[:arrow_count] = slam_x[arrow_indices]
            arrow_y[:arrow_count] = slam_y[arrow_indices]

            arrow_u[:arrow_count] = (
                arrow_length * np.cos(slam_theta[arrow_indices])
            )

            arrow_v[:arrow_count] = (
                arrow_length * np.sin(slam_theta[arrow_indices])
            )

            theta_quiver.set_offsets(
                np.column_stack((arrow_x, arrow_y))
            )

            theta_quiver.set_UVC(
                arrow_u,
                arrow_v,
            )

            goal_x_local = data["goal_x_m"][-1]
            goal_y_local = data["goal_y_m"][-1]
            cos_start = math.cos(origin_yaw)
            sin_start = math.sin(origin_yaw)
            goal_x = origin_x + cos_start * goal_x_local - sin_start * goal_y_local
            goal_y = origin_y + sin_start * goal_x_local + cos_start * goal_y_local
            goal_theta = origin_yaw + data["goal_yaw_rad"][-1]
            goal_marker.set_data([goal_x], [goal_y])
            goal_heading.set_data(
                [goal_x, goal_x + 0.18 * math.cos(goal_theta)],
                [goal_y, goal_y + 0.18 * math.sin(goal_theta)],
            )
        else:
            start_marker.set_data([], [])
            goal_marker.set_data([], [])
            goal_heading.set_data([], [])
            target_path_line.set_data([], [])

        command_fields = {
            tail._joint_sort_key(field): field
            for field in tail.joint_fields["command_joint_"]
        }
        encoder_fields = {
            tail._joint_sort_key(field): field
            for field in tail.joint_fields["encoder_joint_"]
        }
        action_fields = {
            tail._joint_sort_key(field): field
            for field in tail.joint_fields["action_joint_"]
        }

        for joint_index, line in command_lines:
            command_field = command_fields.get(joint_index)
            line.set_data(
                time_s if command_field else [],
                data[command_field] if command_field else [],
            )

        for joint_index, line in encoder_lines:
            encoder_field = encoder_fields.get(joint_index)
            line.set_data(
                time_s if encoder_field else [],
                data[encoder_field] if encoder_field else [],
            )

        for joint_index, line in action_lines:
            action_field = action_fields.get(joint_index)
            line.set_data(
                time_s if action_field else [],
                data[action_field] if action_field else [],
            )

        for axis in joint_axes:
            axis.relim()
            axis.autoscale_view()

        joint_figure.canvas.draw_idle()

        relative_axis.relim()
        relative_axis.autoscale_view()

        relative_yaw_axis.relim()
        relative_yaw_axis.autoscale_view()

        progress_axis.relim()
        progress_axis.autoscale_view()

        if valid_pose.size:
            valid_x = slam_x[valid_pose]
            valid_y = slam_y[valid_pose]

            if target_world_x.size:
                valid_x = np.concatenate((valid_x, target_world_x))
                valid_y = np.concatenate((valid_y, target_world_y))

            margin = 0.2

            xmin = np.min(valid_x) - margin
            xmax = np.max(valid_x) + margin
            ymin = np.min(valid_y) - margin
            ymax = np.max(valid_y) + margin

            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0

            half_range = max(
                (xmax - xmin) / 2.0,
                (ymax - ymin) / 2.0,
                0.2,
            )

            trajectory_axis.set_xlim(
                cx - half_range,
                cx + half_range,
            )
            trajectory_axis.set_ylim(
                cy - half_range,
                cy + half_range,
            )

        for axis in joint_axes:
            axis.relim()
            axis.autoscale_view()

        source = tail.values["source"][-1]
        localization_ok = bool(data["localization_available"][-1])
        reached = bool(data["goal_reached"][-1])
        state = "goal reached" if reached else "tracking"
        if not localization_ok:
            state = "LOCALIZATION UNAVAILABLE"
        status_text = f"G1 Navigation | source={source} | {state}"
        pose_status.set_text(status_text)
        joint_status.set_text(status_text)
        return ()

    animation = FuncAnimation(
        pose_figure,
        update,
        interval=args.interval_ms,
        cache_frame_data=False,
    )
    # Keep references on both windows for GUI backends that collect animations.
    pose_figure._navigation_animation = animation  # type: ignore[attr-defined]
    joint_figure._navigation_animation = animation  # type: ignore[attr-defined]
    plt.show()


if __name__ == "__main__":
    main()
