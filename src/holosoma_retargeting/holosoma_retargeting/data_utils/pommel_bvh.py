"""Parser for metre-valued, full-rotation pommel BVH skeletons.

The source files are already metre-valued and Z-up.  Their horizontal axes
are ``+X = anatomical left`` and ``-Y = forward``; this is a right-handed
frame compatible with OmniRetarget, so no NOKOV Y-up conversion is applied.

The marker53-derived 23-joint export is supported.  The parser preserves its
source joints and hierarchy exactly; End Sites are parsed as BVH metadata but
are not promoted to retargeting joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holosoma_retargeting.data_utils.nokov_bvh import (
    _decode_motion,
    _forward_kinematics,
    _resample_indices,
    _SUPPORTED_CHANNELS,
)

POMMEL_SOURCE_JOINTS = (
    "Hips",
    "Chest",
    "Chest2",
    "Chest3",
    "Chest4",
    "Neck",
    "Head",
    "RightCollar",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
    "LeftCollar",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightHip",
    "RightKnee",
    "RightAnkle",
    "RightToe",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "LeftToe",
)

_ROOT_CHANNELS = (
    "Xposition",
    "Yposition",
    "Zposition",
    "Zrotation",
    "Xrotation",
    "Yrotation",
)
_ROTATION_ONLY_CHANNELS = ("Zrotation", "Xrotation", "Yrotation")


@dataclass(frozen=True)
class PommelBvhMotion:
    """Parsed source motion in its original metre-valued Z-up frame."""

    source_path: Path
    joint_names: tuple[str, ...]
    parents: np.ndarray
    offsets_m: np.ndarray
    end_site_offsets_m: dict[str, np.ndarray]
    channels: tuple[tuple[str, ...], ...]
    frame_time: float
    local_positions_m: np.ndarray
    local_quaternions_wxyz: np.ndarray
    global_positions_m: np.ndarray
    global_quaternions_wxyz: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.global_positions_m.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time


def read_pommel_bvh(path: str | Path) -> PommelBvhMotion:
    """Read and strictly validate a reduced full-rotation pommel BVH."""

    source_path = Path(path)
    lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    try:
        motion_line = next(index for index, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration as exc:
        raise ValueError(f"BVH has no MOTION section: {source_path}") from exc

    joint_names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[tuple[str, ...]] = []
    end_site_offsets: dict[str, np.ndarray] = {}
    active_joint = -1
    in_end_site = False

    for line_number, raw_line in enumerate(lines[:motion_line], start=1):
        stripped = raw_line.strip()
        fields = stripped.split()
        if not fields or fields[0] in {"HIERARCHY", "{"}:
            continue
        if fields[0] in {"ROOT", "JOINT"}:
            if len(fields) != 2:
                raise ValueError(f"Malformed joint declaration at line {line_number}: {raw_line}")
            joint_names.append(fields[1])
            parents.append(active_joint)
            offsets.append(np.zeros(3, dtype=np.float64))
            channels.append(())
            active_joint = len(joint_names) - 1
            continue
        if stripped == "End Site":
            in_end_site = True
            continue
        if stripped == "}":
            if in_end_site:
                in_end_site = False
            elif active_joint >= 0:
                active_joint = parents[active_joint]
            continue
        if fields[0] == "OFFSET":
            if len(fields) != 4 or active_joint < 0:
                raise ValueError(f"Malformed OFFSET at line {line_number}: {raw_line}")
            offset = np.asarray(fields[1:], dtype=np.float64)
            if in_end_site:
                end_site_offsets[joint_names[active_joint]] = offset
            else:
                offsets[active_joint] = offset
            continue
        if fields[0] == "CHANNELS":
            declared_count = int(fields[1])
            declared_channels = tuple(fields[2:])
            if declared_count != len(declared_channels):
                raise ValueError(f"CHANNELS count mismatch at line {line_number}: {raw_line}")
            unknown_channels = set(declared_channels) - _SUPPORTED_CHANNELS
            if unknown_channels:
                raise ValueError(f"Unsupported BVH channels at line {line_number}: {sorted(unknown_channels)}")
            channels[active_joint] = declared_channels

    joint_names_tuple = tuple(joint_names)
    if joint_names_tuple != POMMEL_SOURCE_JOINTS:
        raise ValueError(
            "Unexpected pommel skeleton. Expected joints in hierarchy order "
            f"{POMMEL_SOURCE_JOINTS}, got {joint_names_tuple}"
        )
    if channels[0] != _ROOT_CHANNELS:
        raise ValueError(f"Unexpected pommel Hips channels: {channels[0]}")
    for joint_name, joint_channels in zip(joint_names[1:], channels[1:], strict=True):
        if joint_channels != _ROTATION_ONLY_CHANNELS:
            raise ValueError(f"Unexpected pommel channels for {joint_name}: {joint_channels}")
    missing_end_sites = sorted({"Head", "LeftToe", "RightToe"} - set(end_site_offsets))
    if missing_end_sites:
        raise ValueError(f"Pommel BVH is missing required End Sites: {missing_end_sites}")

    offsets_m = np.asarray(offsets, dtype=np.float64)
    index = {name: i for i, name in enumerate(joint_names)}
    # Guard the coordinate convention before an incorrect NOKOV-like axis swap
    # can silently turn the performer sideways or upside down.
    if offsets_m[index["Chest"], 2] <= 0.0:
        raise ValueError("Expected pommel +Z to be up (Chest must be above Hips)")
    if offsets_m[index["LeftHip"], 0] <= offsets_m[index["RightHip"], 0]:
        raise ValueError("Expected pommel +X to point toward the anatomical left side")
    if end_site_offsets["LeftToe"][1] >= 0.0 or end_site_offsets["RightToe"][1] >= 0.0:
        raise ValueError("Expected pommel -Y to point forward at the foot End Sites")

    motion_header = [line.strip() for line in lines[motion_line + 1 :] if line.strip()]
    if len(motion_header) < 3 or not motion_header[0].startswith("Frames:"):
        raise ValueError(f"Malformed MOTION header: {source_path}")
    declared_frames = int(motion_header[0].split(":", maxsplit=1)[1])
    if not motion_header[1].startswith("Frame Time:"):
        raise ValueError(f"Missing Frame Time in {source_path}")
    frame_time = float(motion_header[1].split(":", maxsplit=1)[1])
    if frame_time <= 0.0:
        raise ValueError(f"Frame Time must be positive, got {frame_time}")

    expected_channels = sum(len(item) for item in channels)
    flat_values = np.fromstring(" ".join(motion_header[2:]), sep=" ", dtype=np.float64)
    expected_values = declared_frames * expected_channels
    if flat_values.size != expected_values:
        raise ValueError(
            f"MOTION contains {flat_values.size} values; expected "
            f"{declared_frames} frames x {expected_channels} channels = {expected_values}"
        )
    frame_values = flat_values.reshape(declared_frames, expected_channels)
    if not np.all(np.isfinite(frame_values)):
        raise ValueError("MOTION contains non-finite values")

    local_positions, local_quaternions = _decode_motion(frame_values, offsets_m, channels)
    global_positions, global_quaternions = _forward_kinematics(
        local_positions, local_quaternions, np.asarray(parents, dtype=np.int64)
    )
    return PommelBvhMotion(
        source_path=source_path.resolve(),
        joint_names=tuple(joint_names),
        parents=np.asarray(parents, dtype=np.int64),
        offsets_m=offsets_m,
        end_site_offsets_m=end_site_offsets,
        channels=tuple(channels),
        frame_time=frame_time,
        local_positions_m=local_positions,
        local_quaternions_wxyz=local_quaternions,
        global_positions_m=global_positions,
        global_quaternions_wxyz=global_quaternions,
    )


def extract_pommel_global_positions(
    path: str | Path,
    *,
    target_fps: float | None = None,
) -> dict[str, object]:
    """Return the original named joints in OmniRetarget XYZ metres."""

    motion = read_pommel_bvh(path)
    positions = motion.global_positions_m
    frame_indices = _resample_indices(motion.num_frames, motion.fps, target_fps)
    positions = positions[frame_indices]
    return {
        "positions": positions,
        "joint_names": list(motion.joint_names),
        "parents": motion.parents.copy(),
        "num_frames": int(positions.shape[0]),
        "num_source_frames": motion.num_frames,
        "num_joints": int(positions.shape[1]),
        "source_fps": motion.fps,
        "target_fps": motion.fps if target_fps is None else float(target_fps),
        "frame_indices": frame_indices,
        "source_joint_names": list(motion.joint_names),
        "source_parents": motion.parents,
        "source_units": "metres",
        "source_axes": "+X left, -Y forward, +Z up",
        "omni_axis_transform": np.eye(3, dtype=np.float64),
    }
