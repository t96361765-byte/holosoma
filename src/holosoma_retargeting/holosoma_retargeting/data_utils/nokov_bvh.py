"""Strict BVH reader for NOKOV Body motion exports.

NOKOV Body BVH files use centimetres and a Y-up world.  Depending on the
export preset, non-root joints contain either three rotation channels or six
local-position plus rotation channels.  Unlike the generic LAFAN reader used
by GMR, this module records and decodes every joint's channel list independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NOKOV_RETARGET_JOINTS = [
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
]

_POSITION_CHANNELS = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
_ROTATION_CHANNELS = {"Xrotation": 0, "Yrotation": 1, "Zrotation": 2}
_SUPPORTED_CHANNELS = set(_POSITION_CHANNELS) | set(_ROTATION_CHANNELS)
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
class NokovBvhMotion:
    """Parsed NOKOV animation in the original Y-up, centimetre convention."""

    source_path: Path
    joint_names: tuple[str, ...]
    parents: np.ndarray
    offsets_cm: np.ndarray
    channels: tuple[tuple[str, ...], ...]
    frame_time: float
    local_positions_cm: np.ndarray
    local_quaternions_wxyz: np.ndarray
    global_positions_cm: np.ndarray
    global_quaternions_wxyz: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.global_positions_cm.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Hamilton product for WXYZ quaternions with arbitrary leading axes."""

    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_term = 2.0 * np.cross(quaternion[..., 1:], vector)
    return vector + quaternion[..., :1] * vector_term + np.cross(quaternion[..., 1:], vector_term)


def _axis_quaternion(angle_radians: np.ndarray, axis: int) -> np.ndarray:
    result = np.zeros((*angle_radians.shape, 4), dtype=np.float64)
    result[..., 0] = np.cos(angle_radians / 2.0)
    result[..., axis + 1] = np.sin(angle_radians / 2.0)
    return result


def _decode_motion(
    frame_values: np.ndarray,
    offsets_cm: np.ndarray,
    channels: list[tuple[str, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = frame_values.shape[0]
    num_joints = len(channels)
    positions = np.broadcast_to(offsets_cm, (num_frames, num_joints, 3)).copy()
    rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float64)
    rotations[..., 0] = 1.0

    cursor = 0
    for joint_index, joint_channels in enumerate(channels):
        joint_values = frame_values[:, cursor : cursor + len(joint_channels)]
        cursor += len(joint_channels)

        rotation_quaternions: list[np.ndarray] = []
        for channel_index, channel_name in enumerate(joint_channels):
            if channel_name in _POSITION_CHANNELS:
                positions[:, joint_index, _POSITION_CHANNELS[channel_name]] = joint_values[:, channel_index]
            elif channel_name in _ROTATION_CHANNELS:
                angle = np.deg2rad(joint_values[:, channel_index])
                rotation_quaternions.append(_axis_quaternion(angle, _ROTATION_CHANNELS[channel_name]))

        if rotation_quaternions:
            joint_rotation = rotation_quaternions[-1]
            for quaternion in reversed(rotation_quaternions[:-1]):
                joint_rotation = _quat_multiply(quaternion, joint_rotation)
            rotations[:, joint_index] = joint_rotation

    norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
    rotations /= np.maximum(norms, np.finfo(np.float64).eps)
    for frame_index in range(1, num_frames):
        flip = np.sum(rotations[frame_index - 1] * rotations[frame_index], axis=-1) < 0.0
        rotations[frame_index, flip] *= -1.0
    return positions, rotations


def _forward_kinematics(
    local_positions: np.ndarray,
    local_quaternions: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global_positions = np.empty_like(local_positions)
    global_quaternions = np.empty_like(local_quaternions)
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            global_positions[:, joint_index] = local_positions[:, joint_index]
            global_quaternions[:, joint_index] = local_quaternions[:, joint_index]
        else:
            global_positions[:, joint_index] = (
                _quat_rotate(global_quaternions[:, parent_index], local_positions[:, joint_index])
                + global_positions[:, parent_index]
            )
            global_quaternions[:, joint_index] = _quat_multiply(
                global_quaternions[:, parent_index], local_quaternions[:, joint_index]
            )
    return global_positions, global_quaternions


def read_nokov_bvh(path: str | Path) -> NokovBvhMotion:
    """Read and validate a NOKOV Body BVH file.

    The returned arrays retain NOKOV's source units and axes.  Use
    :func:`extract_nokov_global_positions` for metre-valued retargeting input.
    """

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
            if len(fields) != 4:
                raise ValueError(f"Malformed OFFSET at line {line_number}: {raw_line}")
            if not in_end_site:
                offsets[active_joint] = np.asarray(fields[1:], dtype=np.float64)
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

    if not joint_names or parents[0] != -1 or joint_names[0] != "Hips":
        raise ValueError("Expected a NOKOV skeleton rooted at Hips")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("BVH contains duplicate joint names")
    missing_joints = sorted(set(NOKOV_RETARGET_JOINTS) - set(joint_names))
    if missing_joints:
        raise ValueError(f"NOKOV BVH is missing required joints: {missing_joints}")
    if channels[0] != _ROOT_CHANNELS:
        raise ValueError(f"Unexpected NOKOV Hips channels: {channels[0]}")
    for joint_name, joint_channels in zip(joint_names[1:], channels[1:], strict=True):
        if joint_channels not in (_ROTATION_ONLY_CHANNELS, _ROOT_CHANNELS):
            raise ValueError(f"Unexpected NOKOV channels for {joint_name}: {joint_channels}")

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

    offsets_cm = np.asarray(offsets, dtype=np.float64)
    local_positions, local_quaternions = _decode_motion(frame_values, offsets_cm, channels)
    global_positions, global_quaternions = _forward_kinematics(local_positions, local_quaternions, np.asarray(parents))
    return NokovBvhMotion(
        source_path=source_path.resolve(),
        joint_names=tuple(joint_names),
        parents=np.asarray(parents, dtype=np.int64),
        offsets_cm=offsets_cm,
        channels=tuple(channels),
        frame_time=frame_time,
        local_positions_cm=local_positions,
        local_quaternions_wxyz=local_quaternions,
        global_positions_cm=global_positions,
        global_quaternions_wxyz=global_quaternions,
    )


def _resample_indices(num_frames: int, source_fps: float, target_fps: float | None) -> np.ndarray:
    if target_fps is None:
        return np.arange(num_frames, dtype=np.int64)
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    duration = (num_frames - 1) / source_fps
    num_samples = int(np.floor(duration * target_fps + 1e-9)) + 1
    sample_times = np.arange(num_samples, dtype=np.float64) / target_fps
    return np.minimum(np.rint(sample_times * source_fps).astype(np.int64), num_frames - 1)


def extract_nokov_global_positions(
    path: str | Path,
    *,
    z_up: bool = False,
    target_fps: float | None = None,
) -> dict[str, object]:
    """Extract named global positions in metres, optionally Z-up/resampled."""

    motion = read_nokov_bvh(path)
    joint_indices = np.asarray([motion.joint_names.index(name) for name in NOKOV_RETARGET_JOINTS])
    source_to_output = {int(source_index): output_index for output_index, source_index in enumerate(joint_indices)}
    output_parents: list[int] = []
    for source_index in joint_indices:
        source_parent = int(motion.parents[source_index])
        while source_parent >= 0 and source_parent not in source_to_output:
            source_parent = int(motion.parents[source_parent])
        output_parents.append(-1 if source_parent < 0 else source_to_output[source_parent])
    positions = motion.global_positions_cm[:, joint_indices] / 100.0
    if z_up:
        positions = positions[..., [0, 2, 1]]
        positions[..., 1] *= -1.0
    frame_indices = _resample_indices(motion.num_frames, motion.fps, target_fps)
    positions = positions[frame_indices]
    return {
        "positions": positions,
        "joint_names": list(NOKOV_RETARGET_JOINTS),
        "parents": np.asarray(output_parents, dtype=np.int64),
        "num_frames": int(positions.shape[0]),
        "num_source_frames": motion.num_frames,
        "num_joints": int(positions.shape[1]),
        "source_fps": motion.fps,
        "target_fps": motion.fps if target_fps is None else float(target_fps),
        "frame_indices": frame_indices,
        "source_joint_names": list(motion.joint_names),
        "source_parents": motion.parents,
    }
