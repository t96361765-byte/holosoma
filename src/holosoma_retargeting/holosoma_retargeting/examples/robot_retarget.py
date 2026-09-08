"""
Unified robot retargeting script for all task types:
- robot_only: Robot-only retargeting with ground interaction
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import trimesh
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PACKAGE_ROOT / "models"

from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY, MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.data_utils.nokov_bvh import extract_nokov_global_positions  # noqa: E402
from holosoma_retargeting.data_utils.nokov_fixed_object import (  # noqa: E402
    create_static_mesh_scene,
    create_static_mesh_urdf,
    create_upper_surface_collision_mesh,
    ground_and_recenter_static_pose,
    level_static_pose,
    load_static_nokov_pose,
    translate_human_joints,
)
from holosoma_retargeting.data_utils.pommel_bvh import extract_pommel_global_positions  # noqa: E402
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,  # type: ignore[import-not-found]
)
from holosoma_retargeting.src.utils import (  # noqa: E402
    augment_object_poses,
    calculate_scale_factor,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_urdf,
    create_scaled_multi_boxes_xml,
    estimate_human_orientation,
    extract_foot_sticking_sequence_velocity,
    extract_object_first_moving_frame,
    load_intermimic_data,
    load_object_data,
    preprocess_motion_data,
    transform_from_human_to_world,
    transform_points_world_to_local,
    transform_y_up_to_z_up,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------- Constants -----------------------------

# Task-specific defaults
DEFAULT_DATA_FORMATS = {
    "robot_only": "smplh",
    "object_interaction": "smplh",
    "climbing": "mocap",
}

DEFAULT_SAVE_DIRS = {
    "robot_only": "demo_results/{robot}/robot_only/omomo",
    "object_interaction": "demo_results/{robot}/object_interaction/omomo",
    "climbing": "demo_results/{robot}/climbing/mocap_climb",
}


# Constants for numpy arrays (not in dataclass to avoid tyro parsing issues)
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_OBJECT_SCALE_NORMAL = np.array([1.0, 1.0, 1.0])
_AUGMENTATION_TRANSLATION = np.array([0.2, 0.0, 0.0])


# Type aliases
TaskType = Literal["robot_only", "object_interaction", "climbing"]
# DataFormat is imported from config_types.data_type


# ----------------------------- Helper Functions -----------------------------


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    task_config: TaskConfig,
    task_type: str,
) -> SimpleNamespace:
    """Create combined task constants from robot and motion data configs.

    Args:
        robot_config: Robot configuration
        motion_data_config: Motion data format configuration
        task_config: Task-specific configuration
        task_type: Type of task ("robot_only", "object_interaction", "climbing")

    Returns:
        SimpleNamespace with all task constants
    """
    task_constants = SimpleNamespace()

    # Copy all attributes from robot_config
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(task_constants, attr, getattr(robot_config, attr))

    # Copy legacy motion data constants (upper-case for compatibility)
    for attr, value in motion_data_config.legacy_constants().items():
        setattr(task_constants, attr, value)

    # Task-specific object setup
    if task_type == "robot_only":
        obj_name = task_config.object_name or "ground"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None
    elif task_type == "object_interaction":
        obj_name = task_config.object_name or "largebox"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = f"models/{obj_name}/{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = f"models/{obj_name}/{obj_name}.obj"
        task_constants.OBJECT_URDF_TEMPLATE = f"models/templates/{obj_name}.urdf.jinja"
    elif task_type == "climbing":
        obj_name = task_config.object_name or "multi_boxes"
        task_constants.OBJECT_NAME = obj_name
        object_dir = task_config.object_dir
        task_constants.OBJECT_DIR = str(object_dir) if object_dir else ""
        task_constants.FIXED_OBJECT = task_config.object_mesh is not None
        task_constants.OBJECT_URDF_FILE = (
            str((MODELS_ROOT / obj_name / f"{obj_name}_static.urdf").resolve())
            if task_constants.FIXED_OBJECT
            else str(object_dir / f"{obj_name}.urdf")
            if object_dir
            else f"{obj_name}.urdf"
        )
        task_constants.OBJECT_MESH_FILE = (
            str(task_config.object_mesh)
            if task_config.object_mesh is not None
            else str(object_dir / f"{obj_name}.obj")
            if object_dir
            else f"{obj_name}.obj"
        )
        task_constants.SCENE_XML_FILE = ""  # Will be set later

    return task_constants


def validate_config(cfg: RetargetingConfig) -> None:
    """Validate configuration consistency.

    Args:
        cfg: Configuration arguments

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate that data_format exists in registry (if provided)
    if cfg.data_format is not None and cfg.data_format not in DEMO_JOINTS_REGISTRY:
        available = ", ".join(sorted(DEMO_JOINTS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown data_format: '{cfg.data_format}'. "
            f"Available formats: {available}. "
            f"Add your format to DEMO_JOINTS_REGISTRY in config_types/data_type.py"
        )

    # Task-specific format requirements
    if cfg.task_type == "climbing" and cfg.data_format not in (None, "mocap", "nokov", "pommel"):
        raise ValueError("Climbing task requires 'mocap', 'nokov', or 'pommel' data format")
    if cfg.task_type == "object_interaction" and cfg.data_format not in (None, "smplh"):
        raise ValueError("Object interaction requires 'smplh' data format")
    # robot_only accepts any format in the registry (already validated above)


def create_ground_points(x_range: tuple[float, float], y_range: tuple[float, float], size: int) -> np.ndarray:
    """Create ground point meshgrid.

    Args:
        x_range: (min, max) x-coordinate range
        y_range: (min, max) y-coordinate range
        size: Number of points per dimension

    Returns:
        (N, 3) array of ground points
    """
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)


def sample_fixed_object_contact_points(
    mesh_file: str | Path,
    sample_count: int,
    task_config: TaskConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Sample the complete upper support cap uniformly by triangle area."""
    if sample_count < 4:
        raise ValueError("fixed_object_sample_count must be at least 4")

    if not 0 < task_config.fixed_object_upper_height_fraction < 1:
        raise ValueError("fixed_object_upper_height_fraction must be between 0 and 1")
    if not 0 < task_config.fixed_object_upper_interior_radius <= 1:
        raise ValueError("fixed_object_upper_interior_radius must be in (0, 1]")
    if not -1 <= task_config.fixed_object_upper_normal_z <= 1:
        raise ValueError("fixed_object_upper_normal_z must be between -1 and 1")

    mesh = trimesh.load(mesh_file, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Object asset is not a non-empty triangle mesh: {mesh_file}")

    face_centers = mesh.triangles_center
    bounds = mesh.bounds
    height_span = bounds[1, 2] - bounds[0, 2]
    xy_half_extent = (bounds[1, :2] - bounds[0, :2]) / 2.0
    if height_span <= 0 or np.any(xy_half_extent <= 0):
        raise ValueError(f"Object mesh has degenerate bounds: {mesh_file}")

    normalized_height = (face_centers[:, 2] - bounds[0, 2]) / height_span
    xy_center = (bounds[0, :2] + bounds[1, :2]) / 2.0
    normalized_radius = np.linalg.norm((face_centers[:, :2] - xy_center) / xy_half_extent, axis=1)
    upper_surface = (
        (normalized_height >= task_config.fixed_object_upper_height_fraction)
        & (mesh.face_normals[:, 2] >= task_config.fixed_object_upper_normal_z)
        & (normalized_radius <= task_config.fixed_object_upper_interior_radius)
    )
    if not np.any(upper_surface):
        raise ValueError("No faces satisfy the fixed-object upper-surface filters")
    points, _ = trimesh.sample.sample_surface(
        mesh,
        sample_count,
        face_weight=upper_surface.astype(float),
        seed=task_config.fixed_object_sampling_seed,
    )

    face_areas = mesh.area_faces
    total_area = float(face_areas.sum())
    stats: dict[str, float | int] = {
        "upper_count": sample_count,
        "upper_face_count": int(np.count_nonzero(upper_surface)),
        "upper_area_fraction": float(face_areas[upper_surface].sum() / total_area),
    }
    return points, stats


def _resolve_nokov_capture_dir(data_path: Path, task_name: str) -> Path:
    """Accept either the NOKOV root folder or the sequence folder itself."""
    nested = data_path / task_name
    if (nested / f"{task_name}-Bodylt.bvh").is_file():
        return nested
    if (data_path / f"{task_name}-Bodylt.bvh").is_file():
        return data_path
    return nested


def align_fixed_nokov_capture(
    human_joints: np.ndarray,
    data_path: Path,
    task_name: str,
    task_config: TaskConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground/recenter a static mesh and apply the same translation to the human."""
    if task_config.object_mesh is None:
        raise ValueError("NOKOV fixed-object climbing requires --task-config.object-mesh")
    if task_config.object_scale <= 0:
        raise ValueError("task_config.object_scale must be positive")

    capture_dir = _resolve_nokov_capture_dir(data_path, task_name)
    object_pose_file = task_config.object_pose_file or capture_dir / f"{task_name}-mushroom.csv"
    raw_pose, pose_sample_count = load_static_nokov_pose(object_pose_file)
    leveled_pose = level_static_pose(raw_pose)
    mesh = trimesh.load(task_config.object_mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Object asset is not a non-empty triangle mesh: {task_config.object_mesh}")
    placed_pose, world_shift = ground_and_recenter_static_pose(
        leveled_pose,
        mesh,
        scale=task_config.object_scale,
    )
    aligned_human = translate_human_joints(human_joints, world_shift)
    object_poses = np.tile(placed_pose, (len(aligned_human), 1))
    logger.info(
        "Aligned fixed NOKOV object from %s (%d samples): xy=(0, 0), base z=0, human shift=%s m",
        object_pose_file,
        pose_sample_count,
        np.array2string(world_shift, precision=7),
    )
    return aligned_human, object_poses, world_shift


def align_fixed_pommel_capture(
    human_joints: np.ndarray,
    task_config: TaskConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Blender mushroom reference while preserving human registration."""

    if task_config.object_mesh is None:
        raise ValueError("Pommel fixed-object climbing requires --task-config.object-mesh")
    if task_config.object_scale <= 0:
        raise ValueError("task_config.object_scale must be positive")
    if not np.isfinite(task_config.pommel_z_compression) or task_config.pommel_z_compression <= 0:
        raise ValueError("pommel_z_compression must be positive and finite")
    pommel_scale_xyz = np.array(
        [
            task_config.object_scale,
            task_config.object_scale,
            task_config.object_scale / task_config.pommel_z_compression,
        ],
        dtype=float,
    )
    position = np.asarray(task_config.pommel_reference_position, dtype=float)
    quaternion = np.asarray(task_config.pommel_reference_quaternion, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("pommel_reference_position must contain three finite XYZ values")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("pommel_reference_quaternion must contain four finite WXYZ values")
    quaternion_norm = np.linalg.norm(quaternion)
    if quaternion_norm < 1e-8:
        raise ValueError("pommel_reference_quaternion must be non-zero")
    quaternion /= quaternion_norm

    mesh = trimesh.load(task_config.object_mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Object asset is not a non-empty triangle mesh: {task_config.object_mesh}")
    reference_pose = np.concatenate((quaternion, position))
    placed_pose, world_shift = ground_and_recenter_static_pose(
        reference_pose,
        mesh,
        scale=pommel_scale_xyz,
    )
    aligned_human = translate_human_joints(human_joints, world_shift)
    radial_offset = float(task_config.pommel_human_radial_offset)
    if not np.isfinite(radial_offset):
        raise ValueError("pommel_human_radial_offset must be finite")
    radial_shift = np.zeros(3, dtype=float)
    if radial_offset != 0.0:
        initial_hips_xy = aligned_human[0, 0, :2]
        horizontal_distance = float(np.linalg.norm(initial_hips_xy))
        if horizontal_distance < 1e-8:
            raise ValueError(
                "Cannot infer the outward pommel direction because the first-frame "
                "Hips XY position is at the mushroom axis"
            )
        radial_shift[:2] = radial_offset * initial_hips_xy / horizontal_distance
        aligned_human = translate_human_joints(aligned_human, radial_shift)
    object_poses = np.tile(placed_pose, (len(aligned_human), 1))
    logger.info(
        "Aligned Blender pommel reference: source xyz=%s -> axis xy=(0, 0), base z=0, "
        "origin z=%.7f m, shared human/object shift=%s m, human radial shift=%s m",
        np.array2string(position, precision=7),
        placed_pose[6],
        np.array2string(world_shift, precision=7),
        np.array2string(radial_shift, precision=7),
    )
    return aligned_human, object_poses


def load_motion_data(
    task_type: TaskType,
    data_format: str,
    data_path: Path,
    task_name: str,
    constants: SimpleNamespace,
    motion_data_config: MotionDataConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load motion data based on task type and format.

    Args:
        task_type: Type of task
        data_format: Data format ("lafan", "smplh", "mocap", "nokov", "pommel")
        data_path: Path to data directory
        task_name: Name of the task/sequence
        constants: Task constants
        motion_data_config: Motion data configuration

    Returns:
        Tuple of (human_joints, object_poses, smpl_scale)
        - human_joints: (T, J, 3) array of joint positions
        - object_poses: (T, 7) array of object poses [qw, qx, qy, qz, x, y, z]
        - smpl_scale: Scaling factor for SMPL compatibility

    Raises:
        FileNotFoundError: If required data files are not found
    """
    logger.info("Loading motion data for task: %s, format: %s", task_name, data_format)

    if task_type == "robot_only":
        if data_format == "lafan":
            npy_path = data_path / f"{task_name}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"LAFAN data file not found: {npy_path}")

            human_joints = np.load(str(npy_path))
            human_joints = transform_y_up_to_z_up(human_joints)
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            # LAFAN-specific spine adjustment
            human_joints[:, spine_joint_idx, -1] -= 0.06
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        elif data_format == "smplh":  # smplh
            pt_path = data_path / f"{task_name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")

            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
        elif data_format == "mocap":
            downsample = 4
            npy_file = data_path / f"{task_name}.npy"
            if not npy_file.exists():
                raise FileNotFoundError(f"MOCAP data file not found: {npy_file}")

            human_joints = np.load(str(npy_file))[::downsample]

            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height
        elif data_format == "nokov":
            bvh_candidates = [
                data_path / f"{task_name}-Bodylt.bvh",
                data_path / f"{task_name}.bvh",
            ]
            bvh_file = data_path if data_path.is_file() else next((path for path in bvh_candidates if path.is_file()), None)
            if bvh_file is None:
                expected = " or ".join(str(path) for path in bvh_candidates)
                raise FileNotFoundError(f"NOKOV Body BVH not found; expected {expected}")

            parsed = extract_nokov_global_positions(bvh_file, z_up=True, target_fps=30.0)
            human_joints = np.asarray(parsed["positions"])
            logger.info(
                "Parsed robot-only NOKOV BVH: %d frames at %.6g Hz -> %d frames at 30 Hz",
                parsed["num_source_frames"],
                parsed["source_fps"],
                parsed["num_frames"],
            )
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        elif data_format == "pommel":
            bvh_file = data_path if data_path.is_file() else data_path / f"{task_name}.bvh"
            if not bvh_file.is_file():
                raise FileNotFoundError(f"Pommel BVH not found: {bvh_file}")

            parsed = extract_pommel_global_positions(bvh_file, target_fps=30.0)
            human_joints = np.asarray(parsed["positions"])
            logger.info(
                "Parsed robot-only pommel BVH without axis swapping: %d frames at %.6g Hz -> "
                "%d frames at 30 Hz (%s)",
                parsed["num_source_frames"],
                parsed["source_fps"],
                parsed["num_frames"],
                parsed["source_axes"],
            )
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        elif data_format == "smplx":
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height
        else:
            # For other custom data format, if it uses consistent .npz file like SMPLX,
            # you can use the same logic as SMPLX.
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height

        # Create dummy object poses for robot_only
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))

    elif task_type == "object_interaction":
        pt_path = data_path / f"{task_name}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")

        human_joints, object_poses = load_intermimic_data(str(pt_path))
        smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)

    elif task_type == "climbing":
        task_dir = _resolve_nokov_capture_dir(data_path, task_name) if data_format == "nokov" else data_path / task_name
        if data_format == "pommel":
            bvh_file = data_path if data_path.is_file() else data_path / f"{task_name}.bvh"
            if not bvh_file.is_file():
                raise FileNotFoundError(f"Pommel BVH not found: {bvh_file}")
            parsed = extract_pommel_global_positions(bvh_file, target_fps=30.0)
            human_joints = np.asarray(parsed["positions"])
            logger.info(
                "Parsed fixed-object pommel BVH without axis swapping: %d frames at %.6g Hz -> "
                "%d frames at 30 Hz (%s)",
                parsed["num_source_frames"],
                parsed["source_fps"],
                parsed["num_frames"],
                parsed["source_axes"],
            )
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        elif data_format == "nokov":
            preferred_bvh = task_dir / f"{task_name}-Bodylt.bvh"
            bvh_files = [preferred_bvh] if preferred_bvh.is_file() else sorted(task_dir.glob("*-Bodylt.bvh"))
            if len(bvh_files) != 1:
                raise FileNotFoundError(f"Expected one NOKOV *-Bodylt.bvh file in {task_dir}, found {len(bvh_files)}")
            parsed = extract_nokov_global_positions(bvh_files[0], z_up=True, target_fps=30.0)
            human_joints = np.asarray(parsed["positions"])
            logger.info(
                "Parsed NOKOV BVH: %d frames at %.6g Hz -> %d frames at 30 Hz",
                parsed["num_source_frames"],
                parsed["source_fps"],
                parsed["num_frames"],
            )
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        else:
            npy_files = list(task_dir.glob("*.npy"))
            if not npy_files:
                raise FileNotFoundError(f"No .npy file found in {task_dir}")
            # MOCAP-specific downsample factor
            human_joints = np.load(str(npy_files[0]))[::4]
            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))

    logger.debug(
        "Loaded %d frames, scale factor: %.4f",
        human_joints.shape[0],
        smpl_scale,
    )
    return human_joints, object_poses, smpl_scale


def setup_object_data(
    task_type: TaskType,
    constants: SimpleNamespace,
    object_dir: Path | None,
    smpl_scale: float,
    task_config: TaskConfig,
    augmentation: bool,
    object_poses: np.ndarray | None = None,
    object_scale_augmented: np.ndarray | None = None,
    human_joints: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Setup object-specific data (ground, object mesh, climbing terrain).
    Args:
        task_type: Type of task
        constants: Task constants
        object_dir: Object directory path (for climbing)
        smpl_scale: SMPL scaling factor
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        object_scale_augmented: Scale factor for augmented objects (default: [1.0, 1.0, 1.2])
    Returns:
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path)
    """
    object_scale_normal = np.array([1.0, 1.0, 1.0])
    if object_scale_augmented is None:
        object_scale_augmented = np.array([1.0, 1.0, 1.2])  # For climbing task augmentation
    logger.info("Setting up object data for task: %s", task_type)

    if task_type == "robot_only":
        # Create ground points meshgrid
        ground_pts = create_ground_points(task_config.ground_range, task_config.ground_range, task_config.ground_size)
        return ground_pts, ground_pts, None

    if task_type == "object_interaction":
        # Load object data
        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")

        object_local_pts, object_local_pts_demo = load_object_data(
            constants.OBJECT_MESH_FILE, smpl_scale=smpl_scale, sample_count=100
        )
        return object_local_pts, object_local_pts_demo, constants.OBJECT_URDF_FILE

    if task_type == "climbing":
        if object_dir is None:
            raise ValueError("object_dir must be provided for climbing task")

        if getattr(constants, "FIXED_OBJECT", False):
            if object_poses is None:
                raise ValueError("Fixed-object climbing requires a synchronized object pose")

            object_dir.mkdir(parents=True, exist_ok=True)
            target_pose = np.asarray(object_poses[0], dtype=float)
            if not np.isfinite(task_config.pommel_z_compression) or task_config.pommel_z_compression <= 0:
                raise ValueError("pommel_z_compression must be positive and finite")
            fixed_object_scale_xyz = np.array(
                [
                    task_config.object_scale,
                    task_config.object_scale,
                    task_config.object_scale / task_config.pommel_z_compression,
                ],
                dtype=float,
            )
            if task_config.fixed_object_surface_points_enabled:
                target_object_points, sampling_stats = sample_fixed_object_contact_points(
                    constants.OBJECT_MESH_FILE,
                    task_config.fixed_object_sample_count,
                    task_config,
                )
                target_object_points *= fixed_object_scale_xyz
                logger.info(
                    "Uniform fixed-object upper-cap sampling: %d points over %d faces "
                    "(%.1f%% of mesh area)",
                    sampling_stats["upper_count"],
                    sampling_stats["upper_face_count"],
                    100 * sampling_stats["upper_area_fraction"],
                )
            else:
                target_object_points = np.empty((0, 3), dtype=float)
                logger.info(
                    "Fixed object is collision-only: no surface points are included in "
                    "the interaction mesh"
                )
            fixed_object_surface_point_count = len(target_object_points)

            ground_nx, ground_ny = task_config.fixed_object_ground_shape
            ground_min, ground_max = task_config.fixed_object_ground_range
            if ground_nx < 2 or ground_ny < 2:
                raise ValueError("fixed_object_ground_shape values must both be at least 2")
            if ground_min >= ground_max:
                raise ValueError("fixed_object_ground_range must be ordered low to high")

            ground_x, ground_y = np.meshgrid(
                np.linspace(ground_min, ground_max, ground_nx),
                np.linspace(ground_min, ground_max, ground_ny),
                indexing="xy",
            )
            ground_points_world = np.column_stack(
                (ground_x.ravel(), ground_y.ravel(), np.zeros(ground_x.size))
            )
            ground_points_local = transform_points_world_to_local(
                target_pose[:4], target_pose[4:7], ground_points_world
            )
            target_object_points = np.vstack((target_object_points, ground_points_local))
            demo_object_points = target_object_points * smpl_scale
            logger.info(
                "Fixed-object interaction points: %d object surface + %d ground grid "
                "(%d x %d over [%.3f, %.3f] m in world XY)",
                fixed_object_surface_point_count,
                len(ground_points_local),
                ground_nx,
                ground_ny,
                ground_min,
                ground_max,
            )

            robot_xml = Path(constants.ROBOT_URDF_FILE).with_suffix(".xml").resolve()
            scene_xml = object_dir / f"{Path(constants.ROBOT_URDF_FILE).stem}_w_{constants.OBJECT_NAME}.xml"
            object_urdf = Path(constants.OBJECT_URDF_FILE)
            upper_collision_mesh = create_upper_surface_collision_mesh(
                Path(constants.OBJECT_MESH_FILE),
                object_dir / f"{constants.OBJECT_NAME}_upper_collision.obj",
                height_fraction=task_config.fixed_object_upper_height_fraction,
                normal_z=task_config.fixed_object_upper_normal_z,
                interior_radius=task_config.fixed_object_upper_interior_radius,
            )
            create_static_mesh_scene(
                robot_xml,
                Path(constants.OBJECT_MESH_FILE),
                scene_xml,
                target_pose,
                scale=fixed_object_scale_xyz,
                object_name=constants.OBJECT_NAME,
                collision_mesh=upper_collision_mesh,
            )
            create_static_mesh_urdf(
                Path(constants.OBJECT_MESH_FILE),
                object_urdf,
                scale=fixed_object_scale_xyz,
                object_name=constants.OBJECT_NAME,
            )
            constants.SCENE_XML_FILE = str(scene_xml.resolve())
            constants.STATIC_OBJECT_POSITION = target_pose[4:7].copy()
            constants.STATIC_OBJECT_QUAT = target_pose[:4].copy()
            constants.FIXED_OBJECT_TARGET_POSE = target_pose.copy()
            logger.info("Generated fixed-object retargeting scene: %s", scene_xml)
            return target_object_points, demo_object_points, str(object_urdf.resolve())

        # Setup climbing-specific object
        box_asset_xml = object_dir / "box_assets.xml"
        scene_xml_name = Path(constants.ROBOT_URDF_FILE).name.replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml")
        scene_xml_file = object_dir / scene_xml_name
        # Set SCENE_XML_FILE in constants BEFORE creating retargeter (needed for temp_retargeter)
        constants.SCENE_XML_FILE = str(scene_xml_file)

        np.random.seed(0)
        print("object mesh file: ", constants.OBJECT_MESH_FILE)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: (
                task_config.surface_weight_high
                if p[2] > task_config.surface_weight_threshold
                else task_config.surface_weight_low
            ),
            sample_count=100,
        )

        if augmentation:
            ground_pts = create_ground_points(
                task_config.climbing_ground_range, task_config.climbing_ground_range, task_config.climbing_ground_size
            )
            object_local_pts_demo = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
            object_scale = object_scale_augmented
            object_local_pts = object_scale * object_local_pts_demo
        else:
            object_scale = object_scale_normal
            object_local_pts_demo = object_local_pts_demo_original
            object_local_pts = object_local_pts_demo

        # Create scaled URDF and XML files
        scale_factors = tuple(float(value) for value in (object_scale * smpl_scale))
        object_urdf_file = create_scaled_multi_boxes_urdf(constants.OBJECT_URDF_FILE, scale_factors)
        object_asset_xml_path = create_scaled_multi_boxes_xml(str(box_asset_xml), scale_factors)
        new_scene_xml_path = create_new_scene_xml_file(str(scene_xml_file), scale_factors, object_asset_xml_path)
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file

    raise ValueError(f"Unknown task type: {task_type}")


def _compute_q_init_base(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter | None = None,
) -> np.ndarray:
    """Compute base robot pose initialization (q_init_base).
    This is a shared helper function used by both single and parallel processing.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses in format [qw, qx, qy, qz, x, y, z]
        constants: Task constants
        retargeter: Optional retargeter instance (needed for climbing)
    Returns:
        q_init_base in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints
    """
    if task_type == "robot_only":
        if data_format in ("lafan", "nokov", "pommel"):
            root_tracking_joint = "Chest2" if data_format == "pommel" else "Spine1"
            spine_joint_idx = constants.DEMO_JOINTS.index(root_tracking_joint)
            human_quat_init = estimate_human_orientation(human_joints, constants.DEMO_JOINTS)
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate(
                [human_joints[0, spine_joint_idx, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)]
            )
        else:  # smplh
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "object_interaction":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "climbing":
        if retargeter is None:
            raise ValueError("retargeter is required for climbing task")
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        root_tracking_joint = "Chest2" if data_format == "pommel" else "Spine1"
        spine_joint_idx = retargeter.demo_joints.index(root_tracking_joint)
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate(
            [
                human_joints[0, spine_joint_idx],
                human_quat_init,
                np.zeros(constants.ROBOT_DOF),
            ]
        )
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return q_init_base


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert object poses from [qw, qx, qy, qz, x, y, z] to MuJoCo order [x, y, z, qw, qx, qy, qz].
    Args:
        object_poses: Object poses array of shape (T, 7) in format [qw, qx, qy, qz, x, y, z]
    Returns:
        Object poses array in MuJoCo order [x, y, z, qw, qx, qy, qz]
    """
    return object_poses[:, [4, 5, 6, 0, 1, 2, 3]]


def build_retargeter_kwargs_from_config(
    retargeter_config: RetargeterConfig,
    constants: SimpleNamespace,
    object_urdf_path: str | None,
    task_type: str,
) -> dict:
    """Build kwargs for InteractionMeshRetargeter from a RetargeterConfig.
    This is a convenience function that allows building kwargs directly from
    a RetargeterConfig without needing a full RetargetingConfig.
    Args:
        retargeter_config: Retargeter configuration
        constants: Task constants
        object_urdf_path: Path to object URDF file
        task_type: Type of task
    Returns:
        Dictionary of kwargs for InteractionMeshRetargeter
    """
    kwargs = {
        "task_constants": constants,
        "object_urdf_path": object_urdf_path,
        "q_a_init_idx": retargeter_config.q_a_init_idx,
        "activate_joint_limits": retargeter_config.activate_joint_limits,
        "activate_obj_non_penetration": retargeter_config.activate_obj_non_penetration,
        "activate_foot_sticking": retargeter_config.activate_foot_sticking,
        "activate_foot_support_z": (
            retargeter_config.activate_foot_support_z
            and task_type == "climbing"
            and getattr(constants, "FIXED_OBJECT", False)
        ),
        "foot_support_ground_height": retargeter_config.foot_support_ground_height,
        "foot_support_z_soft_weight": retargeter_config.foot_support_z_soft_weight,
        "foot_support_z_hard_tolerance": retargeter_config.foot_support_z_hard_tolerance,
        "foot_lock": retargeter_config.foot_lock,
        "hand_sticking": retargeter_config.hand_sticking,
        "hand_tracking_point_offset": retargeter_config.hand_tracking_point_offset,
        "arm_straightness": retargeter_config.arm_straightness,
        "penetration_tolerance": retargeter_config.penetration_tolerance,
        "foot_sticking_tolerance": retargeter_config.foot_sticking_tolerance,
        "self_collision": retargeter_config.self_collision,
        "step_size": retargeter_config.step_size,
        "visualize": retargeter_config.visualize,
        "debug": retargeter_config.debug,
        "w_nominal_tracking_init": retargeter_config.w_nominal_tracking_init,
    }
    if task_type == "climbing":
        kwargs["nominal_tracking_tau"] = retargeter_config.nominal_tracking_tau
    return kwargs


def initialize_robot_pose(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter,
    task_config: TaskConfig,
    augmentation: bool,
    save_dir: Path,
    task_name: str,
    augmentation_translation: np.ndarray | None = None,
    augmentation_rotation: float | None = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize robot pose (q_init, q_nominal) based on task.
    Returns qpos in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints.
    Object poses are returned in MuJoCo order: [0:3] position, [3:7] quaternion.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses (assumed to be in format: [quat, pos] or [pos, quat])
        constants: Task constants
        retargeter: Retargeter instance
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        save_dir: Save directory path
        task_name: Task name
        augmentation_translation: Translation vector for augmentation (default: [0.2, 0.0, 0.0])
    Returns:
        Tuple of (q_init, q_nominal, object_poses_augmented, human_joints_modified, object_poses_modified)
        where qpos is in MuJoCo order and object_poses are in MuJoCo order
    """
    # Use default if not provided
    if augmentation_translation is None:
        augmentation_translation = _AUGMENTATION_TRANSLATION
    logger.info("Initializing robot pose")

    if task_type == "robot_only":
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    if task_type == "object_interaction":
        if augmentation:
            object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
            object_poses_augmented = augment_object_poses(
                object_poses,
                object_moving_frame_idx,
                human_joints[0, 0, :],
                augmentation_translation,
                augmentation_rotation,
            )
            # Convert object_poses to MuJoCo order
            object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
            object_poses = convert_object_poses_to_mujoco_order(object_poses)

            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            return q_nominal[0], q_nominal, object_poses_augmented, human_joints, object_poses
        object_poses_augmented = object_poses.copy()
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
        return q_init, None, object_poses_augmented, human_joints, object_poses

    if task_type == "climbing":
        if getattr(constants, "FIXED_OBJECT", False):
            if augmentation:
                raise ValueError("Augmentation is not supported for fixed NOKOV objects")
            q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, retargeter)
            target_pose = np.asarray(constants.FIXED_OBJECT_TARGET_POSE, dtype=float)
            object_poses_augmented = np.tile(target_pose, (len(object_poses), 1))
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
            return q_init, None, object_poses_augmented, human_joints, object_poses
        if augmentation:
            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            # Convert object_poses to MuJoCo order
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            return q_nominal[0], q_nominal, object_poses, human_joints, object_poses
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, retargeter)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    raise ValueError(f"Unknown task type: {task_type}")


def determine_output_path(
    task_type: TaskType,
    save_dir: Path,
    task_name: str,
    augmentation: bool,
) -> str:
    """Determine output file path based on task and augmentation.
    Args:
        task_type: Type of task
        save_dir: Save directory path
        task_name: Task name
        augmentation: Whether this is an augmentation run
    Returns:
        Output file path
    """
    if task_type == "robot_only":
        return str(save_dir / f"{task_name}.npz")
    if task_type in ("object_interaction", "climbing"):
        suffix = "_augmented" if augmentation else "_original"
        return str(save_dir / f"{task_name}{suffix}.npz")
    raise ValueError(f"Unknown task type: {task_type}")


# ----------------------------- Main -----------------------------


def main(cfg: RetargetingConfig) -> None:
    """Main retargeting pipeline.
    Args:
        cfg: Configuration arguments
    """
    # Validate configuration
    validate_config(cfg)

    robot = cfg.robot
    task_name = cfg.task_name
    task_type = cfg.task_type

    # Set defaults based on task type
    data_format: str = cfg.data_format or DEFAULT_DATA_FORMATS[task_type]
    save_dir = cfg.save_dir if cfg.save_dir is not None else Path(DEFAULT_SAVE_DIRS[task_type].format(robot=robot))
    data_path = cfg.data_path

    os.makedirs(save_dir, exist_ok=True)
    logger.info("Task: %s, Type: %s, Format: %s", task_name, task_type, data_format)
    logger.info("Data path: %s, Save dir: %s", data_path, save_dir)

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = replace(cfg.robot_config, robot_type=robot)

    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = replace(cfg.motion_data_config, data_format=data_format, robot_type=robot)

    if robot == "r1":
        hands = cfg.retargeter.hand_sticking
        if hands.robot_link_names == ("left_rubber_hand_link", "right_rubber_hand_link"):
            hands = replace(hands, robot_link_names=("left_wrist_roll_link", "right_wrist_roll_link"),
                            point_offset=(0.0, 0.0, 0.0))
        if data_format == "nokov" and hands.demo_joint_names == ("L_Wrist", "R_Wrist"):
            hands = replace(hands, demo_joint_names=("LeftHand", "RightHand"))
        if data_format == "pommel" and hands.demo_joint_names == ("L_Wrist", "R_Wrist"):
            hands = replace(hands, demo_joint_names=("LeftWrist", "RightWrist"))
        cfg.retargeter = replace(cfg.retargeter, hand_sticking=hands)

    # Fixed pommel/NOKOV tasks use the reusable in-package mushroom asset. Pose-specific MuJoCo
    # scenes live in the model cache, so save_dir contains only final NPZ files.
    if task_type == "climbing" and data_format in ("nokov", "pommel"):
        object_name = cfg.task_config.object_name or "mushroom"
        object_model_dir = MODELS_ROOT / object_name
        cfg.task_config = replace(
            cfg.task_config,
            object_name=object_name,
            object_mesh=cfg.task_config.object_mesh or object_model_dir / "mushroom_visual.obj",
            object_dir=object_model_dir / "generated_scenes" / task_name,
            object_scale=(
                cfg.task_config.pommel_reference_scale
                if data_format == "pommel"
                else cfg.task_config.object_scale
            ),
        )
    elif task_type == "climbing" and cfg.task_config.object_dir is None:
        cfg.task_config = replace(cfg.task_config, object_dir=data_path / task_name)

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=task_type,
    )

    # Load motion data
    human_joints, object_poses, smpl_scale = load_motion_data(
        task_type, data_format, data_path, task_name, constants, cfg.motion_data_config
    )
    if task_type == "climbing" and data_format == "nokov":
        human_joints, object_poses, _ = align_fixed_nokov_capture(
            human_joints,
            data_path,
            task_name,
            cfg.task_config,
        )
    elif task_type == "climbing" and data_format == "pommel":
        human_joints, object_poses = align_fixed_pommel_capture(human_joints, cfg.task_config)

    # Get toe names from motion data config (depends only on data_format)
    toe_names = cfg.motion_data_config.toe_names

    # Setup object data
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        task_type,
        constants,
        cfg.task_config.object_dir,
        smpl_scale,
        cfg.task_config,
        cfg.augmentation,
        object_poses=object_poses,
        object_scale_augmented=_OBJECT_SCALE_AUGMENTED,
        human_joints=human_joints,
    )

    # Create retargeter
    retargeter_kwargs = build_retargeter_kwargs_from_config(cfg.retargeter, constants, object_urdf_path, task_type)
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)
    logger.info("Retargeter created")

    # Preprocess motion data
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(human_joints, retargeter, toe_names, smpl_scale)
    elif task_type == "climbing" and getattr(constants, "FIXED_OBJECT", False):
        # Build the source interaction graph at the robot/human scale while
        # keeping the target mushroom at its physical scale in the static scene.
        human_joints = human_joints * smpl_scale
        object_poses = object_poses.copy()
        object_poses[:, 4:7] *= smpl_scale
    elif task_type in {"object_interaction", "climbing"}:
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            scale=smpl_scale,
            object_poses=object_poses,
        )

    # Initialize robot pose
    q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
        task_type,
        data_format,
        human_joints,
        object_poses,
        constants,
        retargeter,
        cfg.task_config,
        cfg.augmentation,
        save_dir,
        task_name,
        augmentation_translation=_AUGMENTATION_TRANSLATION,
    )

    # Extract foot sticking sequences
    fixed_object_foot_support = (
        task_type == "climbing"
        and getattr(constants, "FIXED_OBJECT", False)
        and cfg.retargeter.activate_foot_support_z
    )
    foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
        human_joints,
        retargeter.demo_joints,
        toe_names,
        velocity_threshold=cfg.retargeter.foot_contact_velocity_threshold,
        height_threshold=(
            cfg.retargeter.foot_contact_height_threshold if fixed_object_foot_support else None
        ),
        ground_height=cfg.retargeter.foot_support_ground_height,
    )
    if fixed_object_foot_support:
        # The detector consumes source-joint names such as LeftToeBase, but its
        # output uses canonical contact keys (currently L_Toe and R_Toe).
        # Count the returned keys instead of indexing with the input names.
        contact_keys = tuple(foot_sticking_sequences[0])
        support_counts = {
            contact_key: sum(frame[contact_key] for frame in foot_sticking_sequences)
            for contact_key in contact_keys
        }
        logger.info(
            "Fixed-object foot support frames (XY speed <= %.4f m/frame, Z <= %.3f m): %s",
            cfg.retargeter.foot_contact_velocity_threshold,
            cfg.retargeter.foot_support_ground_height
            + cfg.retargeter.foot_contact_height_threshold,
            support_counts,
        )

    # Task-specific foot sticking adjustments
    if task_type == "object_interaction":
        # Disable initial sticking
        foot_sticking_sequences[0][toe_names[0]] = False
        foot_sticking_sequences[0][toe_names[1]] = False

    # Determine output path
    dest_res_path = determine_output_path(task_type, save_dir, task_name, cfg.augmentation)

    # Retarget motion
    logger.info("Starting retargeting...")
    retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_poses=object_poses,
        object_poses_augmented=object_poses_augmented,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=q_nominal,
        original=not cfg.augmentation,
        dest_res_path=dest_res_path,
    )
    logger.info("Retargeting complete. Results saved to: %s", dest_res_path)

    if cfg.retargeter.debug:
        input("Press Enter to exit ...")


if __name__ == "__main__":
    cfg = tyro.cli(RetargetingConfig)
    main(cfg)
