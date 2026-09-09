"""Configuration types for retargeting task settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    """Task-specific configuration parameters.

    These parameters control task-specific behavior like ground meshgrid generation,
    object sampling, augmentation, and scaling. Can be overridden via CLI.
    """

    # Object name
    # Auto-determined based on task_type if None: "largebox" for object_interaction,
    # "multi_boxes" for climbing, "ground" for robot_only
    object_name: str | None = None

    # Ground meshgrid (robot_only task)
    ground_size: int = 15
    ground_range: tuple[float, float] = (-1.0, 1.0)

    # Climbing ground meshgrid (climbing task)
    climbing_ground_size: int = 8
    climbing_ground_range: tuple[float, float] = (-2.0, 2.0)

    # Surface weight parameters for climbing object sampling (climbing task)
    # Used in weighted_surface_sampling: points with z-coordinate > threshold get high weight
    # This biases sampling toward top surfaces (important for climbing contact points)
    surface_weight_threshold: float = 0.9  # z-coordinate threshold for high-weight points
    surface_weight_high: int = 20  # Weight for top surface points (z > threshold)
    surface_weight_low: int = 1  # Weight for other points

    # Object directory (for climbing tasks)
    # Auto-determined from data_path / task_name if None
    object_dir: Path | None = None

    # Fixed climbing object inputs. For NOKOV, object_pose_file defaults to
    # <capture-dir>/<task-name>-mushroom.csv and object_mesh defaults to the
    # reusable models/mushroom/mushroom_visual.obj asset when omitted.
    object_pose_file: Path | None = None
    object_mesh: Path | None = None
    object_scale: float = 1.0
    fixed_object_surface_points_enabled: bool = False
    """Whether a fixed object's surface samples participate in the interaction mesh.

    Disabled by default so the mushroom is represented only by its physical
    collision constraint. The sparse ground patch remains in the graph.
    """
    fixed_object_sample_count: int = 50

    # Blender reference pose for marker53 pommel captures. This is the pose of
    # mushroom_visual in real_pommel.blend before the whole human/object pair
    # is translated to put the mushroom axis at XY=0 and its base at Z=0.
    pommel_reference_position: tuple[float, float, float] = (
        0.02315461076796055,
        -0.6467866897583008,
        0.24471218883991241,
    )
    pommel_reference_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    pommel_reference_scale: float = 0.8005122542381287
    pommel_z_compression: float = 1.0
    """Additional vertical compression ratio for the pommel mesh.

    A value of 1.1 keeps X/Y unchanged and divides the Z scale by 1.1.  The
    compressed mesh is regrounded afterwards, so its base remains at Z=0.
    """
    pommel_human_radial_offset: float = 0.0
    """Horizontal human-only offset away from the mushroom axis, in metres.

    The outward direction is inferred from the first-frame Hips position after
    the mushroom has been recentered to XY=(0, 0).  Z and mushroom pose are not
    changed.  Negative values move the human toward the mushroom.
    """

    # Optional fixed-object contact sampling. These settings are used only when
    # fixed_object_surface_points_enabled is true.
    # For mushroom_visual.obj, 0.72 gives local Z >= 0.121657 m,
    # above the column and lower rim. The cutoff scales with the asset.
    fixed_object_upper_height_fraction: float = 0.72
    fixed_object_upper_normal_z: float = 0.25
    fixed_object_upper_interior_radius: float = 1.0
    fixed_object_sampling_seed: int = 42

    # Sparse ground patch for fixed-object interaction graphs. A 5 x 4 grid
    # contributes 20 points while the mushroom itself remains collision-only.
    fixed_object_ground_shape: tuple[int, int] = (5, 4)
    fixed_object_ground_range: tuple[float, float] = (-1.2, 1.2)
