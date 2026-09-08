"""Utilities for static NOKOV rigid bodies used by climbing tasks.

These helpers consume the synchronized rigid-body CSV and a Z-up human joint
trajectory produced by :mod:`holosoma_retargeting.data_utils.nokov_bvh`.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R  # type: ignore[import-untyped]  # noqa: N817


_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)


def _nokov_segment_rows(csv_path: Path) -> tuple[dict[str, int], list[list[str]]]:
    """Return the named columns and numeric rows from a NOKOV rigid-body CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"NOKOV rigid-body CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_file:
        rows = list(csv.reader(csv_file))

    segment_data_index = next(
        (index for index, row in enumerate(rows) if row and row[0].strip() == "[SegmentData]"),
        None,
    )
    if segment_data_index is None or segment_data_index + 3 >= len(rows):
        raise ValueError(f"Invalid NOKOV rigid-body CSV (missing SegmentData): {csv_path}")

    header = [value.strip() for value in rows[segment_data_index + 2]]
    columns = {name: index for index, name in enumerate(header) if name}
    required = (
        "XToGlobal1",
        "YToGlobal1",
        "ZToGlobal1",
        "QxToGlobal1",
        "QyToGlobal1",
        "QzToGlobal1",
        "QwToGlobal1",
    )
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"NOKOV rigid-body CSV is missing columns {missing}: {csv_path}")

    data_rows = [
        row
        for row in rows[segment_data_index + 3 :]
        if row and row[0].strip().isdigit()
    ]
    if not data_rows:
        raise ValueError(f"NOKOV rigid-body CSV has no data rows: {csv_path}")
    return columns, data_rows


def load_static_nokov_pose(csv_path: Path) -> tuple[np.ndarray, int]:
    """Load one robust static pose as ``[qw, qx, qy, qz, x, y, z]``.

    NOKOV translations are millimetres in Y-up coordinates and its quaternion
    fields are XYZW.  The returned position and orientation are metres and
    Z-up, using the project's ``[x, y, z] -> [x, -z, y]`` convention.
    """
    columns, data_rows = _nokov_segment_rows(csv_path)
    positions_y_up = np.asarray(
        [
            [float(row[columns[f"{axis}ToGlobal1"]]) for axis in "XYZ"]
            for row in data_rows
        ],
        dtype=float,
    ) / 1000.0
    quaternions_xyzw = np.asarray(
        [
            [
                float(row[columns["QxToGlobal1"]]),
                float(row[columns["QyToGlobal1"]]),
                float(row[columns["QzToGlobal1"]]),
                float(row[columns["QwToGlobal1"]]),
            ]
            for row in data_rows
        ],
        dtype=float,
    )

    positions_z_up = positions_y_up @ _Y_UP_TO_Z_UP.T
    basis_rotation = R.from_matrix(_Y_UP_TO_Z_UP)
    rotations_z_up = basis_rotation * R.from_quat(quaternions_xyzw) * basis_rotation.inv()
    converted_xyzw = rotations_z_up.as_quat()
    converted_xyzw[converted_xyzw @ converted_xyzw[0] < 0] *= -1
    median_xyzw = np.median(converted_xyzw, axis=0)
    median_xyzw /= np.linalg.norm(median_xyzw)

    position = np.median(positions_z_up, axis=0)
    pose = np.concatenate([median_xyzw[[3, 0, 1, 2]], position])
    return pose, len(data_rows)


def level_static_pose(pose_wxyz_xyz: np.ndarray) -> np.ndarray:
    """Remove marker-frame roll/pitch while preserving the measured yaw."""
    pose = np.asarray(pose_wxyz_xyz, dtype=float)
    if pose.shape != (7,):
        raise ValueError("pose_wxyz_xyz must have shape (7,)")

    rotation = R.from_quat(pose[[1, 2, 3, 0]])
    local_x_world = rotation.apply(np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(local_x_world[:2]) < 1e-8:
        raise ValueError("Cannot preserve yaw when the object's local X axis is vertical")
    yaw = float(np.arctan2(local_x_world[1], local_x_world[0]))
    leveled_xyzw = R.from_euler("z", yaw).as_quat()

    leveled = pose.copy()
    leveled[:4] = leveled_xyzw[[3, 0, 1, 2]]
    return leveled


def ground_and_recenter_static_pose(
    pose_wxyz_xyz: np.ndarray,
    mesh: trimesh.Trimesh,
    scale: float | tuple[float, float, float] | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Put object XY at zero and its rotated mesh minimum on the ground.

    Returns the placed pose and the single world translation that must also be
    applied to the human joints.  Applying that same translation preserves the
    complete human-object relative position exactly.
    """
    scale_xyz = np.asarray(scale, dtype=float)
    if scale_xyz.ndim == 0:
        scale_xyz = np.repeat(scale_xyz, 3)
    if scale_xyz.shape != (3,) or not np.all(np.isfinite(scale_xyz)) or np.any(scale_xyz <= 0):
        raise ValueError("Object scale must be one positive finite value or three XYZ values")
    pose = np.asarray(pose_wxyz_xyz, dtype=float)
    if pose.shape != (7,):
        raise ValueError("pose_wxyz_xyz must have shape (7,)")

    rotation = R.from_quat(pose[[1, 2, 3, 0]])
    relative_vertices = rotation.apply(np.asarray(mesh.vertices, dtype=float) * scale_xyz)
    grounded_origin_z = -float(relative_vertices[:, 2].min())
    target_position = np.array([0.0, 0.0, grounded_origin_z])
    world_shift = target_position - pose[4:7]

    placed = pose.copy()
    placed[4:7] = target_position
    return placed, world_shift


def translate_human_joints(human_joints_z_up: np.ndarray, world_shift: np.ndarray) -> np.ndarray:
    """Apply the fixed-object placement translation to Z-up human joints."""
    joints = np.asarray(human_joints_z_up, dtype=float)
    shift = np.asarray(world_shift, dtype=float)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("human_joints_z_up must have shape (T, J, 3)")
    if shift.shape != (3,):
        raise ValueError("world_shift must have shape (3,)")
    return joints + shift


def _absolute_asset_paths(root: ET.Element, source_xml: Path) -> None:
    """Make file-backed MuJoCo assets independent of the output XML folder."""
    compiler = root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", ".")) if compiler is not None else Path(".")
    texture_dir = Path(compiler.get("texturedir", ".")) if compiler is not None else Path(".")

    for asset in root.findall("./asset/*"):
        file_name = asset.get("file")
        if not file_name:
            continue
        file_path = Path(file_name)
        if file_path.is_absolute():
            continue
        base_dir = texture_dir if asset.tag == "texture" else mesh_dir
        asset.set("file", (source_xml.parent / base_dir / file_path).resolve().as_posix())

    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)


def create_static_mesh_scene(
    robot_xml: Path,
    object_mesh: Path,
    output_xml: Path,
    pose_wxyz_xyz: np.ndarray,
    scale: float | tuple[float, float, float] | np.ndarray = 1.0,
    object_name: str = "mushroom",
    collision_mesh: Path | None = None,
) -> Path:
    """Create a MuJoCo scene with a visual mesh and fixed collision volume."""
    if not robot_xml.is_file():
        raise FileNotFoundError(f"Robot MuJoCo XML not found: {robot_xml}")
    if not object_mesh.is_file():
        raise FileNotFoundError(f"Object mesh not found: {object_mesh}")
    if collision_mesh is not None and not collision_mesh.is_file():
        raise FileNotFoundError(f"Object collision mesh not found: {collision_mesh}")

    tree = ET.parse(robot_xml)
    root = tree.getroot()
    _absolute_asset_paths(root, robot_xml)
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Robot XML must contain asset and worldbody elements: {robot_xml}")

    mesh_name = f"{object_name}_static_mesh"
    material_name = f"{object_name}_static_material"
    texture_file = object_mesh.with_name(f"{object_name}_texture_0.png")
    scale_xyz = np.asarray(scale, dtype=float)
    if scale_xyz.ndim == 0:
        scale_xyz = np.repeat(scale_xyz, 3)
    if scale_xyz.shape != (3,) or not np.all(np.isfinite(scale_xyz)) or np.any(scale_xyz <= 0):
        raise ValueError("Object scale must be one positive finite value or three XYZ values")
    scale_text = " ".join(f"{value:.12g}" for value in scale_xyz)
    ET.SubElement(
        asset,
        "mesh",
        name=mesh_name,
        file=object_mesh.resolve().as_posix(),
        scale=scale_text,
    )
    collision_mesh_name = mesh_name
    if collision_mesh is not None:
        collision_mesh_name = f"{object_name}_upper_collision_mesh"
        ET.SubElement(
            asset,
            "mesh",
            name=collision_mesh_name,
            file=collision_mesh.resolve().as_posix(),
            scale=scale_text,
        )
    material_attributes = {
        "name": material_name,
        "rgba": "0.72 0.55 0.34 1",
    }
    if texture_file.is_file():
        texture_name = f"{object_name}_static_texture"
        ET.SubElement(
            asset,
            "texture",
            name=texture_name,
            type="2d",
            file=texture_file.resolve().as_posix(),
        )
        material_attributes.update(texture=texture_name, rgba="1 1 1 1")
    ET.SubElement(asset, "material", **material_attributes)

    pose = np.asarray(pose_wxyz_xyz, dtype=float)
    position_text = " ".join(f"{value:.12g}" for value in pose[4:7])
    quaternion_text = " ".join(f"{value:.12g}" for value in pose[:4])
    body = ET.SubElement(
        worldbody,
        "body",
        name=f"{object_name}_static",
        pos=position_text,
        quat=quaternion_text,
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{object_name}_visual",
        type="mesh",
        mesh=mesh_name,
        material=material_name,
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        body,
        "geom",
        name=object_name,
        type="mesh",
        mesh=collision_mesh_name,
        rgba="0 0 0 0",
        contype="1",
        conaffinity="1",
        friction="0.9 0.02 0.001",
    )

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)
    return output_xml


def create_upper_surface_collision_mesh(
    object_mesh: Path,
    output_mesh: Path,
    *,
    height_fraction: float,
    normal_z: float,
    interior_radius: float,
) -> Path:
    """Create a closed convex collision volume from an upper support cap."""

    mesh = trimesh.load(object_mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Object asset is not a non-empty triangle mesh: {object_mesh}")
    centers = mesh.triangles_center
    bounds = mesh.bounds
    height_span = bounds[1, 2] - bounds[0, 2]
    half_extent = (bounds[1, :2] - bounds[0, :2]) / 2.0
    xy_center = (bounds[0, :2] + bounds[1, :2]) / 2.0
    normalized_height = (centers[:, 2] - bounds[0, 2]) / height_span
    normalized_radius = np.linalg.norm((centers[:, :2] - xy_center) / half_extent, axis=1)
    upper_faces = (
        (normalized_height >= height_fraction)
        & (mesh.face_normals[:, 2] >= normal_z)
        & (normalized_radius <= interior_radius)
    )
    if not np.any(upper_faces):
        raise ValueError("No faces satisfy the fixed-object upper collision filters")
    vertex_indices = np.unique(mesh.faces[upper_faces])
    collision = trimesh.Trimesh(vertices=mesh.vertices[vertex_indices], process=False).convex_hull
    if not collision.is_watertight or collision.volume <= 0:
        raise ValueError("Generated upper collision mesh is not a closed positive-volume hull")
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    collision.export(output_mesh)
    return output_mesh


def create_static_mesh_urdf(
    object_mesh: Path,
    output_urdf: Path,
    scale: float | tuple[float, float, float] | np.ndarray = 1.0,
    object_name: str = "mushroom",
) -> Path:
    """Create the matching fixed-mesh URDF used by retargeting visualization."""
    try:
        mesh_filename = object_mesh.resolve().relative_to(output_urdf.parent.resolve()).as_posix()
    except ValueError:
        mesh_filename = object_mesh.resolve().as_posix()
    robot = ET.Element("robot", name=object_name)
    link = ET.SubElement(robot, "link", name=f"{object_name}_link")
    scale_xyz = np.asarray(scale, dtype=float)
    if scale_xyz.ndim == 0:
        scale_xyz = np.repeat(scale_xyz, 3)
    if scale_xyz.shape != (3,) or not np.all(np.isfinite(scale_xyz)) or np.any(scale_xyz <= 0):
        raise ValueError("Object scale must be one positive finite value or three XYZ values")
    mesh_attributes = {
        "filename": mesh_filename,
        "scale": " ".join(f"{value:.12g}" for value in scale_xyz),
    }
    for tag in ("visual", "collision"):
        element = ET.SubElement(link, tag)
        ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "mesh", **mesh_attributes)
    tree = ET.ElementTree(robot)
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    return output_urdf


def write_scene_metadata(output_path: Path, metadata: dict[str, object]) -> Path:
    """Write deterministic, human-readable placement metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
