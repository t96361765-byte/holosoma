#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


def load_npz(npz_path: str):
    with np.load(npz_path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"])
        fps = int(data["fps"]) if "fps" in data else 30
        metadata: dict[str, object] = {}
        if "object_urdf" in data:
            metadata["object_urdf"] = str(np.asarray(data["object_urdf"]).item())
        if "object_pose_in_qpos" in data:
            metadata["object_pose_in_qpos"] = bool(np.asarray(data["object_pose_in_qpos"]).item())
        if "object_position" in data:
            metadata["object_position"] = tuple(np.asarray(data["object_position"], dtype=float))
        if "object_quaternion_wxyz" in data:
            metadata["object_quaternion_wxyz"] = tuple(
                np.asarray(data["object_quaternion_wxyz"], dtype=float)
            )
    return qpos, fps, metadata


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    fps: int | None = None,
    object_metadata: dict[str, object] | None = None,
):
    """
    qpos layout (MuJoCo order):
      [0:3]   robot base position (xyz)
      [3:7]   robot base quat (wxyz)
      [7:7+R] robot joint positions (R = actuated dof)
      [end-7:end-4] (optional) object position (xyz)
      [end-4:end]   (optional) object quat (wxyz)

    We'll infer R from the robot URDF's actuated joints in ViserUrdf.
    """
    server = viser.ViserServer()

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    # URDFs (using yourdfpy so meshes show up)
    robot_urdf_y = yourdfpy.URDF.load(config.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    object_metadata = object_metadata or {}
    object_urdf = config.object_urdf or object_metadata.get("object_urdf")
    object_position = object_metadata.get("object_position", config.object_position)
    object_quaternion_wxyz = object_metadata.get(
        "object_quaternion_wxyz", config.object_quaternion_wxyz
    )

    vo = None
    if config.show_object:
        if object_urdf is None:
            raise ValueError(
                "--show-object requires object_urdf metadata in the NPZ or an explicit --object-urdf"
            )
        object_urdf_y = yourdfpy.URDF.load(str(object_urdf), load_meshes=True, build_scene_graph=True)
        vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

    # A tiny grid
    server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height, position=(0.0, 0.0, 0.0))

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)
    expected_robot_qpos = 7 + robot_dof
    if qpos.ndim != 2 or qpos.shape[1] < expected_robot_qpos:
        raise ValueError(
            f"qpos must have at least {expected_robot_qpos} columns for this robot; got {qpos.shape}"
        )
    metadata_pose_in_qpos = object_metadata.get("object_pose_in_qpos")
    object_pose_in_qpos = config.assume_object_in_qpos
    if metadata_pose_in_qpos is not None:
        object_pose_in_qpos = object_pose_in_qpos and bool(metadata_pose_in_qpos)
    contains_object_in_qpos = (
        config.show_object and object_pose_in_qpos and qpos.shape[1] >= expected_robot_qpos + 7
    )

    # Use fps from config if not provided, otherwise use the one from npz file
    actual_fps = fps if fps is not None else config.fps

    # Set initial mesh visibility
    vr.show_visual = config.show_meshes
    if vo is not None:
        vo.show_visual = config.show_meshes

    # ---------- Additional GUI controls (mesh visibility) ----------
    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=config.show_meshes)

    @show_meshes_cb.on_update
    def _(_):
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    # ---------- Use reusable motion control sliders from viser_utils ----------
    create_motion_control_sliders(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        viser_object=vo,
        object_base_frame=object_root if vo is not None else None,
        contains_object_in_qpos=contains_object_in_qpos,
        static_object_position=object_position,
        static_object_quaternion_wxyz=object_quaternion_wxyz,
        initial_fps=actual_fps,
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
    )
    n_frames = int(qpos.shape[0])
    object_mode = "none"
    if vo is not None:
        object_mode = "qpos" if contains_object_in_qpos else "static"
    print(f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | object={object_mode}")
    if object_mode == "static":
        print(
            "[viser_player] Static object pose | "
            f"position={object_position} | wxyz={object_quaternion_wxyz}"
        )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(cfg: ViserConfig) -> None:
    """Main function for viser player."""
    qpos, fps, object_metadata = load_npz(cfg.qpos_npz)
    make_player(
        config=cfg,
        qpos=qpos,
        fps=fps,
        object_metadata=object_metadata,
    )

    # keep process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
