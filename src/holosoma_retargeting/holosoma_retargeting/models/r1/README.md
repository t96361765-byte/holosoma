# R1 for OmniRetarget

Prepared from `D:/track_dataset/models/r1/R1.urdf` with `data_utils/prepare_r1.py`.
The source URDF matches the official Unitree `unitree_ros/robots/r1_description/R1.urdf`
Git blob `a4eda710a294fb7ab009a64e0871b9536b0dc819` (checked 2026-09-08).
Source: https://github.com/unitreerobotics/unitree_ros/tree/master/robots/r1_description

Use `--robot r1 --data-format nokov` with `examples/robot_retarget.py`.
This package contains `r1_26dof.urdf`, the matching floating-base
`r1_26dof.xml`, and all referenced meshes. Original source assets are unchanged.

- 26 revolute joints: 12 leg, 2 waist, 10 arm, 2 head.
- The source calls each last arm joint `wrist_roll_joint`; there is no wrist pitch/yaw.
- Floating-base output has 33 qpos values and 32 velocity coordinates.
- Head pitch/yaw remain at zero through equal joint bounds in the R1 config.
  Keep joint-limit enforcement enabled. Head coordinates are qpos 31 and 32.
- NOKOV parsing, Y-up to Z-up conversion, solver and NPZ writing are shared with G1.
- NOKOV initial human scaling is 1.2302 / 1.75; this is an anthropometric starting estimate.
- Pommel uses the existing metre-valued Z-up parser without axis swapping or extra human scaling.
  Its mapping shares G1 anatomical points, with R1 pelvis, wrists, ankles and toe frames.
- Hand Laplacian points are the wrist-roll origins (local offset defaults to zero).
- Four sole frames and one toe frame per foot are added without geometry or mass.
  Existing R1 foot collision meshes are retained; no G1 foot spheres are copied.
- The original fixed ankle linkage branches remain a simplified open-tree model.
- Default self-collision and extra arm/hand support objectives remain disabled.

NOKOV climbing uses its capture's mushroom CSV and `--task-config.object-scale`.
Pommel/Blender-specific placement flags do not configure the NOKOV capture.
Pommel climbing supports the shared mushroom reference pose, scale, Z compression
and human radial offset flags. Head pitch/yaw remain fixed at zero in both formats.
This implementation does not add a G1-compatible R1 50 Hz training-data converter.

Regenerate from the package directory:

```powershell
& D:\anaconda3\envs\omniretarget\python.exe .\data_utils\prepare_r1.py --source D:\track_dataset\models\r1\R1.urdf
```
