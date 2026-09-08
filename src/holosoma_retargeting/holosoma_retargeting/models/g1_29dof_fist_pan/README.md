# G1 29-DoF fist-pan model

This model is a retargeting-compatible derivative of `../g1/g1_29dof`.
It preserves the original Holosoma G1 kinematic tree, semantic foot links,
joint limits, and non-hand collision geometry. Only the two rubber-hand bodies
use the fist-pan meshes, inertial parameters, and cylindrical collision shapes
from `D:/track_dataset/models/g1_29dof_fist_pan`.

The fist-pan STL files are metre-scaled in the source asset set and therefore
must not receive a `0.001` mesh scale.

Hand collision geometry:

- cylinder axis: local +X
- cylinder centre: `(0.085, 0, 0)` m
- radius: `0.035` m
- length: `0.15` m

The URDF and MJCF keep the existing link names `left_rubber_hand_link` and
`right_rubber_hand_link` so the current G1 joint mappings work unchanged.
The legacy thumb/pinky marker bodies are omitted, as the pan cylinder is the
complete hand collision shape in the source model.
