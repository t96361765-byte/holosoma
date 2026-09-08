"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int] | tuple[int, int, float]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Each window is (start, end) or (start, end, z_floor).
    If z_floor is given per-window, it overrides the global z_floor for that window.
    Example: {"L_Toe": [(30, 60, 0.15)], "R_Toe": [(10, 20), (80, 95, 0.30)]}"""

    z_floor: float = 0.0
    """Default floor height used by Z pinning constraints (overridden by per-window z)."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class HandStickingConfig:
    """Configuration for hand support sticking constraints."""

    enable: bool = False
    """Whether to detect ground-supported hands and lock their contact points."""

    demo_joint_names: tuple[str, str] = ("L_Wrist", "R_Wrist")
    """Left and right human wrist joint names used for contact detection."""

    robot_link_names: tuple[str, str] = ("left_rubber_hand_link", "right_rubber_hand_link")
    """Left and right robot links carrying the hand contact points."""

    point_offset: tuple[float, float, float] = (0.16, 0.0, 0.0)
    """Contact point in each robot hand link's local frame."""

    height_threshold: float = 0.01
    """Maximum human wrist height above the normalized floor, in meters."""

    velocity_threshold: float = 0.015
    """Maximum human wrist XYZ displacement per frame, in meters."""

    min_contact_frames: int = 3
    """Minimum number of consecutive detected frames required to accept a support segment."""

    tolerance: float = 1e-3
    """Symmetric XYZ tolerance around the fixed hand contact anchor, in meters."""


@dataclass(frozen=True)
class ArmStraightnessConfig:
    """Configuration for full-arm direction and straightness costs."""

    enable: bool = False
    """Whether both robot arm segments track the human shoulder-to-wrist direction."""

    contact_only: bool = False
    """Whether to apply the costs only on detected hand-support frames instead of all frames."""

    weight: float = 1
    """Quadratic weight for both segments against the full human arm direction."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    activate_foot_support_z: bool = True
    """Whether fixed-object climbing should connect detected support soles to the floor."""

    foot_contact_velocity_threshold: float = 0.01
    """Maximum processed toe XY displacement per frame for accepting support."""

    foot_contact_height_threshold: float = 0.06
    """Maximum processed toe height above the floor for accepting support."""

    foot_support_ground_height: float = 0.0
    """World Z of the floor used by sole-height constraints."""

    foot_support_z_soft_weight: float = 1000.0
    """Soft weight pulling every active sole sphere toward the floor."""

    foot_support_z_hard_tolerance: float = 1e-2
    """Hard Z tolerance for the lowest sphere of each supporting foot."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    hand_sticking: HandStickingConfig = field(default_factory=HandStickingConfig)
    """Configuration for detected hand support sticking constraints."""

    hand_tracking_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Local XYZ offset of the robot hand point matched to the human wrist.

    This is independent of hand sticking. For the G1 fist-pan model, (0.16, 0, 0)
    makes the point on the end of the fist-pan track the captured wrist position.
    """

    arm_straightness: ArmStraightnessConfig = field(default_factory=ArmStraightnessConfig)
    """Configuration for independently straightening each arm, optionally only during hand support."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
