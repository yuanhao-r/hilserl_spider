"""FK/IK and base/tool transforms for Tianji right-arm control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from franka_env.envs.tianji.backend import _TELEOP_ROOT, inject_tianji_paths

inject_tianji_paths()

_MARVIN_IMPORT_ERROR = None
_DUAL_IK_IMPORT_ERROR = None

try:
    from marvin_arm_controller import MarvinArmController
    from utils.kine_model import transform_to_wxyzxyz, wxyzxyz_to_transform
except Exception as exc:  # pragma: no cover - runtime dependency
    _MARVIN_IMPORT_ERROR = exc
    MarvinArmController = None
    transform_to_wxyzxyz = None
    wxyzxyz_to_transform = None

try:
    from utils.marvin_dual_arm_waist_teleop import DualArmWaistIK
except Exception as exc: 
    raise ImportError # pragma: no cover - runtime dependency
    _DUAL_IK_IMPORT_ERROR = exc
    DualArmWaistIK = None


def pose6_to_transform(pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float64).reshape(6)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = arr[:3]
    tf[:3, :3] = Rotation.from_euler("xyz", arr[3:]).as_matrix()
    return tf


def transform_to_pose6(tf: np.ndarray) -> np.ndarray:
    mat = np.asarray(tf, dtype=np.float64).reshape(4, 4)
    euler = Rotation.from_matrix(mat[:3, :3]).as_euler("xyz")
    return np.concatenate([mat[:3, 3], euler]).astype(np.float64)


@dataclass
class TianjiKinematicsConfig:
    robot_ip: str = "10.10.13.10"
    marvin_urdf_path: str = str(
        (
            _TELEOP_ROOT
            / "test_teleop/demo/urdf/MarvinCCS/urdf/Marvin M6-S-L-CCS-696-V3.1 urdf.urdf"
        ).resolve()
    )
    dual_ik_urdf_path: str = str(
        (_TELEOP_ROOT / "test_teleop/models/spiderrobot/robot_marvin_arms.urdf").resolve()
    )
    right_arm_base_pose_wxyzxyz: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.707105, 0.707108, -0.000005, 0.000005, 0.319484, -0.012501, 1.127505],
            dtype=np.float64,
        )
    )
    left_arm_base_pose_wxyzxyz: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.707108, -0.707105, -0.000005, -0.000005, 0.319484, 0.012499, 1.127505],
            dtype=np.float64,
        )
    )
    head_base_pose_wxyzxyz: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.354649, -0.045577, 1.239027, -0.313173, 0.3, 0.0, 1.1],
            dtype=np.float64,
        )
    )
    tool_z_offset: float = 0.2
    head_z_offset: float = 0.2


class TianjiKinematics:
    def __init__(self, config: TianjiKinematicsConfig | None = None):
        if _MARVIN_IMPORT_ERROR is not None:
            raise ImportError(
                "Failed to import Tianji FK dependencies from test_teleop/demo."
            ) from _MARVIN_IMPORT_ERROR
        if _DUAL_IK_IMPORT_ERROR is not None:
            raise ImportError(
                "Failed to import Tianji IK dependencies from test_teleop/python."
            ) from _DUAL_IK_IMPORT_ERROR

        self.config = config or TianjiKinematicsConfig()
        self.controller = MarvinArmController(
            robot_ip=self.config.robot_ip,
            test_mode=False,
            urdf_path=self.config.marvin_urdf_path,
        )
        self.controller.init_kinematics()

        self.full_ik_solver = DualArmWaistIK(
            urdf_path=self.config.dual_ik_urdf_path,
            enable_viewer=True,
        )
        self.full_ik_solver.set_scale(1.0, 1.0)
        self.full_ik_solver.start()

        self.base_right_tf = wxyzxyz_to_transform(self.config.right_arm_base_pose_wxyzxyz)
        self.base_left_tf = wxyzxyz_to_transform(self.config.left_arm_base_pose_wxyzxyz)
        self.base_head_tf = wxyzxyz_to_transform(self.config.head_base_pose_wxyzxyz)

        self.tool_tf = np.eye(4, dtype=np.float64)
        self.tool_tf[2, 3] = float(self.config.tool_z_offset)
        self.body_waist_q = np.zeros(2, dtype=np.float64)
        self.right_target_pose = np.eye(4, dtype=np.float64)
        self.left_target_pose = None
        self.head_target_pose = self.base_head_tf.copy()
    
    # tianji controller 和全身的fk 算出来有90°偏差+9cm的误差。没对上就是
    def fk_right_pose6_old(self, right_joints_rad: np.ndarray) -> np.ndarray:
        right_fk = self.controller.compute_fk(np.asarray(right_joints_rad, dtype=np.float64).reshape(7))
        if right_fk is None:
            raise RuntimeError("Failed to compute Tianji right-arm FK.")
        return transform_to_pose6(self.base_right_tf @ right_fk @ self.tool_tf)
    
    def fk_right_pose6(self, right_joints_rad: np.ndarray) -> np.ndarray:
        current_cfg = np.concatenate([right_joints_rad, [0.]*7])
        wxyzxyz = self.full_ik_solver.compute_fk_wxyzxyz(current_cfg, "R_tcp")
        return transform_to_pose6(wxyzxyz_to_transform(wxyzxyz))

    def update_locked_left_head(self, left_joints_rad: np.ndarray) -> None:
        if self.left_target_pose is None:
            left_fk = self.controller.compute_fk(
                np.asarray(left_joints_rad, dtype=np.float64).reshape(7)
            )
            if left_fk is None:
                raise RuntimeError("Failed to compute Tianji left-arm FK.")
            self.left_target_pose = self.base_left_tf @ left_fk @ self.tool_tf
        self.head_target_pose = self.base_head_tf.copy()

    def pose_from_right_joints_deg(self, right_joints_deg: np.ndarray) -> np.ndarray:
        return self.fk_right_pose6(np.deg2rad(np.asarray(right_joints_deg, dtype=np.float64).reshape(7)))

    def sync_ik_seed(self, left_rad: np.ndarray, right_rad: np.ndarray) -> None:
        current_cfg = np.concatenate(
            [np.asarray(right_rad, dtype=np.float64).reshape(7), np.asarray(left_rad, dtype=np.float64).reshape(7)]
        )
        if hasattr(self.full_ik_solver, "sync_with_current_cfg"):
            self.full_ik_solver.sync_with_current_cfg(current_cfg, reset_delta_reference=True)

    def compute_full_body_ik(
        self,
        current_left_rad: np.ndarray,
        current_right_rad: np.ndarray,
        force_seed_from_current: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        current_left_rad = np.asarray(current_left_rad, dtype=np.float64).reshape(7)
        current_right_rad = np.asarray(current_right_rad, dtype=np.float64).reshape(7)
        self.update_locked_left_head(current_left_rad)

        left_target = transform_to_wxyzxyz(self.left_target_pose)
        right_target = transform_to_wxyzxyz(self.right_target_pose)
        head_target = transform_to_wxyzxyz(self.head_target_pose)

        left_pose_xyzwxyz = np.concatenate([left_target[4:7], left_target[0:4]])
        right_pose_xyzwxyz = np.concatenate([right_target[4:7], right_target[0:4]])
        head_pose_xyzwxyz = np.concatenate([head_target[4:7], head_target[0:4]])
        head_pose_xyzwxyz[2] += float(self.config.head_z_offset)

        current_cfg = np.concatenate([self.body_waist_q, current_right_rad, current_left_rad])
        if force_seed_from_current:
            self.sync_ik_seed(current_left_rad, current_right_rad)
            ik_seed_cfg = None
        else:
            ik_seed_cfg = current_cfg[2:]

        # print("final pose: ", right_pose_xyzwxyz)
        solver_cfg, ik_error = self.full_ik_solver.compute_ik(
            ik_seed_cfg,
            head_pose_xyzwxyz,
            right_pose_xyzwxyz,
            left_pose_xyzwxyz,
            force_seed_from_current=force_seed_from_current,
        )
        if solver_cfg is None:
            return None, None, None

        solver_cfg = np.concatenate([current_cfg[:2], solver_cfg])
        right_target_rad = solver_cfg[2:9]
        left_target_rad = solver_cfg[9:]
        return left_target_rad, right_target_rad, ik_error

    def solve_right_pose(
        self,
        target_pose6: np.ndarray,
        current_left_rad: np.ndarray,
        current_right_rad: np.ndarray,
        force_seed_from_current: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        self.right_target_pose = pose6_to_transform(target_pose6)
        return self.compute_full_body_ik(
            current_left_rad=current_left_rad,
            current_right_rad=current_right_rad,
            force_seed_from_current=force_seed_from_current,
        )

    def close(self) -> None:
        if self.controller is not None and hasattr(self.controller, "cleanup"):
            self.controller.cleanup()
