"""Gym interface for Tianji MARVIN right arm on dual-arm humanoid platform."""

import copy
import os
import queue
import sys
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
from multiprocess import Process, Queue
from loop_rate_limiters import RateLimiter

from franka_env.camera.fisheye_capture import FisheyeCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.envs.tianji_ik_client import TianjiIKClient
from franka_env.envs.wow_skin import WowSkin
from scipy.linalg import expm

IMAGE_SIZE = (256, 256)
class ImageDisplayer(Process):
    def __init__(self, queue_obj, name):
        super().__init__()
        self.queue = queue_obj
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            frame = np.concatenate(
                [
                    cv2.resize(v, IMAGE_SIZE)
                    for k, v in img_array.items()
                    if "full" not in k
                ],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


class RealtimeForce6DPlotter:
    """Non-blocking realtime plotter for 6D force/torque samples."""

    DEFAULT_LABELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")

    def __init__(
        self,
        max_points: int = 100,
        update_interval: float = 0.05,
        title: str = "Realtime 6D Force",
        labels: Sequence[str] | None = None,
    ):
        self.max_points = int(max_points)
        self.update_interval = float(update_interval)
        self.title = title
        self.labels = tuple(labels) if labels is not None else self.DEFAULT_LABELS
        if len(self.labels) != 6:
            raise ValueError("labels must contain exactly 6 names")

        self._times = deque(maxlen=self.max_points)
        self._values = deque(maxlen=self.max_points)
        self._start_t = time.monotonic()
        self._last_draw_t = 0.0
        self._closed = False

        self._plt = None
        self._fig = None
        self._axes = None
        self._lines = None

    def add_force_point(self, force6d, timestamp: float | None = None) -> None:
        """Append one 6D sample and refresh the plot at the configured rate."""
        values = np.asarray(force6d, dtype=np.float64).reshape(-1)
        if values.shape[0] != 6:
            raise ValueError(f"force6d must have 6 values, got shape {values.shape}")

        sample_t = time.monotonic() if timestamp is None else float(timestamp)
        plot_t = sample_t - self._start_t if timestamp is None else sample_t
        self._times.append(plot_t)
        self._values.append(values.copy())

        now = time.monotonic()
        if now - self._last_draw_t >= self.update_interval:
            self._draw()
            self._last_draw_t = now

    def add_point(self, force6d, timestamp: float | None = None) -> None:
        """Alias for add_force_point()."""
        self.add_force_point(force6d, timestamp=timestamp)

    def reset(self) -> None:
        self._times.clear()
        self._values.clear()
        self._start_t = time.monotonic()
        self._last_draw_t = 0.0
        if self._lines is not None:
            for line in self._lines:
                line.set_data([], [])
            self._draw(force=True)

    def close(self) -> None:
        self._closed = True
        if self._plt is not None and self._fig is not None:
            self._plt.close(self._fig)

    def _ensure_plot(self) -> None:
        if self._fig is not None:
            return

        import matplotlib.pyplot as plt

        self._plt = plt
        self._plt.ion()
        self._fig, self._axes = self._plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        self._fig.canvas.manager.set_window_title(self.title)
        self._fig.suptitle(self.title)

        self._lines = []
        for idx, label in enumerate(self.labels):
            axis = self._axes[0] if idx < 3 else self._axes[1]
            (line,) = axis.plot([], [], label=label, linewidth=0.5, marker=".")
            self._lines.append(line)

        self._axes[0].set_ylabel("Force")
        self._axes[1].set_ylabel("Torque")
        self._axes[1].set_xlabel("Time (s)")
        for axis in self._axes:
            axis.grid(True, alpha=0.35)
            axis.legend(loc="upper right")

        self._fig.tight_layout()
        self._plt.show(block=False)

    def _draw(self, force: bool = False) -> None:
        if self._closed or (not force and not self._times):
            return

        self._ensure_plot()
        xs = np.asarray(self._times, dtype=np.float64)
        ys = np.asarray(self._values, dtype=np.float64)
        if ys.size == 0:
            ys = np.empty((0, 6), dtype=np.float64)

        for idx, line in enumerate(self._lines):
            line.set_data(xs, ys[:, idx] if len(ys) else [])

        for axis in self._axes:
            axis.relim()
            axis.autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


def _inject_tianji_paths() -> Path:
    """Ensure test_teleop paths are importable for Tianji SDK and IK modules."""
    workspace_root = Path(__file__).resolve().parents[4]
    teleop_root = workspace_root / "test_teleop"
    python_dir = teleop_root / "python"
    demo_dir = teleop_root / "demo"

    for path in (python_dir, demo_dir):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    return teleop_root


_TELEOP_ROOT = _inject_tianji_paths()
_MARVIN_IMPORT_ERROR = None
_DUAL_IK_IMPORT_ERROR = None

try:
    from marvin_arm_controller import MarvinArmController
    from utils.kine_model import (
        transform_to_wxyzxyz,
        wxyzxyz_to_transform,
    )
except Exception as exc:  # pragma: no cover - runtime dependency check
    _MARVIN_IMPORT_ERROR = exc
    MarvinArmController = None
    transform_to_wxyzxyz = None
    wxyzxyz_to_transform = None

try:
    from utils.marvin_dual_arm_waist_teleop import DualArmWaistIK
except Exception as exc:  # pragma: no cover - runtime dependency check
    _DUAL_IK_IMPORT_ERROR = exc
    DualArmWaistIK = None


class DefaultTianjiEnvConfig:
    """Default configuration for Tianji right-arm environment."""

    SERVER_URL: str = "10.10.13.10"
    ROBOT_IP: str = "10.10.13.10"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: dict[str, callable] = {}
    WOWSKIN_PORT: str = "/dev/ttyACM0"

    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    RESET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))

    CONTROL_FREQ: float = 20.0
    IK_SERVER_URL: str | None = None # "http://127.0.0.1:8765"
    IK_SERVER_TIMEOUT: float = 1.0
    ACTION_SCALE = np.zeros((3,))
    # 复位/示教等插值运动参数（不影响策略 step 主频）
    INTERPOLATE_HZ: float = 40.0
    INTERPOLATE_MAX_STEP_DEG: float = 0.8
    INTERPOLATE_MAX_POS_STEP_M: float = 0.0025
    INTERPOLATE_MAX_ROT_STEP_RAD: float = 0.025
    INTERPOLATE_EASE: bool = True
    INTERPOLATE_SETTLE_TIMEOUT: float = 0.45
    INTERPOLATE_SETTLE_TOL_DEG: float = 0.2
    INTERPOLATE_SETTLE_HZ: float = 60.0
    RESET_HOLD_DURING_SLEEP: bool = True
    RESET_HOLD_HZ: float = 40.0
    RESET_CONTINUOUS_MODE: bool = True
    RESET_INTERMEDIATE_SETTLE: bool = False
    RESET_FINAL_SETTLE_TIMEOUT: float = 0.12
    RESET_HANDOFF_HOLD_SEC: float = 0.10
    RESET_ASYNC_HANDOFF_HOLD: bool = True
    RESET_HANDOFF_MAX_SEC: float = 1.0
    RESET_HANDOFF_HZ: float = 60.0
    # reset 交接窗口：若动作接近零，则先保持关节 hold，避免首拍 IK 跳解
    RESET_HANDOFF_ZERO_ACTION_EPS: float = 1e-6
    # 仅当位移动作超过该阈值才认为“控制已介入”（raw action 空间）
    RESET_HANDOFF_UNLOCK_ACTION_EPS: float = 0.05
    # 检测到介入后的第一拍是否丢弃位移动作（只保持），降低切换瞬态
    RESET_HANDOFF_DROP_FIRST_CONTROL_STEP: bool = True
    RESET_USE_POSITION_MODE: bool = True
    RESET_POSITION_MODE_VEL_RATIO: int = 12
    RESET_POSITION_MODE_ACC_RATIO: int = 12
    RESET_RESTORE_IMPEDANCE_ON_EXIT: bool = True
    RESET_IMPEDANCE_VEL_RATIO: int = 90
    RESET_IMPEDANCE_ACC_RATIO: int = 90
    RESET_TOP_TO_RESET_LINEAR: bool = False
    RESET_TOP_TO_RESET_TIMEOUT: float = 2.0
    RESET_REPLAY_ALIGN_DWELL_SEC: float = 0.0
    RESET_AFTER_RESET_DWELL_SEC: float = 0.0
    RESET_AFTER_RANDOM_DWELL_SEC: float = 0.0
    QUICK_REGRASP_DWELL_TOP_SEC: float = 0.0
    QUICK_REGRASP_DWELL_TARGET_SEC: float = 0.0
    QUICK_REGRASP_DWELL_FINAL_TOP_SEC: float = 0.0
    RESET_APPLY_CUSTOM_IMPEDANCE: bool = False
    RESET_IMPEDANCE_TARGET_ARM: str = "B"
    RESET_IMPEDANCE_JOINT_K = None
    RESET_IMPEDANCE_JOINT_D = None
    RESET_IMPEDANCE_CART_K = None
    RESET_IMPEDANCE_CART_D = None
    RESET_IMPEDANCE_CART_D_SECONDARY = None
    BASIC_JOINT_RESET = np.zeros((7,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0
    ABS_POSE_LIMIT_HIGH = np.array([0.85, 0.35, 1.50, np.pi, np.pi, np.pi])
    ABS_POSE_LIMIT_LOW = np.array([-0.20, -0.55, 0.40, -np.pi, -np.pi, -np.pi])
    # 若配置的安全框与真实关键位姿(当前/RESET/TARGET/GRASP)不一致，自动扩框避免“被边界吸住”
    AUTO_EXPAND_ABS_POSE_LIMIT: bool = True
    ABS_POSE_LIMIT_EXPAND_MARGIN_XYZ = np.array([0.1, 0.1, 0.1], dtype=np.float64)

    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0
    # 调试开关：None 表示沿用环境变量 HILSERL_DEBUG_TWITCH
    DEBUG_TWITCH = None
    DEBUG_TWITCH_RING = None
    DEBUG_SAFETY_CLIP: bool = True

    RIGHT_ARM_BASE_POSE_WXYZXYZ = np.array(
        [0.707105, 0.707108, -0.000005, 0.000005, 0.319484, -0.012501, 1.127505],
        dtype=np.float64,
    )
    LEFT_ARM_BASE_POSE_WXYZXYZ = np.array(
        [0.707108, -0.707105, -0.000005, -0.000005, 0.319484, 0.012499, 1.127505],
        dtype=np.float64,
    )
    HEAD_BASE_POSE_WXYZXYZ = np.array(
        [0.354649, -0.045577, 1.239027, -0.313173, 0.3, 0.0, 1.1],
        dtype=np.float64,
    )

    TOOL_Z_OFFSET: float = 0.2
    HEAD_Z_OFFSET: float = 0.2

    DUAL_IK_URDF_PATH: str = str(
        (_TELEOP_ROOT / "models/spiderrobot/robot_marvin_arms.urdf").resolve()
    )
    MARVIN_URDF_PATH: str = str(
        (
            _TELEOP_ROOT
            / "demo/urdf/MarvinCCS/urdf/Marvin M6-S-L-CCS-696-V3.1 urdf.urdf"
        ).resolve()
    )


def _pose6_to_transform(pose: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = pose[:3]
    tf[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    return tf


def _transform_to_pose6(tf: np.ndarray) -> np.ndarray:
    euler = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz")
    return np.concatenate([tf[:3, 3], euler]).astype(np.float64)


class TianjiEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultTianjiEnvConfig = None,
        set_load=False,
    ):
        if config is None:
            config = DefaultTianjiEnvConfig()

        self.config = config
        self.fake_env = fake_env

        self.rate_limiter = RateLimiter(self.config.CONTROL_FREQ)
        self.last_timestamp = 0.0
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = np.array(config.TARGET_POSE, dtype=np.float64)
        self._RESET_POSE = np.array(config.RESET_POSE, dtype=np.float64)
        self._GRASP_POSE = np.array(config.GRASP_POSE, dtype=np.float64)
        self._BASIC_JOINT_RESET = np.array(config.BASIC_JOINT_RESET, dtype=np.float64)
        self._REWARD_THRESHOLD = np.array(config.REWARD_THRESHOLD, dtype=np.float64)

        self.url = config.SERVER_URL
        self.robot_ip = config.ROBOT_IP
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.hz = hz
        self.interpolate_hz = float(getattr(config, "INTERPOLATE_HZ", 40.0))
        self.interpolate_max_step_deg = float(
            getattr(config, "INTERPOLATE_MAX_STEP_DEG", 0.8)
        )
        self.interpolate_max_pos_step_m = float(
            getattr(config, "INTERPOLATE_MAX_POS_STEP_M", 0.0025)
        )
        self.interpolate_max_rot_step_rad = float(
            getattr(config, "INTERPOLATE_MAX_ROT_STEP_RAD", 0.025)
        )
        self.interpolate_ease = bool(getattr(config, "INTERPOLATE_EASE", True))
        self.interpolate_settle_timeout = float(
            getattr(config, "INTERPOLATE_SETTLE_TIMEOUT", 0.45)
        )
        self.interpolate_settle_tol_deg = float(
            getattr(config, "INTERPOLATE_SETTLE_TOL_DEG", 0.2)
        )
        self.interpolate_settle_hz = float(
            getattr(config, "INTERPOLATE_SETTLE_HZ", 60.0)
        )
        self.reset_hold_during_sleep = bool(
            getattr(config, "RESET_HOLD_DURING_SLEEP", True)
        )
        self.reset_hold_hz = float(getattr(config, "RESET_HOLD_HZ", 40.0))
        self.reset_continuous_mode = bool(
            getattr(config, "RESET_CONTINUOUS_MODE", True)
        )
        self.reset_intermediate_settle = bool(
            getattr(config, "RESET_INTERMEDIATE_SETTLE", False)
        )
        self.reset_final_settle_timeout = float(
            getattr(config, "RESET_FINAL_SETTLE_TIMEOUT", 0.12)
        )
        self.reset_handoff_hold_sec = float(
            getattr(config, "RESET_HANDOFF_HOLD_SEC", 0.10)
        )
        self.reset_async_handoff_hold = bool(
            getattr(config, "RESET_ASYNC_HANDOFF_HOLD", True)
        )
        self.reset_handoff_max_sec = float(
            getattr(config, "RESET_HANDOFF_MAX_SEC", 1.0)
        )
        self.reset_handoff_hz = float(
            getattr(config, "RESET_HANDOFF_HZ", 60.0)
        )
        self.reset_handoff_zero_action_eps = float(
            getattr(config, "RESET_HANDOFF_ZERO_ACTION_EPS", 1e-6)
        )
        self.reset_handoff_unlock_action_eps = float(
            getattr(config, "RESET_HANDOFF_UNLOCK_ACTION_EPS", 0.05)
        )
        self.reset_handoff_drop_first_control_step = bool(
            getattr(config, "RESET_HANDOFF_DROP_FIRST_CONTROL_STEP", True)
        )
        self.reset_use_position_mode = bool(
            getattr(config, "RESET_USE_POSITION_MODE", True)
        )
        self.reset_position_mode_vel_ratio = int(
            getattr(config, "RESET_POSITION_MODE_VEL_RATIO", 12)
        )
        self.reset_position_mode_acc_ratio = int(
            getattr(config, "RESET_POSITION_MODE_ACC_RATIO", 12)
        )
        self.reset_restore_impedance_on_exit = bool(
            getattr(config, "RESET_RESTORE_IMPEDANCE_ON_EXIT", True)
        )
        self.reset_impedance_vel_ratio = int(
            getattr(config, "RESET_IMPEDANCE_VEL_RATIO", 90)
        )
        self.reset_impedance_acc_ratio = int(
            getattr(config, "RESET_IMPEDANCE_ACC_RATIO", 90)
        )
        self.reset_top_to_reset_linear = bool(
            getattr(config, "RESET_TOP_TO_RESET_LINEAR", False)
        )
        self.reset_top_to_reset_timeout = float(
            getattr(config, "RESET_TOP_TO_RESET_TIMEOUT", 2.0)
        )
        self.reset_replay_align_dwell_sec = float(
            getattr(config, "RESET_REPLAY_ALIGN_DWELL_SEC", 0.0)
        )
        self.reset_after_reset_dwell_sec = float(
            getattr(config, "RESET_AFTER_RESET_DWELL_SEC", 0.0)
        )
        self.reset_after_random_dwell_sec = float(
            getattr(config, "RESET_AFTER_RANDOM_DWELL_SEC", 0.0)
        )
        self.quick_regrasp_dwell_top_sec = float(
            getattr(config, "QUICK_REGRASP_DWELL_TOP_SEC", 0.0)
        )
        self.quick_regrasp_dwell_target_sec = float(
            getattr(config, "QUICK_REGRASP_DWELL_TARGET_SEC", 0.0)
        )
        self.quick_regrasp_dwell_final_top_sec = float(
            getattr(config, "QUICK_REGRASP_DWELL_FINAL_TOP_SEC", 0.0)
        )
        self.reset_apply_custom_impedance = bool(
            getattr(config, "RESET_APPLY_CUSTOM_IMPEDANCE", False)
        )
        self._reset_motion_mode_entered = False

        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = float(config.RANDOM_XY_RANGE)
        self.random_rz_range = float(config.RANDOM_RZ_RANGE)
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD

        self.max_distance = None
        self.last_distance = None
        self.curr_gripper_pos = 1.0
        self.last_gripper_act = time.time()
        self.lastsent = time.time()

        self.controller = None
        self.full_ik_solver = None
        self.ik_client = None
        self._warned_gripper_unavailable = False
        # 关节空间直控/模式切换后，下一次 IK 需要用当前实测关节角重新对齐种子
        self._ik_need_seed_refresh = True
        self._handoff_hold_stop_evt = threading.Event()
        self._handoff_hold_thread = None
        self._post_reset_realign_pending = False
        self._post_reset_hold_target_ql = None
        self._post_reset_hold_target_qr = None

        self.base_right_tf = None
        self.base_left_tf = None
        self.base_head_tf = None
        self.tool_tf = np.eye(4)
        self.tool_tf[2, 3] = config.TOOL_Z_OFFSET
        self.head_z_offset = config.HEAD_Z_OFFSET
        self.body_waist_q = np.zeros(2, dtype=np.float64)

        self.right_target_pose = np.eye(4)
        self.left_target_pose = np.eye(4)
        self.head_target_pose = np.eye(4)
        self._left_pose_locked = False
        self._head_pose_locked = False

        self._last_pose6 = None
        self._last_pose_t = None
        env_debug_twitch = bool(int(os.environ.get("HILSERL_DEBUG_TWITCH", "0")))
        cfg_debug_twitch = getattr(config, "DEBUG_TWITCH", None)
        self.debug_twitch = (
            env_debug_twitch if cfg_debug_twitch is None else bool(cfg_debug_twitch)
        )
        env_ring = int(os.environ.get("HILSERL_DEBUG_TWITCH_RING", "80"))
        cfg_ring = getattr(config, "DEBUG_TWITCH_RING", None)
        self._debug_joint_ring = deque(
            maxlen=env_ring if cfg_ring is None else int(cfg_ring)
        )
        self.debug_safety_clip = bool(getattr(config, "DEBUG_SAFETY_CLIP", False))
        self.auto_expand_abs_pose_limit = bool(
            getattr(config, "AUTO_EXPAND_ABS_POSE_LIMIT", True)
        )
        margin_xyz = np.array(
            getattr(config, "ABS_POSE_LIMIT_EXPAND_MARGIN_XYZ", 0.02),
            dtype=np.float64,
        ).reshape(-1)
        if margin_xyz.size == 1:
            margin_xyz = np.full(3, float(margin_xyz.item()), dtype=np.float64)
        if margin_xyz.size != 3:
            margin_xyz = np.array([0.02, 0.02, 0.02], dtype=np.float64)
        self.abs_pose_limit_expand_margin_xyz = np.abs(margin_xyz)

        if not self.fake_env:
            if _MARVIN_IMPORT_ERROR is not None:
                raise ImportError(
                    "Failed to import Tianji dependencies. Make sure test_teleop/demo and "
                    "test_teleop/python are available and SDK dependencies are installed."
                ) from _MARVIN_IMPORT_ERROR

            self.controller = MarvinArmController(
                robot_ip=self.robot_ip,
                test_mode=False,
                urdf_path=config.MARVIN_URDF_PATH,
            )
            self.controller.init_kinematics()
            connected = self.controller.init_robot_connection()
            if not connected:
                raise RuntimeError(
                    f"Failed to connect Tianji robot at {self.robot_ip}."
                )
            self._has_right_gripper_api = hasattr(self.controller, "right_gripper")

            ik_server_url = os.environ.get(
                "HILSERL_TIANJI_IK_URL",
                getattr(config, "IK_SERVER_URL", None) or "",
            ).strip()
            if ik_server_url:
                self.ik_client = TianjiIKClient(
                    ik_server_url,
                    timeout=float(getattr(config, "IK_SERVER_TIMEOUT", 1.0)),
                )
                print(f"Using remote Tianji IK server: {ik_server_url}")
            else:
                if _DUAL_IK_IMPORT_ERROR is not None:
                    raise ImportError(
                        "Failed to import Tianji IK dependencies. Start "
                        "examples/utils/tianji_ik_server.py and set "
                        "HILSERL_TIANJI_IK_URL to use remote IK, or install local "
                        "IK dependencies."
                    ) from _DUAL_IK_IMPORT_ERROR
                self.full_ik_solver = DualArmWaistIK(
                    urdf_path=config.DUAL_IK_URDF_PATH,
                    enable_viewer=False,
                )
                self.full_ik_solver.set_scale(1.0, 1.0)
                self.full_ik_solver.start()

            self.base_right_tf = wxyzxyz_to_transform(
                np.array(config.RIGHT_ARM_BASE_POSE_WXYZXYZ, dtype=np.float64)
            )
            self.base_left_tf = wxyzxyz_to_transform(
                np.array(config.LEFT_ARM_BASE_POSE_WXYZXYZ, dtype=np.float64)
            )
            self.base_head_tf = wxyzxyz_to_transform(
                np.array(config.HEAD_BASE_POSE_WXYZXYZ, dtype=np.float64)
            )

            if config.WOWSKIN_PORT is not None:
                self.force_sensor = WowSkin(port=config.WOWSKIN_PORT)
                self.force_sensor.reset_baseline()
            else:
                self.force_sensor = None

            self._update_currpos()
            self._set_default_task_poses_from_current()
        else:
            self.force_sensor = None
            self.currjoint = np.zeros(7, dtype=np.float64)
            self.currpos = np.zeros(6, dtype=np.float64)
            self.currvel = np.zeros(6, dtype=np.float64)
            self.currforce = np.zeros(15, dtype=np.float64)
            self.currtorque = np.zeros(3, dtype=np.float64)
            self.resetpos = np.zeros(6, dtype=np.float64)

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        self.xyz_bounding_box = gym.spaces.Box(
            np.array(config.ABS_POSE_LIMIT_LOW[:3], dtype=np.float64),
            np.array(config.ABS_POSE_LIMIT_HIGH[:3], dtype=np.float64),
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array(config.ABS_POSE_LIMIT_LOW[3:], dtype=np.float64),
            np.array(config.ABS_POSE_LIMIT_HIGH[3:], dtype=np.float64),
            dtype=np.float64,
        )
        self._ensure_safety_box_contains_key_poses(reason="init")

        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(15,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "eef_force": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0,
                            255,
                            shape=(*IMAGE_SIZE, 3),
                            dtype=np.uint8,
                        )
                        for key in config.REALSENSE_CAMERAS
                    }
                ),
            }
        )

        self.cycle_count = 0

        if self.fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = Queue()
            self.displayer = ImageDisplayer(self.img_queue, f"tianji_{self.robot_ip}")
            self.displayer.start()

        self.forece6d_plotter = RealtimeForce6DPlotter(max_points=300, update_interval=0.05)

        if set_load:
            pass

        from pynput import keyboard

        self.terminate = False

        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

        print("Initialized Tianji MARVIN Env.")

    def _joints_deg_to_matrix(self, joints_deg):
        """利用正运动学(FK)，把关节角度转化为6D笛卡尔位姿用于计算Reward"""
        joints_rad = np.array(joints_deg, dtype=np.float64) * self.controller.DEG_TO_RAD
        fk_mat = self.controller.compute_fk(joints_rad)
        pose_mat = fk_mat @ self.tool_tf
        return pose_mat

    def _joints_deg_to_pose6(self, joints_deg):
        """利用正运动学(FK)，把关节角度转化为6D笛卡尔位姿用于计算Reward"""
        joints_rad = np.array(joints_deg, dtype=np.float64) * self.controller.DEG_TO_RAD
        fk_mat = self.controller.compute_fk(joints_rad)
        pose_mat = self.base_right_tf @ fk_mat @ self.tool_tf
        return _transform_to_pose6(pose_mat)

    def _apply_rpy_to_pose6(self, pose, rpy):
        """利用正运动学(FK)，把关节角度转化为6D笛卡尔位姿用于计算Reward"""
        pose_mat = _pose6_to_transform(pose)
        pose_mat[:3, :3] = pose_mat[:3, :3] @ Rotation.from_euler("xyz", rpy).as_matrix()
        return _transform_to_pose6(pose_mat)

    def _set_default_task_poses_from_current(self):
        """初始化时，直接根据 config 中的关节角自动计算对应的笛卡尔目标"""
        if hasattr(self.config, "TARGET_JOINTS"):
            self._TARGET_POSE = self._joints_deg_to_pose6(self.config.TARGET_JOINTS)
        else:
            self._TARGET_POSE = self.currpos.copy()

        if hasattr(self.config, "RESET_JOINTS"):
            self._RESET_POSE = self._joints_deg_to_pose6(self.config.RESET_JOINTS)
            self.resetpos = self._RESET_POSE.copy()
        else:
            self._RESET_POSE = self.currpos.copy()
            self.resetpos = self.currpos.copy()

        print("\n\n\n!!!! reset_pose: ", self._RESET_POSE)

        if hasattr(self.config, "GRASP_JOINTS"):
            self._GRASP_POSE = self._joints_deg_to_pose6(self.config.GRASP_JOINTS)
        else:
            self._GRASP_POSE = self._TARGET_POSE.copy()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _debug_log(self, message: str):
        if not self.debug_twitch:
            return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[TWITCH {ts}] {message}")

    def _debug_append_joint_sample(
        self,
        tag: str,
        target_qr: np.ndarray | None = None,
        ql_deg: np.ndarray | None = None,
        qr_deg: np.ndarray | None = None,
    ):
        if not self.debug_twitch or self.fake_env or self.controller is None:
            return
        if ql_deg is None:
            ql_deg = (
                np.array(self.controller.get_joint_pos_rad(arm_id=1), dtype=np.float64)
                / self.controller.DEG_TO_RAD
            )
        if qr_deg is None:
            qr_deg = (
                np.array(self.controller.get_joint_pos_rad(arm_id=2), dtype=np.float64)
                / self.controller.DEG_TO_RAD
            )
        self._debug_joint_ring.append(
            {
                "t": time.time(),
                "tag": tag,
                "ql": np.array(ql_deg, dtype=np.float64),
                "qr": np.array(qr_deg, dtype=np.float64),
                "target_qr": None
                if target_qr is None
                else np.array(target_qr, dtype=np.float64),
            }
        )

    def _debug_dump_recent_joint(self, reason: str, window: int = 3):
        if not self.debug_twitch:
            return
        recent = list(self._debug_joint_ring)[-max(1, window * 2) :]
        self._debug_log(f"{reason}: dump {len(recent)} samples")
        for rec in recent:
            qr_head = np.round(rec["qr"][:3], 3).tolist()
            if rec["target_qr"] is None:
                self._debug_log(f"{rec['tag']} qr[:3]={qr_head}")
            else:
                tgt_head = np.round(rec["target_qr"][:3], 3).tolist()
                max_err = float(np.max(np.abs(rec["qr"] - rec["target_qr"])))
                self._debug_log(
                    f"{rec['tag']} qr[:3]={qr_head} target[:3]={tgt_head} max_err={max_err:.4f}deg"
                )

    def debug_dump_recent_joint_samples(self, reason: str = "manual", window: int = 3):
        self._debug_dump_recent_joint(reason=reason, window=window)

    def _mark_ik_seed_refresh(self, reason: str = ""):
        self._ik_need_seed_refresh = True
        if self.debug_twitch and reason:
            self._debug_log(f"ik_seed_refresh marked: {reason}")

    @staticmethod
    def _normalize_kd_list(values, name: str):
        if values is None:
            return None
        arr = np.array(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 7 or not np.all(np.isfinite(arr)):
            print(f"[ANTI_SAG] ignore invalid {name}, expect 7 finite values.")
            return None
        return arr.tolist()

    def _apply_custom_impedance_profile_if_needed(self):
        if (
            self.fake_env
            or self.controller is None
            or not self.reset_apply_custom_impedance
            or not hasattr(self.controller, "set_cartesian_impedance_profile")
        ):
            return
        joint_k = self._normalize_kd_list(
            getattr(self.config, "RESET_IMPEDANCE_JOINT_K", None),
            "RESET_IMPEDANCE_JOINT_K",
        )
        joint_d = self._normalize_kd_list(
            getattr(self.config, "RESET_IMPEDANCE_JOINT_D", None),
            "RESET_IMPEDANCE_JOINT_D",
        )
        cart_k = self._normalize_kd_list(
            getattr(self.config, "RESET_IMPEDANCE_CART_K", None),
            "RESET_IMPEDANCE_CART_K",
        )
        cart_d = self._normalize_kd_list(
            getattr(self.config, "RESET_IMPEDANCE_CART_D", None),
            "RESET_IMPEDANCE_CART_D",
        )
        cart_d_secondary = self._normalize_kd_list(
            getattr(self.config, "RESET_IMPEDANCE_CART_D_SECONDARY", None),
            "RESET_IMPEDANCE_CART_D_SECONDARY",
        )
        if (
            joint_k is None
            and joint_d is None
            and cart_k is None
            and cart_d is None
            and cart_d_secondary is None
        ):
            return
        ok = self.controller.set_cartesian_impedance_profile(
            joint_k=joint_k,
            joint_d=joint_d,
            cart_k=cart_k,
            cart_d=cart_d,
            cart_d_secondary=cart_d_secondary,
            target_arm=getattr(self.config, "RESET_IMPEDANCE_TARGET_ARM", "B"),
        )
        if not ok:
            print("[ANTI_SAG] failed to apply custom impedance profile.")

    def _enter_reset_motion_mode(self, reason: str = ""):
        if self.fake_env or self.controller is None:
            return
        if self._reset_motion_mode_entered:
            return
        self._reset_motion_mode_entered = True
        if self.reset_use_position_mode and hasattr(self.controller, "enter_position_mode"):
            ok = self.controller.enter_position_mode(
                vel_ratio=self.reset_position_mode_vel_ratio,
                acc_ratio=self.reset_position_mode_acc_ratio,
            )
            if not ok:
                print("[ANTI_SAG] failed to switch to position mode for reset.")
        else:
            # 不切位置模式时，确保每次 reset 入口都把自定义阻抗写回去。
            self._apply_custom_impedance_profile_if_needed()
        self._mark_ik_seed_refresh(f"enter_reset_motion_mode {reason}")

    def _restore_control_mode_after_reset(self, reason: str = ""):
        if self.fake_env or self.controller is None:
            self._reset_motion_mode_entered = False
            return
        if not self._reset_motion_mode_entered:
            return
        self._reset_motion_mode_entered = False
        if self.reset_restore_impedance_on_exit and hasattr(
            self.controller, "enter_cartesian_impedance_mode"
        ):
            ok = self.controller.enter_cartesian_impedance_mode(
                vel_ratio=self.reset_impedance_vel_ratio,
                acc_ratio=self.reset_impedance_acc_ratio,
            )
            if not ok:
                print("[ANTI_SAG] failed to restore cartesian impedance mode.")
            self._apply_custom_impedance_profile_if_needed()
        self._mark_ik_seed_refresh(f"restore_control_mode_after_reset {reason}")

    def _pause_with_hold(self, sec: float, reason: str = ""):
        dt = max(0.0, float(sec))
        if dt <= 0:
            return
        if (
            self.fake_env
            or self.controller is None
            or not self.reset_hold_during_sleep
        ):
            time.sleep(dt)
            return
        hold_hz = max(1.0, float(self.reset_hold_hz))
        period = 1.0 / hold_hz
        target_ql = (
            np.array(self.controller.get_joint_pos_rad(arm_id=1), dtype=np.float64)
            / self.controller.DEG_TO_RAD
        )
        target_qr = (
            np.array(self.controller.get_joint_pos_rad(arm_id=2), dtype=np.float64)
            / self.controller.DEG_TO_RAD
        )
        end_t = time.perf_counter() + dt
        next_tick = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now >= end_t:
                break
            self.controller.step(
                np.array(target_ql, dtype=np.float64),
                np.array(target_qr, dtype=np.float64),
                verbose=False,
            )
            next_tick += period
            sleep_dt = min(max(0.0, next_tick - time.perf_counter()), max(0.0, end_t - time.perf_counter()))
            if sleep_dt > 0:
                time.sleep(sleep_dt)
        self._update_currpos()
        self.cmd_pose = self.currpos.copy()
        self.nextpos = self.currpos.copy()

    def _stop_async_handoff_hold(self, block: bool = True):
        th = getattr(self, "_handoff_hold_thread", None)
        if th is None:
            return
        try:
            self._handoff_hold_stop_evt.set()
            if block and th.is_alive():
                th.join(timeout=0.2)
        finally:
            self._handoff_hold_thread = None

    def _start_async_handoff_hold(self, max_sec: float | None = None):
        if (
            self.fake_env
            or self.controller is None
            or not self.reset_async_handoff_hold
        ):
            return
        self._stop_async_handoff_hold(block=False)
        hold_sec = self.reset_handoff_max_sec if max_sec is None else float(max_sec)
        hold_sec = max(0.0, hold_sec)
        if hold_sec <= 0:
            return
        hold_hz = max(1.0, float(self.reset_handoff_hz))
        period = 1.0 / hold_hz
        target_ql = (
            np.array(self.controller.get_joint_pos_rad(arm_id=1), dtype=np.float64)
            / self.controller.DEG_TO_RAD
        )
        target_qr = (
            np.array(self.controller.get_joint_pos_rad(arm_id=2), dtype=np.float64)
            / self.controller.DEG_TO_RAD
        )
        self._handoff_hold_stop_evt = threading.Event()
        stop_evt = self._handoff_hold_stop_evt

        def _worker():
            end_t = time.perf_counter() + hold_sec
            next_tick = time.perf_counter()
            while (not stop_evt.is_set()) and (time.perf_counter() < end_t):
                try:
                    self.controller.step(
                        np.array(target_ql, dtype=np.float64),
                        np.array(target_qr, dtype=np.float64),
                        verbose=False,
                    )
                except Exception:
                    break
                next_tick += period
                sleep_dt = max(0.0, next_tick - time.perf_counter())
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

        th = threading.Thread(target=_worker, daemon=True)
        self._handoff_hold_thread = th
        th.start()

    def _is_zero_motion_action(self, action: np.ndarray) -> bool:
        arr = np.array(action, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 6:
            return True
        eps = max(0.0, float(self.reset_handoff_zero_action_eps))
        return float(np.max(np.abs(arr[:6]))) <= eps

    def _has_control_intent(self, action: np.ndarray) -> bool:
        arr = np.array(action, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 6:
            return False
        eps = max(0.0, float(self.reset_handoff_unlock_action_eps))
        return float(np.max(np.abs(arr[:6]))) > eps

    def _async_handoff_hold_alive(self) -> bool:
        th = getattr(self, "_handoff_hold_thread", None)
        return bool(th is not None and th.is_alive())

    def _capture_post_reset_hold_target(self):
        if self.fake_env or self.controller is None:
            return
        self._post_reset_hold_target_ql = (
            np.array(self.controller.get_joint_pos_rad(arm_id=1), dtype=np.float64)
            / self.controller.DEG_TO_RAD
        )
        self._post_reset_hold_target_qr = (
            np.array(self.controller.get_joint_pos_rad(arm_id=2), dtype=np.float64)
            / self.controller.DEG_TO_RAD
        )

    def _hold_post_reset_target_once(self):
        if self.fake_env or self.controller is None:
            return
        if (
            self._post_reset_hold_target_ql is None
            or self._post_reset_hold_target_qr is None
        ):
            self._capture_post_reset_hold_target()
        target_ql = np.array(self._post_reset_hold_target_ql, dtype=np.float64)
        target_qr = np.array(self._post_reset_hold_target_qr, dtype=np.float64)
        self.controller.step(
            np.array(target_ql, dtype=np.float64),
            np.array(target_qr, dtype=np.float64),
            verbose=False,
        )
        self._update_currpos()
        self.cmd_pose = self.currpos.copy()
        self.nextpos = self.currpos.copy()

    def _settle_right_joint_target(self, target_deg: np.ndarray, timeout_override: float | None = None):
        if self.fake_env or self.controller is None:
            return
        timeout_raw = self.interpolate_settle_timeout if timeout_override is None else timeout_override
        timeout = max(0.0, float(timeout_raw))
        if timeout <= 0:
            return
        tol_deg = max(1e-4, float(self.interpolate_settle_tol_deg))
        hz = max(1.0, float(self.interpolate_settle_hz))
        period = 1.0 / hz
        target = np.array(target_deg, dtype=np.float64).reshape(-1)
        if target.shape[0] != 7:
            return
        end_t = time.perf_counter() + timeout
        next_tick = time.perf_counter()
        while True:
            ql_now = (
                np.array(self.controller.get_joint_pos_rad(arm_id=1), dtype=np.float64)
                / self.controller.DEG_TO_RAD
            )
            qr_now = (
                np.array(self.controller.get_joint_pos_rad(arm_id=2), dtype=np.float64)
                / self.controller.DEG_TO_RAD
            )
            max_err = float(np.max(np.abs(qr_now - target)))
            self.controller.step(
                np.array(ql_now, dtype=np.float64),
                np.array(target, dtype=np.float64),
                verbose=False,
            )
            if max_err <= tol_deg or time.perf_counter() >= end_t:
                break
            next_tick += period
            sleep_dt = max(0.0, next_tick - time.perf_counter())
            if sleep_dt > 0:
                time.sleep(sleep_dt)

    def _ensure_safety_box_contains_key_poses(self, reason: str = ""):
        poses = []
        for name in ("currpos", "_RESET_POSE", "_TARGET_POSE", "_GRASP_POSE"):
            pose = getattr(self, name, None)
            if pose is None:
                continue
            arr = np.array(pose, dtype=np.float64).reshape(-1)
            if arr.shape[0] == 6 and np.all(np.isfinite(arr)):
                poses.append(arr)
        if not poses:
            return

        stacked = np.vstack(poses)
        need_low_xyz = (
            np.min(stacked[:, :3], axis=0) - self.abs_pose_limit_expand_margin_xyz
        )
        need_high_xyz = (
            np.max(stacked[:, :3], axis=0) + self.abs_pose_limit_expand_margin_xyz
        )

        cur_low = np.array(self.xyz_bounding_box.low, dtype=np.float64)
        cur_high = np.array(self.xyz_bounding_box.high, dtype=np.float64)

        if self.auto_expand_abs_pose_limit:
            new_low = np.minimum(cur_low, need_low_xyz)
            new_high = np.maximum(cur_high, need_high_xyz)
            if np.max(np.abs(new_low - cur_low)) > 1e-9 or np.max(
                np.abs(new_high - cur_high)
            ) > 1e-9:
                self.xyz_bounding_box = gym.spaces.Box(
                    new_low.astype(np.float64),
                    new_high.astype(np.float64),
                    dtype=np.float64,
                )
                print(
                    "[SAFETY_BOX] auto-expand"
                    f"{'' if reason == '' else f'({reason})'} "
                    f"xyz_bounding_box.low={np.round(cur_low, 4).tolist()} "
                    f"xyz_bounding_box.high={np.round(cur_high, 4).tolist()} "
                    f"xyz_low={np.round(new_low, 4).tolist()} "
                    f"xyz_high={np.round(new_high, 4).tolist()}"
                )
        else:
            bad_low = need_low_xyz < cur_low - 1e-9
            bad_high = need_high_xyz > cur_high + 1e-9
            if np.any(bad_low) or np.any(bad_high):
                names = np.array(["x", "y", "z"])
                clipped_axes = names[np.logical_or(bad_low, bad_high)].tolist()
                print(
                    "[SAFETY_BOX] warning key pose outside configured xyz bounds "
                    f"axes={clipped_axes} "
                    f"need_low={np.round(need_low_xyz, 4).tolist()} "
                    f"need_high={np.round(need_high_xyz, 4).tolist()} "
                    f"cfg_low={np.round(cur_low, 4).tolist()} "
                    f"cfg_high={np.round(cur_high, 4).tolist()}"
                )

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        pose = np.array(pose, dtype=np.float64).copy()
        
        # 限制 XYZ 空间位置
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        
        # 限制 欧拉角姿态（删掉原本 sign 相关的错误逻辑，直接使用标准的 np.clip）
        pose[3:] = np.clip(
            pose[3:], self.rpy_bounding_box.low, self.rpy_bounding_box.high
        )
        
        return pose

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        print("infer time: ", start_time - self.last_timestamp)

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self._post_reset_realign_pending and (not self._has_control_intent(action)):
            gripper_action = (
                (action[6] if action.shape[0] > 6 else 0.0)
                * self.action_scale[2]
            )
            self._send_gripper_command(gripper_action)
            if self._async_handoff_hold_alive():
                self._update_currpos()
                self.cmd_pose = self.currpos.copy()
                self.nextpos = self.currpos.copy()
            else:
                self._hold_post_reset_target_once()
            self.curr_path_length += 1
            ob = self._get_obs()
            reward, finish = 0, False
            done = (
                self.curr_path_length >= self.max_episode_length
                or finish
                or getattr(self, "terminate", False)
            )
            return ob, reward, done, False, {"succeed": finish}

        if self._post_reset_realign_pending:
            self._stop_async_handoff_hold(block=True)
            self._update_currpos()
            self.cmd_pose = self.currpos.copy()
            self.nextpos = self.currpos.copy()
            self._post_reset_realign_pending = False
            if self.reset_handoff_drop_first_control_step:
                gripper_action = (
                    (action[6] if action.shape[0] > 6 else 0.0)
                    * self.action_scale[2]
                )
                self._send_gripper_command(gripper_action)
                self._hold_post_reset_target_once()
                self.curr_path_length += 1
                ob = self._get_obs()
                reward, finish = 0, False
                done = (
                    self.curr_path_length >= self.max_episode_length
                    or finish
                    or getattr(self, "terminate", False)
                )
                return ob, reward, done, False, {"succeed": finish}

        self._stop_async_handoff_hold(block=True)

        # 1. 确保指令目标状态被正确初始化 (阻断 IK/FK 累积误差)
        if not hasattr(self, "cmd_pose"):
            self.cmd_pose = self.currpos.copy()

        # 2. 将当前的 6D cmd_pose 转换为 4x4 变换矩阵 (T_target)
        T_target = _pose6_to_transform(self.cmd_pose)

        # 3. 提取平移和旋转的增量 (Twist)
        lin_vel = action[:3] * self.action_scale[0]
        
        # 旋转部分如果极小则忽略
        if np.all(np.abs(action[3:6]) < 1e-4) or abs(self.action_scale[1]) < 1e-6:
            ang_vel = np.zeros(3)
        else:
            ang_vel = action[3:6] * self.action_scale[1]

        d_twist = np.concatenate([lin_vel, ang_vel])

        # 4. 使用矩阵指数 (expm) 计算相对变换 (这完美对齐了你测试脚本中的 calc_transform_after_twist)
        twist_mat = np.zeros((4, 4))
        w = d_twist[3:]
        v = d_twist[:3]
        twist_mat[:3, :3] = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        twist_mat[:3, 3] = v

        # T_final = T_relative @ T_initial
        T_relative = expm(twist_mat)
        # target_t = T_target[:3, 3]
        # T_target[:3, 3] = 0.0
        T_target_new = np.eye(4)
        T_target_new[:3, :3] = T_relative[:3, :3] @ T_target[:3, :3]
        T_target_new[:3, 3] = T_target[:3, 3] + T_relative[:3, 3]

        # 5. 还原回 6D pose 并更新 cmd_pose
        self.nextpos = _transform_to_pose6(T_target_new)
        
        # self.cmd_pose = self.nextpos.copy()

        # 6. 发送指令
        gripper_action = action[6] * self.action_scale[2]
        self._send_gripper_command(gripper_action)
        
        # 使用修复后的 clip_safety_box 过滤后再发送给 IK
        safe_target = self.clip_safety_box(self.nextpos)
        if self.debug_safety_clip:
            delta = safe_target - self.nextpos
            if np.max(np.abs(delta)) > 1e-9:
                names = np.array(["x", "y", "z", "rx", "ry", "rz"])
                clipped = names[np.abs(delta) > 1e-9].tolist()
                print(
                    "[SAFETY_CLIP] clipped="
                    f"{clipped} next={np.round(self.nextpos, 4).tolist()} "
                    f"safe={np.round(safe_target, 4).tolist()} "
                    f"low={np.round(self.xyz_bounding_box.low.tolist() + self.rpy_bounding_box.low.tolist(), 4).tolist()} "
                    f"high={np.round(self.xyz_bounding_box.high.tolist() + self.rpy_bounding_box.high.tolist(), 4).tolist()}"
                )
        # 【关键修复】：必须将 cmd_pose 强制对齐到 safe_target！
        # 彻底解决碰到边界后“积分饱和”导致按键失灵的问题
        self.cmd_pose = safe_target.copy()

        exec_time = time.time() - start_time
        print("before ik time: ", exec_time)

        self._send_pos_command(safe_target)

        self.curr_path_length += 1
        exec_time = time.time() - start_time
        print("ik time: ", exec_time)
        self._update_currpos()

        self.rate_limiter.sleep()
        exec_time = time.time() - start_time
        print("sync time: ", exec_time)
        ob = self._get_obs()
        print("obs time: ", time.time() - start_time - exec_time)
    
        reward, finish = -0.02, False
        done = (
            self.curr_path_length >= self.max_episode_length
            or finish
            or getattr(self, "terminate", False)
        )

        # self.forece6d_plotter.add_point(ob["state"]["eef_force"])

        self.last_timestamp = time.time()
        return ob, reward, done, False, {"succeed": finish}

    def compute_reward(self, obs):
        print("\n\n\ncompute reward")
        current_pose = obs["state"]["tcp_pose"]
        current_rot = Rotation.from_euler("xyz", current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(
            np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler])
        )
        distance = np.linalg.norm(delta)
        if self.max_distance is None:
            self.max_distance = distance
            self.last_distance = distance
            return 0, False

        if distance >= self.last_distance:
            reward = 0
        else:
            reward = (self.last_distance - distance) / self.max_distance
            self.last_distance = distance
        return reward, False

    def get_im(self) -> Dict[str, np.ndarray]:
        images = {}
        display_images = {}
        full_res_images = {}

        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = (
                    self.config.IMAGE_CROP[key](rgb)
                    if key in self.config.IMAGE_CROP
                    else rgb
                )
                resized = cv2.resize(
                    cropped_rgb,
                    self.observation_space["images"][key].shape[:2][::-1],
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[f"{key}_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                input(f"{key} camera frozen. Check connection and press enter to retry...")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)

        return images

    def interpolate_move(
        self,
        goal: np.ndarray,
        timeout: float,
        is_reset=False,
        ease: bool | None = None,
    ):
        rate_hz = max(1.0, float(self.interpolate_hz))
        self._update_currpos()
        goal_arr = np.array(goal, dtype=np.float64).reshape(-1)
        if goal_arr.shape[0] != 6:
            print(f"Invalid cartesian goal shape: {goal_arr.shape}, expected (6,)")
            return

        pos_delta = float(np.linalg.norm(goal_arr[:3] - self.currpos[:3]))
        rot_delta = float(np.max(np.abs(goal_arr[3:] - self.currpos[3:])))
        max_pos_step = max(1e-6, float(self.interpolate_max_pos_step_m))
        max_rot_step = max(1e-6, float(self.interpolate_max_rot_step_rad))
        steps_by_timeout = int(np.ceil(max(0.0, float(timeout)) * rate_hz))
        steps_by_pos = int(np.ceil(pos_delta / max_pos_step))
        steps_by_rot = int(np.ceil(rot_delta / max_rot_step))
        steps = max(2, steps_by_timeout, steps_by_pos, steps_by_rot)

        use_ease = self.interpolate_ease if ease is None else bool(ease)
        if use_ease:
            u = np.linspace(0.0, 1.0, steps)
            s = 0.5 - 0.5 * np.cos(np.pi * u)  # smoothstep-like ease in/out
            path = self.currpos[None, :] + (goal_arr - self.currpos)[None, :] * s[:, None]
        else:
            path = np.linspace(self.currpos, goal_arr, steps)
        period = 1.0 / rate_hz
        next_tick = time.perf_counter()
        for p in path:
            self._send_pos_command(p, is_reset=is_reset)
            self._update_currpos()
            next_tick += period
            sleep_dt = max(0.0, next_tick - time.perf_counter())
            if sleep_dt > 0:
                time.sleep(sleep_dt)
        self.nextpos = goal_arr.copy()
        self.cmd_pose = self.nextpos.copy()
        self._update_currpos()

    def interpolate_move_waypoints(
        self,
        waypoints: list[np.ndarray],
        timeout: float | None = None,
        is_reset=False,
    ):
        """连续经过多个笛卡尔关键点，不在中间关键点刹停。"""
        if self.fake_env:
            return
        if waypoints is None or len(waypoints) == 0:
            return

        self._update_currpos()
        start = self.currpos.copy()
        targets: list[np.ndarray] = []
        prev = start
        for wp in waypoints:
            arr = np.array(wp, dtype=np.float64).reshape(-1)
            if arr.shape[0] != 6:
                continue
            arr = self.clip_safety_box(arr)
            if np.max(np.abs(arr - prev)) < 1e-6:
                continue
            targets.append(arr)
            prev = arr
        if not targets:
            return

        rate_hz = max(1.0, float(self.interpolate_hz))
        max_pos_step = max(1e-6, float(self.interpolate_max_pos_step_m))
        max_rot_step = max(1e-6, float(self.interpolate_max_rot_step_rad))

        seg_metrics = []
        seg_steps = []
        prev = start
        for tgt in targets:
            pos_delta = float(np.linalg.norm(tgt[:3] - prev[:3]))
            rot_delta = float(np.max(np.abs(tgt[3:] - prev[3:])))
            metric = max(
                pos_delta / max_pos_step,
                rot_delta / max_rot_step,
                1.0,
            )
            seg_metrics.append(metric)
            seg_steps.append(max(2, int(np.ceil(metric))))
            prev = tgt

        total_steps = int(np.sum(seg_steps))
        if timeout is not None:
            desired_steps = max(2, int(np.ceil(max(0.0, float(timeout)) * rate_hz)))
            if desired_steps > total_steps and total_steps > 0:
                scale = float(desired_steps) / float(total_steps)
                seg_steps = [max(2, int(np.ceil(s * scale))) for s in seg_steps]

        path_parts = []
        prev = start
        for i, tgt in enumerate(targets):
            n = max(2, int(seg_steps[i]))
            seg = np.linspace(prev, tgt, n, endpoint=False)
            path_parts.append(seg)
            prev = tgt
        path = np.vstack(path_parts + [targets[-1][None, :]])

        period = 1.0 / rate_hz
        next_tick = time.perf_counter()
        for p in path:
            self._send_pos_command(p, is_reset=is_reset)
            self._update_currpos()
            next_tick += period
            sleep_dt = max(0.0, next_tick - time.perf_counter())
            if sleep_dt > 0:
                time.sleep(sleep_dt)
        print("interpolate_move_waypoints插值结束")
        # time.sleep(1.0)
        print("interpolate_move_waypoints插值结束并sleep结束")
        self.nextpos = targets[-1].copy()
        print("更新self.nextpos ")
        self.cmd_pose = self.nextpos.copy()
        print("更新self.cmd_pose ")
        self._update_currpos()
        print("运行update currentpos")

    def interpolate_joint_move(
        self,
        target_joints_deg: np.ndarray,
        timeout: float = 2.0,
        settle: bool = True,
        settle_timeout: float | None = None,
    ):
        """纯关节空间的平滑移动，完全绕过逆运动学，杜绝抽搐！"""
        rate_hz = max(1.0, float(self.interpolate_hz))
        steps_guess = max(2, int(np.ceil(timeout * rate_hz)))
        if self.fake_env:
            return

        target = np.array(target_joints_deg, dtype=np.float64).reshape(-1)
        if target.shape[0] != 7:
            print(f"Invalid target joint shape: {target.shape}, expected (7,)")
            return

        rate_hz = max(1.0, float(self.interpolate_hz))
        
        # 获取当前的左右臂关节角度
        current_ql = self.controller.get_joint_pos_rad(arm_id=1) / self.controller.DEG_TO_RAD
        current_qr = self.controller.get_joint_pos_rad(arm_id=2) / self.controller.DEG_TO_RAD
        # self._debug_log(
        #     f"interpolate_joint_move start steps~{steps_guess} timeout={timeout:.2f}s "
        #     f"target_qr[:3]={np.round(target[:3], 3).tolist()}"
        # )
        self._debug_append_joint_sample(
            tag="interp_start",
            target_qr=target,
            ql_deg=np.array(current_ql, dtype=np.float64),
            qr_deg=np.array(current_qr, dtype=np.float64),
        )

        # 目标与当前几乎一致时直接跳过，避免重复发同点指令导致偶发抖动
        if np.max(np.abs(current_qr - target)) < 1e-3:
            self._update_currpos()
            self.cmd_pose = self.currpos.copy()
            self._mark_ik_seed_refresh("interpolate_joint_move skip_same_target")
            self._debug_append_joint_sample(tag="interp_skip_same_target", target_qr=target)
            self._debug_dump_recent_joint(reason="interpolate_joint_move skip", window=3)
            return

        # 自适应步数：同时考虑总时长与单关节最大步进角，减少“每拍跨太大”导致的卡顿
        max_delta_deg = float(np.max(np.abs(target - current_qr)))
        step_cap_deg = max(1e-3, float(self.interpolate_max_step_deg))
        steps_by_timeout = int(np.ceil(timeout * rate_hz))
        steps_by_delta = int(np.ceil(max_delta_deg / step_cap_deg))
        steps = max(2, steps_by_timeout, steps_by_delta)

        # 在关节空间生成插值轨迹
        if self.interpolate_ease:
            u = np.linspace(0.0, 1.0, steps)
            s = 0.5 - 0.5 * np.cos(np.pi * u)  # ease in/out
            path_r = current_qr[None, :] + (target - current_qr)[None, :] * s[:, None]
        else:
            path_r = np.linspace(current_qr, target, steps)

        period = 1.0 / rate_hz
        next_tick = time.perf_counter()
        for pr in path_r:
            self._debug_append_joint_sample(tag="interp_cmd", target_qr=pr)
            self.controller.step(
                np.array(current_ql, dtype=np.float64),
                np.array(pr, dtype=np.float64),
                verbose=False
            )
            next_tick += period
            sleep_dt = max(0.0, next_tick - time.perf_counter())
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            self._debug_append_joint_sample(tag="interp_fb", target_qr=pr)
        # 到点后继续短暂闭环，减少“到点即松一下”导致的下垂
        if settle:
            self._settle_right_joint_target(target, timeout_override=settle_timeout)

        # 移动完毕后，强制同步底层的真实位置，防止下一轮启动时跳变
        self._update_currpos()
        self.cmd_pose = self.currpos.copy()
        self._mark_ik_seed_refresh("interpolate_joint_move end")
        self._debug_dump_recent_joint(reason="interpolate_joint_move end", window=3)

    def interpolate_joint_waypoints(
        self,
        waypoints_deg: list[np.ndarray],
        timeout: float | None = None,
        settle_final: bool = True,
        settle_timeout: float | None = None,
    ):
        """连续经过多个关节关键点，不在中间关键点刹停。"""
        if self.fake_env:
            return
        if waypoints_deg is None or len(waypoints_deg) == 0:
            return

        current_ql = (
            np.array(self.controller.get_joint_pos_rad(arm_id=1), dtype=np.float64)
            / self.controller.DEG_TO_RAD
        )
        current_qr = (
            np.array(self.controller.get_joint_pos_rad(arm_id=2), dtype=np.float64)
            / self.controller.DEG_TO_RAD
        )

        targets: list[np.ndarray] = []
        prev = current_qr.copy()
        for wp in waypoints_deg:
            arr = np.array(wp, dtype=np.float64).reshape(-1)
            if arr.shape[0] != 7:
                continue
            if np.max(np.abs(arr - prev)) < 1e-3:
                continue
            targets.append(arr)
            prev = arr
        if not targets:
            self._update_currpos()
            self.cmd_pose = self.currpos.copy()
            return

        rate_hz = max(1.0, float(self.interpolate_hz))
        step_cap_deg = max(1e-3, float(self.interpolate_max_step_deg))
        seg_steps = []
        prev = current_qr.copy()
        for tgt in targets:
            max_delta_deg = float(np.max(np.abs(tgt - prev)))
            seg_steps.append(max(2, int(np.ceil(max_delta_deg / step_cap_deg))))
            prev = tgt

        total_steps = int(np.sum(seg_steps))
        if timeout is not None:
            desired_steps = max(2, int(np.ceil(max(0.0, float(timeout)) * rate_hz)))
            if desired_steps > total_steps and total_steps > 0:
                scale = float(desired_steps) / float(total_steps)
                seg_steps = [max(2, int(np.ceil(s * scale))) for s in seg_steps]

        path_parts = []
        prev = current_qr.copy()
        for i, tgt in enumerate(targets):
            n = max(2, int(seg_steps[i]))
            seg = np.linspace(prev, tgt, n, endpoint=False)
            path_parts.append(seg)
            prev = tgt
        path_r = np.vstack(path_parts + [targets[-1][None, :]])

        period = 1.0 / rate_hz
        next_tick = time.perf_counter()
        for pr in path_r:
            self.controller.step(
                np.array(current_ql, dtype=np.float64),
                np.array(pr, dtype=np.float64),
                verbose=False,
            )
            next_tick += period
            sleep_dt = max(0.0, next_tick - time.perf_counter())
            if sleep_dt > 0:
                time.sleep(sleep_dt)

        if settle_final:
            self._settle_right_joint_target(
                np.array(targets[-1], dtype=np.float64),
                timeout_override=settle_timeout,
            )

        self._update_currpos()
        self.cmd_pose = self.currpos.copy()
        self._mark_ik_seed_refresh("interpolate_joint_waypoints end")

    
    def go_to_reset(self, joint_reset=False, replay_start_pose=None):
        """安全的宏观复位：纯关节空间移动"""

        intermediate_settle = (
            self.reset_intermediate_settle if self.reset_continuous_mode else True
        )
        # 1. 先安全退回到插槽正上方 (防止直接复位撞坏主板)
        if hasattr(self.config, "TOP_JOINTS"):
            self.interpolate_joint_move(
                self.config.TOP_JOINTS,
                timeout=1.5,
                settle=intermediate_settle,
                settle_timeout=0.0,
            )

        # 2. 如果有轨迹回放的起点，用笛卡尔微调过去
        if replay_start_pose is not None:
            self.interpolate_move(replay_start_pose, timeout=1.0, is_reset=True)
            self._pause_with_hold(
                self.reset_replay_align_dwell_sec,
                reason="go_to_reset replay_start_pose",
            )
            return

        # 3. 移动到初始待命点
        if hasattr(self.config, "RESET_JOINTS"):
            if self.reset_top_to_reset_linear and hasattr(self, "_joints_deg_to_pose6"):
                reset_pose = self._joints_deg_to_pose6(
                    np.array(self.config.RESET_JOINTS, dtype=np.float64)
                )
                reset_pose = self.clip_safety_box(np.array(reset_pose, dtype=np.float64))

                self.interpolate_move(
                    reset_pose,
                    timeout=max(0.2, float(self.reset_top_to_reset_timeout)),
                    is_reset=True,
                )
            else:
                final_settle = (not self.randomreset)
                self.interpolate_joint_move(
                    self.config.RESET_JOINTS,
                    timeout=2.0,
                    settle=final_settle,
                    settle_timeout=self.reset_final_settle_timeout if final_settle else 0.0,
                )


        if not self.randomreset:
            self._pause_with_hold(
                self.reset_after_reset_dwell_sec,
                reason="go_to_reset RESET_JOINTS",
            )

        # 4. 最后加上用于数据增强的微小随机偏移 (笛卡尔系)
        if self.randomreset:
            base_pose = (
                self._RESET_POSE.copy()
                if hasattr(self, "_RESET_POSE")
                else self.currpos.copy()
            )
            random_pose = base_pose.copy()
            random_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            random_pose[3:] = euler_random

            self.interpolate_move(random_pose, timeout=1.0, is_reset=True)
            self._pause_with_hold(
                self.reset_after_random_dwell_sec,
                reason="go_to_reset random_pose",
            )


    def go_to_rest(self):
        self._gripper_control(False)

        self._pause_with_hold(1.0, reason="quick_regrasp open gripper")

        print("[自动复位] 向上拔出到安全点...")
        if hasattr(self.config, "TOP_JOINTS"):
            self.interpolate_joint_move(
                self.config.TOP_JOINTS,
                timeout=1.5,
                settle=False,
                settle_timeout=0.0,
            )

    def quick_regrasp(self):
        """全自动流水线抓取：纯关节空间移动"""
        intermediate_settle = (
            self.reset_intermediate_settle if self.reset_continuous_mode else True
        )
        print("[自动复位] 张开夹爪...")
        self._gripper_control(False)
        self._pause_with_hold(1.0, reason="quick_regrasp open gripper")

        print("[自动复位] 向上拔出到安全点...")
        if hasattr(self.config, "TOP_JOINTS"):
            self.interpolate_joint_move(
                self.config.TOP_JOINTS,
                timeout=1.5,
                settle=intermediate_settle,
                settle_timeout=0.0,
            )
        self._pause_with_hold(
            self.quick_regrasp_dwell_top_sec,
            reason="quick_regrasp at TOP",
        )

        print("[自动复位] 下降到抓取点...")
        if hasattr(self.config, "TARGET_JOINTS"):
            self.interpolate_joint_move(
                self.config.TARGET_JOINTS,
                timeout=1.5,
                settle=intermediate_settle,
                settle_timeout=0.0,
            )
        self._pause_with_hold(
            self.quick_regrasp_dwell_target_sec,
            reason="quick_regrasp at TARGET",
        )

        print("[自动复位] 闭合夹爪...")
        self._gripper_control(True)
        self.last_gripper_act = time.time()
        self._pause_with_hold(1.5, reason="quick_regrasp close gripper")

        print("[自动复位] 抓取完毕，提起到安全点...")
        if hasattr(self.config, "TOP_JOINTS"):
            self.interpolate_joint_move(
                self.config.TOP_JOINTS,
                timeout=1.5,
                settle=intermediate_settle,
                settle_timeout=0.0,
            )
        self._pause_with_hold(
            self.quick_regrasp_dwell_final_top_sec,
            reason="quick_regrasp final TOP",
        )

    def reset_cycle(self):
        print('resetting cycle')
        self.rate_limiter = RateLimiter(self.config.CONTROL_FREQ)

    def reset(self, joint_reset=False, replay_start_pose=None, **kwargs):
        self._enter_reset_motion_mode(reason="TianjiEnv.reset")
        restored_mode = False
        try:
            self._stop_async_handoff_hold()
            self.last_gripper_act = time.time()
            if self.save_video:
                self.save_video_recording()

            if self.should_regrasp:
                self.regrasp()
                self.should_regrasp = False
            
            if True:
                self.quick_regrasp()

            self.go_to_reset(joint_reset=joint_reset, replay_start_pose=replay_start_pose)
            self.curr_path_length = 0

            if self.force_sensor is not None:
                self.force_sensor.reset_baseline()
                
            self._update_currpos()
            
            # 【关键修复】：将 cmd_pose 强制对齐到复位后的姿态！
            # 否则第二轮开始时，IK 收到的依然是你第一轮压到很低位置的旧指令，导致起步抽搐！
            self.cmd_pose = self.currpos.copy()
            self.nextpos = self.currpos.copy()
            # self._capture_post_reset_hold_target()
            self._post_reset_realign_pending = True
            # 先恢复到运行阶段控制模式，再做交接 hold，避免“reset return 后 finally 才切模式”的短暂掉力感
            # self._restore_control_mode_after_reset(reason="TianjiEnv.reset pre_handoff")
            restored_mode = True
            # self._pause_with_hold(
            #     self.reset_handoff_hold_sec,
            #     reason="reset handoff to step",
            # )
            # self._start_async_handoff_hold(max_sec=self.reset_handoff_max_sec)
            
            obs = self._get_obs()
            self.terminate = False
            self.max_distance = None
            return obs, {}
        finally:
            if not restored_mode:
                self._restore_control_mode_after_reset(reason="TianjiEnv.reset")
            self.reset_cycle()

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists("./videos"):
                    os.makedirs("./videos")

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                for camera_key in self.recording_frames[0].keys():
                    video_path = f"./videos/tianji_{camera_key}_{timestamp}.mp4"

                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]

                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )

                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])

                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")

            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if "serial_number" in kwargs:
                cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
                self.cap[cam_name] = cap
            elif "camera_index" in kwargs:
                cap = VideoCapture(FisheyeCapture(name=cam_name, **kwargs))
                self.cap[cam_name] = cap
            else:
                print(f"Can NOT figure camera config for: {cam_name}")


    def close_cameras(self):
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _send_pos_command(self, pos: np.ndarray, is_reset=False):
        if self.fake_env:
            return 0

        arr = np.array(pos, dtype=np.float64)
        target_tf = _pose6_to_transform(arr)
        self.right_target_pose = target_tf
        return self._solve_and_send_ik()

    def _send_joint_command(self, joint: np.ndarray):
        if self.fake_env:
            return 0

        arr = np.array(joint, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 7:
            print(f"Invalid joint command shape: {arr.shape}")
            return -1

        left_rad = self.controller.get_joint_pos_rad(arm_id=1)
        left_deg = np.array(left_rad) / self.controller.DEG_TO_RAD
        ok = self.controller.step(
            np.array(left_deg, dtype=np.float64),
            np.array(arr, dtype=np.float64),
            verbose=False,
        )
        self._update_currpos()
        self._mark_ik_seed_refresh("_send_joint_command")
        return 0 if ok else -1

    def _gripper_control(self, cmd: bool):
        """cmd=True close, cmd=False open."""
        if self.fake_env:
            return

        if not getattr(self, "_has_right_gripper_api", False):
            if not self._warned_gripper_unavailable:
                print(
                    "Tianji gripper API unavailable or disabled in MarvinArmController. "
                    "Gripper commands will be ignored."
                )
                self._warned_gripper_unavailable = True
            self.curr_gripper_pos = 0.0 if cmd else 1.0
            return

        try:
            if cmd:
                self.controller.right_gripper(False)
                self.curr_gripper_pos = 0.0
                
            else:
                self.controller.right_gripper(True)
                self.curr_gripper_pos = 1.0
                
        except Exception as e:
            print(f"ERROR when control Tianji gripper: {e}")

    def _send_gripper_command(self, pos: float, mode="binary"):
        if self.fake_env:
            return

        if mode == "binary":
            if (
                (pos <= -0.5)
                and (self.curr_gripper_pos > 0.85)
                and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):
                self._gripper_control(True)
                self.last_gripper_act = time.time()
                print("Gripper Close!")
                time.sleep(self.gripper_sleep)
            elif (
                (pos >= 0.5)
                and (self.curr_gripper_pos < 0.85)
                and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):
                self._gripper_control(False)
                self.last_gripper_act = time.time()
                print("Gripper Release!")
                time.sleep(self.gripper_sleep)
            else:
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    # right arm pose only
    def _compute_full_body_ik(
        self,
        current_ql: np.ndarray,
        current_qr: np.ndarray,
        force_seed_from_current: bool,
    ):
        if self.ik_client is not None:
            result = self.ik_client.solve(
                current_ql_rad=current_ql,
                current_qr_rad=current_qr,
                left_target_pose=self.left_target_pose,
                right_target_pose=self.right_target_pose,
                head_target_pose=self.head_target_pose,
                body_waist_q=self.body_waist_q,
                head_z_offset=self.head_z_offset,
                force_seed_from_current=force_seed_from_current,
            )
            ql_target = np.array(result["ql_target_rad"], dtype=np.float64)
            qr_target = np.array(result["qr_target_rad"], dtype=np.float64)
            return ql_target, qr_target

        left_hand_target = transform_to_wxyzxyz(self.left_target_pose)
        right_hand_target = transform_to_wxyzxyz(self.right_target_pose)
        head_target = transform_to_wxyzxyz(self.head_target_pose)

        left_pose_xyzwxyz = np.concatenate(
            [left_hand_target[4:7], left_hand_target[0:4]]
        )
        right_pose_xyzwxyz = np.concatenate(
            [right_hand_target[4:7], right_hand_target[0:4]]
        )
        head_pose_xyzwxyz = np.concatenate([head_target[4:7], head_target[0:4]])
        head_pose_xyzwxyz[2] += self.head_z_offset

        current_cfg = list(self.body_waist_q) + list(current_qr) + list(current_ql)
        if force_seed_from_current and hasattr(self.full_ik_solver, "sync_with_current_cfg"):
            self.full_ik_solver.sync_with_current_cfg(
                current_cfg[2:], reset_delta_reference=True
            )
            ik_seed_cfg = None
        else:
            ik_seed_cfg = current_cfg[2:]

        solver_cfg, _ = self.full_ik_solver.compute_ik(
            ik_seed_cfg,
            head_pose_xyzwxyz,
            right_pose_xyzwxyz,
            left_pose_xyzwxyz,
            force_seed_from_current=force_seed_from_current,
        )
        if solver_cfg is None:
            return None, None

        solver_cfg = np.concatenate([current_cfg[:2], solver_cfg])
        qr_target = solver_cfg[2:9]
        ql_target = solver_cfg[9:]
        return ql_target, qr_target

    def _solve_ik(self, pose):
        if self.fake_env:
            return None

        arr = np.array(pose, dtype=np.float64)
        target_tf = _pose6_to_transform(arr)
        self.right_target_pose = target_tf

        try:
            current_ql = self.controller.get_joint_pos_rad(arm_id=1)
            current_qr = self.controller.get_joint_pos_rad(arm_id=2)
            ql_target, qr_target = self._compute_full_body_ik(
                current_ql=current_ql,
                current_qr=current_qr,
                force_seed_from_current=False,
            )
            if qr_target is None:
                return None
            self._ik_need_seed_refresh = True

            joint_cmd_right = np.array(qr_target / self.controller.DEG_TO_RAD, dtype=np.float64)

            return joint_cmd_right
        except Exception as e:
            return None

    def _solve_and_send_ik(self):
        if self.fake_env:
            return 0

        try:
            current_ql = self.controller.get_joint_pos_rad(arm_id=1)
            current_qr = self.controller.get_joint_pos_rad(arm_id=2)
            force_seed_from_current = bool(self._ik_need_seed_refresh)
            ql_target, qr_target = self._compute_full_body_ik(
                current_ql=current_ql,
                current_qr=current_qr,
                force_seed_from_current=force_seed_from_current,
            )
            if ql_target is None or qr_target is None:
                return -1
            self._ik_need_seed_refresh = False

            joint_cmd_left = np.array(ql_target / self.controller.DEG_TO_RAD, dtype=np.float64)
            joint_cmd_right = np.array(qr_target / self.controller.DEG_TO_RAD, dtype=np.float64)
            self._debug_append_joint_sample(
                tag="ik_cmd",
                target_qr=joint_cmd_right,
                ql_deg=np.array(current_ql / self.controller.DEG_TO_RAD, dtype=np.float64),
                qr_deg=np.array(current_qr / self.controller.DEG_TO_RAD, dtype=np.float64),
            )

            ok = self.controller.step(joint_cmd_left, joint_cmd_right, verbose=False)
            self._debug_append_joint_sample(tag="ik_fb", target_qr=joint_cmd_right)
            return 0 if ok else -1
        except Exception as e:
            print(f"Failed to solve/send Tianji IK: {e}")
            return -1

    def _update_currpos(self):
        if self.fake_env:
            return

        current_ql = self.controller.get_joint_pos_rad(arm_id=1)
        current_qr = self.controller.get_joint_pos_rad(arm_id=2)

        left_fk = self.controller.compute_fk(current_ql)
        right_fk = self.controller.compute_fk(current_qr)
        if left_fk is None or right_fk is None:
            raise RuntimeError("Failed to compute FK from Tianji arm states.")

        current_left_pose = self.base_left_tf @ left_fk @ self.tool_tf
        current_right_pose = self.base_right_tf @ right_fk @ self.tool_tf

        if not self._left_pose_locked:
            self.left_target_pose = current_left_pose
            self._left_pose_locked = True

        if not self._head_pose_locked:
            self.head_target_pose = self.base_head_tf.copy()
            self._head_pose_locked = True

        self.right_target_pose = current_right_pose

        current_pose6 = _transform_to_pose6(current_right_pose)
        now = time.time()
        if self._last_pose6 is None or self._last_pose_t is None:
            self.currvel = np.zeros(6, dtype=np.float64)
        else:
            dt = max(1e-3, now - self._last_pose_t)
            self.currvel = (current_pose6 - self._last_pose6) / dt

        self._last_pose6 = current_pose6.copy()
        self._last_pose_t = now

        self.currjoint = np.array(current_qr, dtype=np.float64)
        self.currpos = current_pose6

        if self.force_sensor is not None:
            self.currforce = self.force_sensor.get_force_data()
        else:
            self.currforce = np.zeros(15, dtype=np.float64)
        self.currtorque = np.zeros(3, dtype=np.float64)

        self.q = np.zeros(6, dtype=np.float64)
        self.dq = np.zeros(6, dtype=np.float64)

    def update_currpos(self):
        self._update_currpos()

    def _get_obs(self) -> dict:
        images = self.get_im()
        # right arm eef force
        eef_force = self.controller.get_eef_force(arm_id=2)
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
            "eef_force": eef_force
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        self._stop_async_handoff_hold()
        if hasattr(self, "listener"):
            self.listener.stop()

        if hasattr(self, "cap") and self.cap is not None:
            self.close_cameras()

        if self.display_image and hasattr(self, "img_queue"):
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            if hasattr(self, "displayer"):
                self.displayer.join(timeout=1.0)

        if self.controller is not None:
            try:
                self.controller.cleanup()
            except Exception:
                pass
