"""Gym interface for Tianji MARVIN right arm on dual-arm humanoid platform."""

import copy
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
from multiprocess import Process, Queue

from franka_env.camera.fisheye_capture import FisheyeCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.envs.wow_skin import WowSkin
from scipy.linalg import expm


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
                    cv2.resize(v, (128, 128))
                    for k, v in img_array.items()
                    if "full" not in k
                ],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


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

try:
    from marvin_arm_controller import MarvinArmController
    from utils.kine_model import (
        transform_to_wxyzxyz,
        wxyzxyz_to_transform,
    )
    from utils.marvin_dual_arm_waist_teleop import DualArmWaistIK
except Exception as exc:  # pragma: no cover - runtime dependency check
    _MARVIN_IMPORT_ERROR = exc
    MarvinArmController = None
    DualArmWaistIK = None
    transform_to_wxyzxyz = None
    wxyzxyz_to_transform = None


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

    ACTION_SCALE = np.zeros((3,))
    BASIC_JOINT_RESET = np.zeros((7,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0
    ABS_POSE_LIMIT_HIGH = np.array([0.85, 0.35, 1.50, np.pi, np.pi, np.pi])
    ABS_POSE_LIMIT_LOW = np.array([-0.20, -0.55, 0.40, -np.pi, -np.pi, -np.pi])

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
        self._warned_gripper_unavailable = False

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
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0,
                            255,
                            shape=(128, 128, 3),
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

    def _set_default_task_poses_from_current(self):
        curr = self.currpos.copy()

        def _resolve(pose_like):
            pose = np.array(pose_like, dtype=np.float64)
            if pose.shape != (6,) or np.all(np.abs(pose) < 1e-8):
                return curr.copy()
            return pose

        self._TARGET_POSE = _resolve(self._TARGET_POSE)
        self._GRASP_POSE = _resolve(self._GRASP_POSE)
        self._RESET_POSE = _resolve(self._RESET_POSE)
        self.resetpos = self._RESET_POSE.copy()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
        action = np.clip(action, self.action_space.low, self.action_space.high)

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
        T_target_new = T_relative @ T_target

        # 5. 还原回 6D pose 并更新 cmd_pose
        self.nextpos = _transform_to_pose6(T_target_new)
        
        # self.cmd_pose = self.nextpos.copy()

        # 6. 发送指令
        gripper_action = action[6] * self.action_scale[2]
        self._send_gripper_command(gripper_action)
        
        # 使用修复后的 clip_safety_box 过滤后再发送给 IK
        safe_target = self.clip_safety_box(self.nextpos)
        # 【关键修复】：必须将 cmd_pose 强制对齐到 safe_target！
        # 彻底解决碰到边界后“积分饱和”导致按键失灵的问题
        self.cmd_pose = safe_target.copy()

        self._send_pos_command(safe_target)

        self.curr_path_length += 1
        _ = time.time() - start_time

        self._update_currpos()
        ob = self._get_obs()
        reward, finish = 0, False
        done = (
            self.curr_path_length >= self.max_episode_length
            or finish
            or getattr(self, "terminate", False)
        )

        return ob, reward, done, False, {"succeed": finish}

    def compute_reward(self, obs):
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

    def interpolate_move(self, goal: np.ndarray, timeout: float, is_reset=False):
        steps = max(2, int(timeout * self.hz))
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p, is_reset=is_reset)
            self._update_currpos()
        self.nextpos = path[-1]
        self._update_currpos()

    def go_to_reset(self, joint_reset=False, replay_start_pose=None):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """     
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos, is_reset=True)
        time.sleep(0.3)

        # pull up (改为插值平滑上升，避免瞬间抽动)
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = reset_pose[2] + 0.07
        self.interpolate_move(reset_pose, timeout=0.5, is_reset=True)
        self._update_currpos()
        
        if replay_start_pose is not None:
            self.interpolate_move(replay_start_pose, timeout=1.0, is_reset=True)
            time.sleep(0.5)
            return

        # perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            self._send_joint_command(self._BASIC_JOINT_RESET)
            time.sleep(0.5)
            return

        # perform Cartesian reset
        reset_pose = self.resetpos.copy()
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_random
            
        # 【关键修复】：使用 interpolate_move 平滑移动到复位点，不要用 _send_pos_command 硬跳
        self.interpolate_move(reset_pose, timeout=1.0, is_reset=True)
        time.sleep(0.5)

    def reset(self, joint_reset=False, replay_start_pose=None, **kwargs):
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
        
        obs = self._get_obs()
        self.terminate = False
        self.max_distance = None
        return obs, {}

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

    def _solve_and_send_ik(self):
        if self.fake_env:
            return 0

        try:
            current_ql = self.controller.get_joint_pos_rad(arm_id=1)
            current_qr = self.controller.get_joint_pos_rad(arm_id=2)

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
            solver_cfg, _ = self.full_ik_solver.compute_ik(
                current_cfg[2:],
                head_pose_xyzwxyz,
                right_pose_xyzwxyz,
                left_pose_xyzwxyz,
            )

            if solver_cfg is None:
                return -1

            solver_cfg = np.concatenate([current_cfg[:2], solver_cfg])
            qr_target = solver_cfg[2:9]
            ql_target = solver_cfg[9:]

            joint_cmd_left = np.array(ql_target / self.controller.DEG_TO_RAD, dtype=np.float64)
            joint_cmd_right = np.array(qr_target / self.controller.DEG_TO_RAD, dtype=np.float64)

            ok = self.controller.step(joint_cmd_left, joint_cmd_right, verbose=False)
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
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
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
