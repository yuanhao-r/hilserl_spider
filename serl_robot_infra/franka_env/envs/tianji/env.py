"""Minimal Tianji Gym env backed by tlop joint control."""

from __future__ import annotations

import copy
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Sequence

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from franka_env.camera.fisheye_capture import FisheyeCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.envs.tianji.backend import TianjiBackendConfig, TianjiTlopBackend
from franka_env.envs.tianji.kinematics import (
    TianjiKinematics,
    TianjiKinematicsConfig,
    pose6_to_transform,
    transform_to_pose6,
)

IMAGE_SIZE = (128, 128)


class ImageDisplayer(threading.Thread):
    def __init__(self, queue_obj, name):
        super().__init__(daemon=True)
        self.queue = queue_obj
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            frame = np.concatenate(
                [cv2.resize(v, IMAGE_SIZE) for k, v in img_array.items() if "full" not in k],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


@dataclass
class DefaultTianjiEnvConfig:
    server_url: str = "10.10.13.10"
    robot_ip: str = "10.10.13.10"
    realsense_cameras: Dict = field(default_factory=dict)
    image_crop: Dict[str, callable] = field(default_factory=dict)
    action_scale: np.ndarray = field(
        default_factory=lambda: np.array([0.005, 0.005, 1.0], dtype=np.float64)
    )
    abs_pose_limit_low: np.ndarray = field(
        default_factory=lambda: np.array([0.7505, -0.1617, 0.5230, -np.pi, -np.pi, -np.pi], dtype=np.float64)
    )
    abs_pose_limit_high: np.ndarray = field(
        default_factory=lambda: np.array([1.8005, -0.08217, 1.6230, np.pi, np.pi, np.pi], dtype=np.float64)
    )
    reset_joints: np.ndarray | None = None
    random_reset: bool = False
    random_xy_range: float = 0.0
    random_rz_range: float = 0.0
    random_dx_min: float = 0.0
    random_dx_max: float = 0.0
    random_dy_min: float = 0.0
    random_dy_max: float = 0.0
    random_dz_min: float = 0.0
    random_dz_max: float = 0.0
    reset_wait_sec: float = 2.0
    reset_hold_hz: float = 40.0
    random_reset_wait_sec: float = 1.0
    control_hz: float = 10.0
    gripper_sleep: float = 0.6
    max_episode_length: int = 200
    display_image: bool = False
    fake_env: bool = False
    backend: TianjiBackendConfig = field(default_factory=TianjiBackendConfig)
    kinematics: TianjiKinematicsConfig = field(default_factory=TianjiKinematicsConfig)

    # Compatibility aliases for existing HIL-SERL config style.
    def __post_init__(self):
        alias_pairs = (
            ("SERVER_URL", "server_url"),
            ("ROBOT_IP", "robot_ip"),
            ("REALSENSE_CAMERAS", "realsense_cameras"),
            ("IMAGE_CROP", "image_crop"),
            ("ACTION_SCALE", "action_scale"),
            ("ABS_POSE_LIMIT_LOW", "abs_pose_limit_low"),
            ("ABS_POSE_LIMIT_HIGH", "abs_pose_limit_high"),
            ("RESET_JOINTS", "reset_joints"),
            ("RESET_POSE", "reset_pose"),
            ("RANDOM_RESET", "random_reset"),
            ("RANDOM_XY_RANGE", "random_xy_range"),
            ("RANDOM_RZ_RANGE", "random_rz_range"),
            ("RANDOM_DX_MIN", "random_dx_min"),
            ("RANDOM_DX_MAX", "random_dx_max"),
            ("RANDOM_DY_MIN", "random_dy_min"),
            ("RANDOM_DY_MAX", "random_dy_max"),
            ("RANDOM_DZ_MIN", "random_dz_min"),
            ("RANDOM_DZ_MAX", "random_dz_max"),
            ("RESET_WAIT_SEC", "reset_wait_sec"),
            ("RESET_HOLD_HZ", "reset_hold_hz"),
            ("RANDOM_RESET_WAIT_SEC", "random_reset_wait_sec"),
            ("CONTROL_HZ", "control_hz"),
            ("CONTROL_FREQ", "control_hz"),
            ("GRIPPER_SLEEP", "gripper_sleep"),
            ("MAX_EPISODE_LENGTH", "max_episode_length"),
            ("DISPLAY_IMAGE", "display_image"),
        )
        for old_name, new_name in alias_pairs:
            if hasattr(self, old_name):
                setattr(self, new_name, getattr(self, old_name))

        backend_aliases = {
            "TLOP_TRANSPORT": "transport",
            "TLOP_UDP_HOST": "udp_host",
            "TLOP_UDP_PORT": "udp_port",
            "TLOP_WAIT_READY_TIMEOUT": "wait_ready_timeout",
            "TLOP_SERVICE_TIMEOUT": "service_timeout",
            "TLOP_POSE_TIMEOUT": "pose_timeout",
            "TLOP_GRIPPER_TIMEOUT": "gripper_timeout",
        }
        for old_name, new_name in backend_aliases.items():
            if hasattr(self, old_name):
                setattr(self.backend, new_name, getattr(self, old_name))

        kin_aliases = {
            "ROBOT_IP": "robot_ip",
            "MARVIN_URDF_PATH": "marvin_urdf_path",
            "DUAL_IK_URDF_PATH": "dual_ik_urdf_path",
            "RIGHT_ARM_BASE_POSE_WXYZXYZ": "right_arm_base_pose_wxyzxyz",
            "LEFT_ARM_BASE_POSE_WXYZXYZ": "left_arm_base_pose_wxyzxyz",
            "HEAD_BASE_POSE_WXYZXYZ": "head_base_pose_wxyzxyz",
            "TOOL_Z_OFFSET": "tool_z_offset",
            "HEAD_Z_OFFSET": "head_z_offset",
        }
        for old_name, new_name in kin_aliases.items():
            if hasattr(self, old_name):
                setattr(self.kinematics, new_name, getattr(self, old_name))


class TianjiEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultTianjiEnvConfig | None = None,
        set_load=False,
    ):
        del set_load
        self.config = config or DefaultTianjiEnvConfig()
        if hasattr(self.config, "__post_init__"):
            self.config.__post_init__()

        self.fake_env = bool(fake_env or getattr(self.config, "fake_env", False))
        self.save_video = bool(save_video)
        self.hz = float(getattr(self.config, "control_hz", hz))
        self.action_scale = np.asarray(self.config.action_scale, dtype=np.float64)
        self.max_episode_length = int(self.config.max_episode_length)
        self.display_image = bool(self.config.display_image)
        self.gripper_sleep = float(self.config.gripper_sleep)
        self.reset_joints = (
            None
            if self.config.reset_joints is None
            else np.asarray(self.config.reset_joints, dtype=np.float64).reshape(7)
        )
        # self.resetpos = np.asarray(self.config.reset_pose, dtype=np.float64)
        self.randomreset = bool(self.config.random_reset)
        self.random_xy_range = float(self.config.random_xy_range)
        self.random_rz_range = float(self.config.random_rz_range)

        self.backend = None
        self.kinematics = None
        self.currjoint = np.zeros(7, dtype=np.float64)
        self.currpos = np.zeros(6, dtype=np.float64)
        self.currvel = np.zeros(6, dtype=np.float64)
        self.currforce = np.zeros(15, dtype=np.float64)
        self.currtorque = np.zeros(3, dtype=np.float64)
        self.curr_gripper_pos = 1.0
        self.cmd_pose = np.zeros(6, dtype=np.float64)
        self.nextpos = np.zeros(6, dtype=np.float64)
        self._last_pose6 = None
        self._last_pose_t = None
        self._last_sent_left_rad = None
        self._last_sent_right_rad = None
        self._ik_need_seed_refresh = True
        self.curr_path_length = 0
        self.terminate = False
        self.last_gripper_act = time.time()
        self._last_step_t = 0.0
        if not self.fake_env:
            self.backend = TianjiTlopBackend(self.config.backend)
            self.backend.connect()
            self.kinematics = TianjiKinematics(self.config.kinematics)
            self._update_currpos()
          # if self.reset_joints is None:
            # raise ValueError("TianjiEnv requires RESET_JOINTS to compute reset pose.")
            self.resetpos = self.kinematics.pose_from_right_joints_deg(self.reset_joints)
            # self.resetpos = self.kinematics.
            print(
                "[TianjiEnv] reset pose from RESET_JOINTS "
                f"{np.round(self.resetpos, 4).tolist()}"
            )
        self.cmd_pose = self.currpos.copy()
        self.nextpos = self.currpos.copy()

        self.xyz_bounding_box = gym.spaces.Box(
            np.asarray(self.config.abs_pose_limit_low[:3], dtype=np.float64),
            np.asarray(self.config.abs_pose_limit_high[:3], dtype=np.float64),
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.asarray(self.config.abs_pose_limit_low[3:], dtype=np.float64),
            np.asarray(self.config.abs_pose_limit_high[3:], dtype=np.float64),
            dtype=np.float64,
        )

        self.action_space = gym.spaces.Box(
            -np.ones((7,), dtype=np.float32),
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
                        key: gym.spaces.Box(0, 255, shape=(*IMAGE_SIZE, 3), dtype=np.uint8)
                        for key in self.config.realsense_cameras
                    }
                ),
            }
        )

        self.recording_frames = []
        self.cap = OrderedDict()
        if not self.fake_env:
            self.init_cameras(self.config.realsense_cameras)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "TianjiEnv")
            self.displayer.start()

        print("Initialized minimal TianjiEnv with tlop backend.")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        clipped = np.asarray(pose, dtype=np.float64).copy()
        clipped[:3] = np.clip(clipped[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        clipped[3:] = np.clip(clipped[3:], self.rpy_bounding_box.low, self.rpy_bounding_box.high)
        return clipped

    def _rate_sleep(self) -> None:
        if self.hz <= 0:
            return
        now = time.perf_counter()
        if self._last_step_t > 0:
            dt = 1.0 / self.hz - (now - self._last_step_t)
            if dt > 0:
                time.sleep(dt)
        self._last_step_t = time.perf_counter()

    def _pose_after_action(self, action: np.ndarray) -> np.ndarray:
        pose = self.cmd_pose.copy()
        pose[:3] += action[:3] * float(self.action_scale[0])
        if self.action_scale.shape[0] > 1 and abs(float(self.action_scale[1])) > 1e-9:
            delta_rot = Rotation.from_euler("xyz", action[3:6] * float(self.action_scale[1]))
            pose[3:] = (delta_rot * Rotation.from_euler("xyz", self.cmd_pose[3:])).as_euler("xyz")
        return pose

    def step(self, action: np.ndarray) -> tuple:
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(7), -1.0, 1.0)
        self._rate_sleep()

        self.nextpos = self._pose_after_action(action)
        # print("nextpos:", self.nextpos, flush=True)
        # TODO 不要高频开关 后续需要重构这部分代码。 当前任务不需要开
        # gripper_action = action[6] * float(self.action_scale[2] if self.action_scale.shape[0] > 2 else 1.0)
        # self._send_gripper_command(gripper_action)

        ret, ik_error = self._send_pos_command(self.nextpos)
        if ret == 0:
            self.cmd_pose = self.nextpos.copy()
        else:
            self._update_currpos()
            self.cmd_pose = self.currpos.copy()
            self.nextpos = self.currpos.copy()

        self.curr_path_length += 1
        self._update_currpos()
        obs = self._get_obs()
        done = bool(self.curr_path_length >= self.max_episode_length or self.terminate)
        return obs, 0.0, done, False, {"succeed": False}
    
    
    def reset(self, joint_reset=False, replay_start_pose=None, **kwargs):
        if self.save_video:
            self.save_video_recording()
        self.last_gripper_act = time.time()
        
        self.go_to_reset()

        self._update_currpos()
        self.curr_path_length = 0

        self.cmd_pose = self.currpos.copy()
        self.nextpos = self.currpos.copy()
        # self._ik_need_seed_refresh = True
        # self._last_step_t = 0.0
        return self._get_obs(), {"succeed": False}
    
    def go_to_reset(self):
        base_pose = self.resetpos.copy()

        if self.randomreset:
            target_pose = base_pose.copy()
            target_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            target_pose[3:] = base_pose[3:].copy()
            target_pose[5] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
        else:
            target_pose = base_pose.copy()

        print(f"[TianjiEnv] reset target pose {np.round(target_pose, 4).tolist()}")
        ret, ik_error = self._send_pos_command(
            target_pose,
            force_seed_from_current=True,
            reset_stats=True,
        )
        while ret != 0 and ik_error > 0.0001:
            ret, ik_error = self._send_pos_command(
                target_pose,
                force_seed_from_current=False,
                reset_stats=True,
            )
        time.sleep(float(self.config.reset_wait_sec))

    def _move_to_right_joints_deg(self, right_joints_deg: np.ndarray, hold_sec: float = 0.0) -> None:
        right_rad = np.deg2rad(np.asarray(right_joints_deg, dtype=np.float64).reshape(7))
        self.backend.send_joints_rad(None, right_rad)
        self._last_sent_left_rad = None
        self._last_sent_right_rad = right_rad.copy()
        self._ik_need_seed_refresh = True
        # self._hold_current_target(hold_sec, right_rad=right_rad)

    def _hold_current_target(self, hold_sec: float, right_rad: np.ndarray | None = None) -> None:
        if self.fake_env or self.backend is None or hold_sec <= 0:
            return
        hz = max(1.0, float(self.config.reset_hold_hz))
        period = 1.0 / hz
        end_t = time.perf_counter() + float(hold_sec)
        while time.perf_counter() < end_t:
            _, right_now = self.backend.get_joints_rad()
            right_target = (
                right_rad
                if right_rad is not None
                else self._last_sent_right_rad
                if self._last_sent_right_rad is not None
                else right_now
            )
            self.backend.send_joints_rad(None, right_target)
            time.sleep(period)

    def _compute_full_body_ik(
        self,
        current_ql: np.ndarray,
        current_qr: np.ndarray,
        force_seed_from_current: bool,
    ):
        return self.kinematics.compute_full_body_ik(
            current_left_rad=current_ql,
            current_right_rad=current_qr,
            force_seed_from_current=force_seed_from_current,
        )

    def _solve_ik(self, pose):
        arr = np.array(pose, dtype=np.float64)
        self.kinematics.right_target_pose = pose6_to_transform(arr)

        try:
            current_ql, current_qr = self.backend.get_joints_rad()
            ql_target, qr_target = self._compute_full_body_ik(
                current_ql=current_ql,
                current_qr=current_qr,
                force_seed_from_current=False,
            )
            if qr_target is None:
                return None
            self._ik_need_seed_refresh = True
            return np.rad2deg(np.asarray(qr_target, dtype=np.float64))
        except Exception:
            return None

    def _solve_and_send_ik(self, reset_stats=False):
        try:
            current_ql, current_qr = self.backend.get_joints_rad()
            force_seed_from_current = bool(self._ik_need_seed_refresh)
            ql_target, qr_target, ik_error = self._compute_full_body_ik(
                current_ql=current_ql,
                current_qr=current_qr,
                force_seed_from_current=force_seed_from_current,
            )
            if ql_target is None or qr_target is None:
                return -1, ik_error
            self._ik_need_seed_refresh = False

            ok = self.backend.send_joints_rad(None, qr_target)
            if not ok:
                print(f"[TianjiEnv] tlop step failed: {self.backend.last_error}")
                return -1, ik_error

            self._last_sent_left_rad = None
            self._last_sent_right_rad = np.asarray(qr_target, dtype=np.float64).copy()
            if reset_stats:
                print(f"[_last_sent_right_rad] sent IK command: {np.round(qr_target, 4).tolist()}")
            return 0, ik_error
        except Exception as exc:
            print(f"Failed to solve/send Tianji IK: {exc}")
            return -1, 100

    def _send_pos_command(self, pose: np.ndarray, force_seed_from_current: bool | None = False, reset_stats=False) -> int:
        self.kinematics.right_target_pose = pose6_to_transform(pose)
        if force_seed_from_current is not None:
            self._ik_need_seed_refresh = bool(force_seed_from_current)
        return self._solve_and_send_ik(reset_stats=reset_stats)

    def _send_gripper_command(self, pos: float, mode="binary") -> None:
        if self.fake_env or mode != "binary":
            return
        now = time.time()
        if pos < 0.5 and now - self.last_gripper_act > self.gripper_sleep:
            if self.backend.set_gripper_closed(True):
                self.curr_gripper_pos = 0.0
            else:
                print(f"[TianjiEnv] gripper close failed: {self.backend.last_error}")
            self.last_gripper_act = now
        elif pos >= 0.5  and now - self.last_gripper_act > self.gripper_sleep:
            if self.backend.set_gripper_closed(False):
                self.curr_gripper_pos = 1.0
            else:
                print(f"[TianjiEnv] gripper open failed: {self.backend.last_error}")
            self.last_gripper_act = now

    def _update_currpos(self) -> None:

        left_rad, right_rad = self.backend.get_joints_rad()
        pose6 = self.kinematics.fk_right_pose6(right_rad)
        self.kinematics.update_locked_left_head(left_rad)
        self.kinematics.right_target_pose = pose6_to_transform(pose6)
        now = time.time()
        if self._last_pose6 is None or self._last_pose_t is None:
            self.currvel = np.zeros(6, dtype=np.float64)
        else:
            self.currvel = (pose6 - self._last_pose6) / max(1e-3, now - self._last_pose_t)
        self._last_pose6 = pose6.copy()
        self._last_pose_t = now
        self.currjoint = np.asarray(right_rad, dtype=np.float64)
        self.currpos = pose6
        self.currforce = np.zeros(15, dtype=np.float64)
        self.currtorque = np.zeros(3, dtype=np.float64)

    def update_currpos(self) -> None:
        self._update_currpos()

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos.copy(),
            "tcp_vel": self.currvel.copy(),
            "gripper_pose": np.array([self.curr_gripper_pos], dtype=np.float64),
            "tcp_force": self.currforce.copy(),
            "tcp_torque": self.currtorque.copy(),
            "eef_force": np.zeros(6, dtype=np.float64),
        }
        return copy.deepcopy({"images": images, "state": state_observation})

    def get_im(self) -> Dict[str, np.ndarray]:
        images = {}
        display_images = {}
        full_res_images = {}
        if self.fake_env or not self.cap:
            for key, space in self.observation_space["images"].spaces.items():
                images[key] = np.zeros(space.shape, dtype=np.uint8)
            return images

        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.image_crop[key](rgb) if key in self.config.image_crop else rgb
                resized = cv2.resize(cropped_rgb, self.observation_space["images"][key].shape[:2][::-1])
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[f"{key}_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                input(f"{key} camera frozen. Check connection and press enter to retry...")
                cap.close()
                self.init_cameras(self.config.realsense_cameras)
                return self.get_im()

        if self.save_video:
            self.recording_frames.append(full_res_images)
        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def init_cameras(self, name_serial_dict=None) -> None:
        self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in (name_serial_dict or {}).items():
            if "serial_number" in kwargs:
                self.cap[cam_name] = VideoCapture(RSCapture(name=cam_name, **kwargs))
            elif "camera_index" in kwargs:
                self.cap[cam_name] = VideoCapture(FisheyeCapture(name=cam_name, **kwargs))
            else:
                print(f"Can NOT figure camera config for: {cam_name}")

    def close_cameras(self) -> None:
        try:
            for cap in getattr(self, "cap", {}).values():
                cap.close()
        except Exception as exc:
            print(f"Failed to close cameras: {exc}")

    def set_display_overlay(self, lines: Sequence[str] | None) -> None:
        del lines

    def save_video_recording(self) -> None:
        if not self.recording_frames:
            return
        os.makedirs("./videos", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for camera_key in self.recording_frames[0].keys():
            video_path = f"./videos/tianji_{camera_key}_{timestamp}.mp4"
            first_frame = self.recording_frames[0][camera_key]
            height, width = first_frame.shape[:2]
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (width, height),
            )
            for frame_dict in self.recording_frames:
                writer.write(frame_dict[camera_key])
            writer.release()
            print(f"Saved video for camera {camera_key} at {video_path}")
        self.recording_frames.clear()

    def close(self) -> None:
        self.close_cameras()
        if self.display_image and hasattr(self, "img_queue"):
            self.img_queue.put(None)
            cv2.destroyAllWindows()
        if self.backend is not None:
            self.backend.close()
        if self.kinematics is not None:
            self.kinematics.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
