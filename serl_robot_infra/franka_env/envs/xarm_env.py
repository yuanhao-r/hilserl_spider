"""Gym Interface for Realman"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.fisheye_capture import FisheyeCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler

from xarm.wrapper import XArmAPI
from franka_env.envs.wow_skin import WowSkin


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break
            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultXArmEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "192.168.1.232"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: dict[str, callable] = {}
    WOWSKIN_PORT: str = "/dev/ttyACM0"
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    BASIC_JOINT_RESET = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class XArmEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultXArmEnvConfig = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._GRASP_POSE = config.GRASP_POSE
        self._BASIC_JOINT_RESET = config.BASIC_JOINT_RESET
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = config.RESET_POSE
        self.fake_env = fake_env
        
        self.max_distance = None
        self.last_distance = None
        
        if not self.fake_env:
            self.arm = XArmAPI(self.url)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(7)
            self.arm.set_state(state=0)
            self.arm_mode = 7
            
            self.arm.set_tgpio_modbus_baudrate(baud=115200)
            
            self.curr_gripper_pos = 1.0
            
            if config.WOWSKIN_PORT is not None:
                self.force_sensor = WowSkin(port=config.WOWSKIN_PORT)
                self.force_sensor.reset_baseline()
            else:
                self.force_sensor = None
            self._update_currpos()
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),  # xyz + rpy
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(15,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for key in config.REALSENSE_CAMERAS}
                ),
            }
        )
        self.cycle_count = 0

        if self.fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        if set_load:
            # input("Put arm into programing mode and press enter.")
            # requests.post(self.url + "set_load", json=self.config.LOAD_PARAM)
            # input("Put arm into execution mode and press enter.")
            # for _ in range(2):
            #     self._recover()
            #     time.sleep(1)
            pass

        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("Initialized Xarm.")
    
    def __del__(self):
        self.arm.disconnect()

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = pose[3:]

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                min(np.abs(self.rpy_bounding_box.low[0]),np.abs(self.rpy_bounding_box.high[0])),
                max(np.abs(self.rpy_bounding_box.low[0]),np.abs(self.rpy_bounding_box.high[0]))
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = euler

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        if np.all(action[3:6] < 0.0001) or abs(self.action_scale[1]) < 0.0001:
            self.nextpos[3:] = self.currpos[3:]
        else:
            self.nextpos[3:] = (
                Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
                * Rotation.from_euler("xyz", self.currpos[3:])
            ).as_euler("xyz")

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        # time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward, finish = 0, False # self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or finish or self.terminate
        
        return ob, reward, done, False, {"succeed": finish}

    def compute_reward(self, obs):
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_euler("xyz", current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        distance = np.linalg.norm(delta)
        if self.max_distance is None:
            self.max_distance = distance
            self.last_distance = distance
            return 0, False
        else:
            if distance >= self.last_distance:
                reward = 0
            else:
                reward = (self.last_distance - distance) / self.max_distance
                self.last_distance = distance
            return reward, False
            # if np.all(delta < self._REWARD_THRESHOLD):
            #     return True
            # else:
            #     # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            #     return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float, is_reset=False):
        """Move the robot to the goal position with linear interpolation."""
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p, is_reset=is_reset)
            self._update_currpos()
            # time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            reset_pose = self.resetpos.copy()
            self._send_joint_command(reset_pose)
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_random
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._gripper_control(False)
        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "192.168.1.18":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
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
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if "serial_number" in kwargs:
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap
            elif "camera_index" in kwargs:
                cap = VideoCapture(
                    FisheyeCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap
            else:
                print(f"Can NOT figure camera config for: {cam_name}")

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _send_pos_command(self, pos: np.ndarray, is_reset=False):
        """Internal function to send position command to the robot."""
        arr = np.array(pos).astype(np.float32)
        
        arr[:3] *= 1000 # xarm control in mm, tanslate m to mm
        
        if is_reset:
            speed = 120
            if self.arm_mode != 0:
                self.arm.set_mode(0)
                self.arm.set_state(state=0)
                self.arm_mode = 0
            ret = self.arm.set_position(x=arr[0], y=arr[1], z=arr[2], roll=arr[3], pitch=arr[4], yaw=arr[5], speed=speed, wait=True, is_radian=True)
        else:
            speed = 40
            if self.arm_mode != 7:
                self.arm.set_mode(7)
                self.arm.set_state(state=0)
                self.arm_mode = 7
            ret = self.arm.set_position(x=arr[0], y=arr[1], z=arr[2], roll=arr[3], pitch=arr[4], yaw=arr[5], speed=speed, wait=False, is_radian=True)
            time.sleep(0.05)
        return ret
        
    def _send_joint_command(self, joint: np.ndarray):
        """Internal function to send joint command to the robot."""
        arr = np.array(joint).astype(np.float32)
        if self.arm_mode != 0:
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            self.arm_mode = 0
        speed = 50
        self.arm.set_servo_angle(angle=arr.tolist(), speed=speed, wait=True)
        
    def _gripper_control(self, cmd: bool):
        '''cmd: True for close, False for release.'''
        if self.fake_env:
            return
        
        try:
            if cmd:
                code, ret = self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
            else:
                code, ret = self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])

            code, ret = self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        except Exception as e:
            print(f"ERROR when control gripper: {e}")
            return
        
        if cmd:
            self.curr_gripper_pos = 0.0
        else:
            self.curr_gripper_pos = 1.0

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if self.fake_env:
            return
        
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                self._gripper_control(True)
                self.last_gripper_act = time.time()
                print("Gripper Close!")
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                self._gripper_control(False)
                self.last_gripper_act = time.time()
                print("Gripper Release!")
                time.sleep(self.gripper_sleep)
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ret, joint_state = self.arm.get_joint_states(is_radian=False)
        self.currjoint = np.array(joint_state[0][:6])
        
        ret, pos = self.arm.get_position(is_radian=True)
        self.currpos = np.array(pos)
        self.currvel = np.array(joint_state[1][:6])
        
        # xarm control in mm, tanslate m to mm
        self.currpos[:3] /= 1000.0
        self.currvel[:3] /= 1000.0

        if self.force_sensor is not None:
            self.currforce = self.force_sensor.get_force_data()
        else:
            self.currforce = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.currtorque = np.array([0, 0, 0]) # TODO
        # self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array([0, 0, 0, 0, 0, 0]) # TODO
        self.dq = np.array([0, 0, 0, 0, 0, 0]) # TODO

        # ret, gripper_state = self.arm.rm_get_gripper_state()
        # self.curr_gripper_pos = np.array(gripper_state["actpos"] / 100.0) # resize to [0, 1]

    def update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

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
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()