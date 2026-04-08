import copy
import os
import time
import numpy as np
from pynput import keyboard

from franka_env.envs.xarm_env import XArmEnv
from franka_env.envs.tianji_env import TianjiEnv

ROBOT_BACKEND = os.environ.get("HILSERL_ARM_BACKEND", "tianji").lower()

if ROBOT_BACKEND in {"tianji", "marvin"}:
    BaseRAMRobotEnv = TianjiEnv
else:
    BaseRAMRobotEnv = XArmEnv


class RAMEnv(BaseRAMRobotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False
        self.auto_quick_regrasp = bool(getattr(self.config, "AUTO_QUICK_REGRASP", False))
        self._gripper_control(True)

        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()

    def go_to_reset(self, joint_reset=False, replay_start_pose=None):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """     
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos, is_reset=True)
        time.sleep(0.3)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = reset_pose[2] + 0.07
        self._send_pos_command(reset_pose, is_reset=True)
        # self.interpolate_move(reset_pose, timeout=0.2, is_reset=True)
        self._update_currpos()
        
        if replay_start_pose is not None:
            ret = self._send_pos_command(replay_start_pose, is_reset=True)
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
        ret = self._send_pos_command(reset_pose, is_reset=True)
        if ret != 0:
            self._send_joint_command(self._BASIC_JOINT_RESET)
        
        time.sleep(0.5)


    def regrasp(self):
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = reset_pose[2] + 0.07
        self._send_pos_command(reset_pose, is_reset=True)
        # self.interpolate_move(reset_pose, timeout=1, is_reset=True)

        input("Press enter to release gripper...")
        self._send_gripper_command(1.0)
        input("Place RAM in holder and press enter to grasp...")
        top_pose = self._GRASP_POSE.copy()
        top_pose[2] += 0.1
        self._send_pos_command(top_pose, is_reset=True)
        time.sleep(0.5)

        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.1
        self._send_pos_command(grasp_pose, is_reset=True)

        self._send_gripper_command(-1.0)
        print("Regrasp Done!")
        self.last_gripper_act = time.time()
        time.sleep(1.0)
        
    def quick_regrasp(self):
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        
        self._gripper_control(False)
        time.sleep(1.5)

        top_pose = self._GRASP_POSE.copy()
        top_pose[2] += 0.1
        self._send_pos_command(top_pose, is_reset=True)
        time.sleep(2.0)

        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.1
        self._send_pos_command(grasp_pose, is_reset=True)

        self._gripper_control(True)
        self.last_gripper_act = time.time()
        time.sleep(1.5)

    def reset(self, joint_reset=False, replay_start_pose=None, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # if True:
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False
        
        if self.auto_quick_regrasp:
            self.quick_regrasp()

        self.go_to_reset(joint_reset=joint_reset, replay_start_pose=replay_start_pose)
        self.curr_path_length = 0

        if self.force_sensor is not None:
            self.force_sensor.reset_baseline()
        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        self.max_distance = None
        return obs, {}
