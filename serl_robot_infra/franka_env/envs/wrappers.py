import time
import os
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import requests
from scipy.spatial.transform import Rotation as R
from franka_env.envs.franka_env import FrankaEnv
from typing import List

from pynput import keyboard
import pickle as pkl
import random

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def _apply_deadband(action: np.ndarray, deadband: float) -> np.ndarray:
    filtered = np.array(action, dtype=np.float32, copy=True)
    filtered[np.abs(filtered) < deadband] = 0.0
    return filtered

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.success_key = False
        self.listener = keyboard.Listener(
            on_press=self.on_press)
        self.listener.start()
        
    def on_press(self, key):
        try:
            if str(key) == 'Key.f4':
                self.success_key = True
        except AttributeError:
            pass
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if self.success_key:
            rew += 1
            done = True
            self.success_key = False
        else:
            rew = rew
        # if done:
        #     while True:
        #         try:
        #             rew = int(input("Success? (1/0)"))
        #             assert rew == 0 or rew == 1
        #             break
        #         except:
        #             continue
        info['succeed'] = (rew >= 1)
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.success_key = False
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (6,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        # tcp_pose = observation["state"]["tcp_pose"]
        # observation["state"]["tcp_pose"] = np.concatenate(
        #     (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        # )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        # self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = -np.ones((7,), dtype=np.float32)
        new_action[:6] = action[:6].copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(
        self,
        env,
        action_indices=None,
        gripper_control=True,
        deadband=0.05,
        require_button=False,
        expert_linear_scale=1.0,
        expert_angular_scale=1.0,
    ):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False
        self.gripper_control = gripper_control

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices
        self.deadband = float(os.environ.get("HILSERL_SPACEMOUSE_DEADBAND", deadband))
        self.require_button = bool(
            int(os.environ.get("HILSERL_SPACEMOUSE_REQUIRE_BUTTON", int(require_button)))
        )
        self.expert_linear_scale = float(
            os.environ.get("HILSERL_SPACEMOUSE_LINEAR_SCALE", expert_linear_scale)
        )
        self.expert_angular_scale = float(
            os.environ.get("HILSERL_SPACEMOUSE_ANGULAR_SCALE", expert_angular_scale)
        )
        env_debug_twitch = bool(int(os.environ.get("HILSERL_DEBUG_TWITCH", "0")))
        cfg_debug_twitch = None
        try:
            base_env = self.env.unwrapped
            cfg_debug_twitch = getattr(getattr(base_env, "config", None), "DEBUG_TWITCH", None)
        except Exception:
            cfg_debug_twitch = None
        self.debug_twitch = (
            env_debug_twitch if cfg_debug_twitch is None else bool(cfg_debug_twitch)
        )
        
        self.pause_control = False
        self.listener = keyboard.Listener(
            on_press=self.on_press)
        self.listener.start()
        
    def on_press(self, key):
        try:
            if str(key) == 'Key.f6':
                if self.pause_control:
                    self.pause_control = False
                    print('Continue to run!!!')
                else:
                    self.pause_control = True
                    print('Pause!!!')
        except AttributeError:
            pass

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        expert_a = _apply_deadband(expert_a, self.deadband)
        if expert_a.shape[0] >= 3:
            expert_a[:3] *= self.expert_linear_scale
        if expert_a.shape[0] >= 6:
            expert_a[3:6] *= self.expert_angular_scale
        button_pressed = bool(self.left or self.right)
        intervened = bool(np.linalg.norm(expert_a) > 1e-6)

        if self.gripper_enabled:
            if self.gripper_control:
                if self.left:  # close gripper
                    gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                    intervened = True
                elif self.right:  # open gripper
                    gripper_action = np.random.uniform(0.9, 1, size=(1,))
                    intervened = True
                else:
                    gripper_action = np.zeros((1,))
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a
            intervened = bool(np.linalg.norm(expert_a) > 1e-6)

        if self.require_button and not button_pressed:
            intervened = False

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)
        # 左/右键用于成功/失败判定时，本拍直接发零位移，避免结束帧额外抖动一下
        if (not self.gripper_control) and (self.left or self.right):
            new_action = np.zeros_like(new_action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        info["left"] = self.left
        info["right"] = self.right
        
        if not self.gripper_control:
            if self.left:
                rew += 1
                done = True
                if self.debug_twitch:
                    base_env = self.env.unwrapped
                    if hasattr(base_env, "_debug_log"):
                        base_env._debug_log("spacemouse left button -> episode done")
                    if hasattr(base_env, "debug_dump_recent_joint_samples"):
                        base_env.debug_dump_recent_joint_samples(
                            reason="spacemouse_left_done", window=3
                        )
            elif self.right:
                rew = rew
                done = True
                if self.debug_twitch:
                    base_env = self.env.unwrapped
                    if hasattr(base_env, "_debug_log"):
                        base_env._debug_log("spacemouse right button -> episode done")
                    if hasattr(base_env, "debug_dump_recent_joint_samples"):
                        base_env.debug_dump_recent_joint_samples(
                            reason="spacemouse_right_done", window=3
                        )
                
        if "intervene_action" not in info and replaced:
            info["intervene_action"] = new_action
            
        while self.pause_control:
            time.sleep(1.0)
        
        return obs, rew, done, truncated, info
    
class ReplayIntervention(gym.ActionWrapper):
    def __init__(self, env, replay_file_list: list, action_scale=(1, 1, 1)):
        super().__init__(env)

        self.action_scale = action_scale
        assert replay_file_list is not None and len(replay_file_list) > 0
        self.replay_list = []
        self.traj_count = 0
        for path in replay_file_list:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                trajectory = []
                for transition in transitions:
                    temp_pose = transition['infos']['original_state_obs']['tcp_pose']
                    if len(trajectory) == 0 or np.linalg.norm(temp_pose - trajectory[-1]) > 0.001:
                        trajectory.append(temp_pose)
                    if transition['infos']["succeed"]:
                        self.replay_list.append(copy.deepcopy(trajectory))
                        trajectory.clear()
                        self.traj_count += 1
        
        print(f"Get replay trajectory num: {self.traj_count}")              
        
        self.replay_index = random.randint(0, self.traj_count - 1)
        self.step_index = 0
        self.cur_pose = None
        
        self.control_key = False
        self.replay_control = False
        self.listener = keyboard.Listener(
            on_press=self.on_press)
        self.listener.start()
        
    def on_press(self, key):
        try:
            if str(key) == 'Key.f8':
                if not self.control_key:
                    self.control_key = True
                    print('Change to REPLAY in next run!!!')
            elif str(key) == 'Key.f9':
                if self.control_key:
                    self.control_key = False
                    print('Change to CONTROL in next run!!!')
        except AttributeError:
            pass

    def step(self, action):
        if self.replay_control:
            assert self.cur_pose is not None
            replay_action = action.copy()
            replay_action[:6] = (self.replay_list[self.replay_index][self.step_index] - self.cur_pose).copy()
            if self.action_scale[0] == 0:
                replay_action[0:3] = np.zeros_like(replay_action[0:3])
            else:
                replay_action[0:3] = replay_action[0:3] / self.action_scale[0]
            
            if self.action_scale[1] == 0:
                replay_action[3:6] = np.zeros_like(replay_action[3:6])
            else:
                replay_action[3:6] = replay_action[3:6] / self.action_scale[1]
            
            if self.action_scale[2] == 0:
                replay_action[6] = np.zeros_like(replay_action[6])
            else:
                replay_action[6] = replay_action[6] / self.action_scale[2]
            replaced = True
        else:
            replay_action = action
            replaced = False

        obs, rew, done, truncated, info = self.env.step(replay_action)
        self.cur_pose = obs['state']['tcp_pose']
        if replaced:
            info["intervene_action"] = replay_action
            self.step_index += 1
            if self.step_index >= len(self.replay_list[self.replay_index]):
                rew += 1
                done = True
            else:
                rew = rew
                done = False
        
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        if self.control_key:
            self.replay_control = True
            self.replay_index = random.randint(0, self.traj_count - 1)
            self.step_index = 0
            self.cur_pose = self.replay_list[self.replay_index][0]
            return self.env.reset(replay_start_pose=self.cur_pose, **kwargs)
        else:
            self.replay_control = False
            return self.env.reset(**kwargs)

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info
