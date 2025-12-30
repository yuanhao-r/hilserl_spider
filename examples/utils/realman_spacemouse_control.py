import time
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import numpy as np
from Robotic_Arm.rm_robot_interface import *
from scipy.spatial.transform import Rotation
import json

expert = SpaceMouseExpert()
action_scale = (0.006, 0.036, 1)

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm('192.168.124.18', 8080)

record_list = []

run_time = 10000
while run_time > 0:
    run_time -= 1
    expert_a, buttons = expert.get_action()
    
    if np.linalg.norm(expert_a) < 0.001:
        expert_a = np.zeros_like(expert_a)
    
    ret, state = arm.rm_get_current_arm_state()
    pos = np.array(state["pose"])
    joint = np.array(state['joint'])
    xyz_delta = expert_a[:3] * action_scale[0]
    target_pose = pos.copy()
    target_pose[0] += xyz_delta[0]
    target_pose[1] += xyz_delta[1]
    target_pose[2] += xyz_delta[2]
    target_euler = (Rotation.from_euler("xyz", expert_a[3:6] * action_scale[1]) * Rotation.from_euler("xyz", pos[3:])).as_euler("xyz")
    target_pose[3:] = target_euler
    
    if np.all(np.abs(pos - target_pose.tolist()) < 0.001):
        time.sleep(0.05)
        continue
    
    MAX_STEP = 0.08
    
    
    
    target_pose = rm_inverse_kinematics_params_t(joint.tolist(), target_pose.tolist(), 1)
    ret, target_joint = arm.rm_algo_inverse_kinematics(target_pose)
    arm.rm_movej_follow(target_joint)
    continue
    
    if ret == 0:
        differences = [abs(target_joint[j] - joint.tolist()[j]) for j in range(len(target_joint))]
        required_steps = []
        for diff in differences:
            if diff == 0:
                required_steps.append(1)
            else:
                steps = int(diff / MAX_STEP) + (1 if diff % MAX_STEP > 0 else 0)
                required_steps.append(steps)
        total_steps = max(required_steps)
        # 确保至少有2步（包含A和B）
        total_steps = max(total_steps, 2)
        
        sequence = []
        for i in range(total_steps):
            t = i / (total_steps - 1)
            interpolated = [joint.tolist()[j] + t * (target_joint[j] - joint.tolist()[j]) for j in range(len(target_joint))]
            sequence.append(interpolated)
            
        for joint in sequence:
            arm.rm_movej_canfd(joint, True, 0, 0, 50)
            time.sleep(0.05) 
