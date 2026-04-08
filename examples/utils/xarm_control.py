from xarm.wrapper import XArmAPI
import time
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import numpy as np
from scipy.spatial.transform import Rotation
import copy

ip = "192.168.1.198"

arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
time.sleep(1)

# arm.move_gohome(wait=True)
target_pos = [-402.958649, -576.19873, 199.391266, -3.123522, -0.012916, -2.533455]
# arm.set_position(x=target_pos[0], y=target_pos[1], z=target_pos[2], roll=target_pos[3], pitch=target_pos[4], yaw=target_pos[5], speed=60, wait=True, is_radian=True)

print(arm.get_position(is_radian=True))
print(arm.get_joint_states(is_radian=False))

# # set mode: cartesian online trajectory planning mode
# # the running command will be interrupted when the next command is received
# arm.set_mode(7)
# arm.set_state(0)
# time.sleep(1)

# speed = 60

# expert = SpaceMouseExpert()
# action_scale = (10, 0.06, 1)

# run_time = 1000
# while run_time > 0:
#     run_time -= 1
#     expert_a, buttons = expert.get_action()
    
#     if np.linalg.norm(expert_a) < 0.001:
#         expert_a = np.zeros_like(expert_a)

#     ret, pos = arm.get_position(is_radian=True)
#     xyz_delta = expert_a[:3] * action_scale[0]
#     target_x = pos[0] + xyz_delta[0]
#     target_y = pos[1] + xyz_delta[1]
#     target_z = pos[2] + xyz_delta[2]
    
#     target_euler = (Rotation.from_euler("xyz", expert_a[3:6] * action_scale[1]) * Rotation.from_euler("xyz", pos[3:])).as_euler("xyz")
    
#     arm.set_position(x=target_x, y=target_y, z=target_z, roll=target_euler[0], pitch=target_euler[1], yaw=target_euler[2], speed=speed, wait=False, is_radian=True)
#     time.sleep(0.1)

# # set_mode: position mode
# arm.set_mode(0)
# arm.set_state(0)
# arm.move_gohome(wait=True)
arm.disconnect()