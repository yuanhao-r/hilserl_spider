from xarm.wrapper import XArmAPI
import time
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import numpy as np
from scipy.spatial.transform import Rotation
import copy

ip = "192.168.1.232"

arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
time.sleep(1)

# arm.move_gohome(wait=True)
target_pos = [682.394348, -144.569656, 60.516685, 3.085307, 0.039522, 1.144267]
# arm.set_position(x=target_pos[0], y=target_pos[1], z=target_pos[2], roll=target_pos[3], pitch=target_pos[4], yaw=target_pos[5], speed=60, wait=True, is_radian=True)

print(arm.get_position(is_radian=True))
print(arm.get_joint_states(is_radian=False))

code, angles = arm.get_servo_angle(is_radian=True)

if code == 0:
    print(f"所有关节角度（°）：{angles}")  # 输出示例：[0.0, -90.0, 90.0, 0.0, 0.0, 0.0]
else:
    print(f"获取失败，状态码：{code}")
