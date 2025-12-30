from Robotic_Arm.rm_robot_interface import *
import time

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
print(handle.id)

ret, state = arm.rm_get_current_arm_state()
print(state)

joint = state['joint']

# while True:
#     joint[6] -= 1.0
#     arm.rm_movej_canfd(joint, True, 0, 0, 50)
#     time.sleep(0.05)

# arm.rm_movel([-0.2746,0.0028,0.305, 3.135, -0.074, 0.065], 20, 0, 0, 1)

arm.rm_delete_robot_arm()