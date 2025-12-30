from Robotic_Arm.rm_robot_interface import *
import argparse
import time

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
print(handle.id)

# 配置控制器RS485端口为RTU主站
print(arm.rm_set_modbus_mode(1,115200,2))

parser = argparse.ArgumentParser(description='夹爪控制')
parser.add_argument('--cmd', type=int, default=1, help='True: 关闭夹爪, False: 打开夹爪')
args = parser.parse_args()

cmd = args.cmd
time.sleep(5)

params1 = rm_peripheral_read_write_params_t(port=1, device=1, address=258)
params2 = rm_peripheral_read_write_params_t(port=1, device=1, address=259)
params3 = rm_peripheral_read_write_params_t(port=1, device=1, address=264)
data1 = int(0)
if cmd:
    data2 = int(12000)
else:
    data2 = int(0)
data3 = int(1)

arm.rm_write_single_register(params1, data1)
arm.rm_write_single_register(params2, data2)
arm.rm_write_single_register(params3, data3)

arm.rm_delete_robot_arm()