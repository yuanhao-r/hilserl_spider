from xarm.wrapper import XArmAPI
import time
import argparse

parser = argparse.ArgumentParser(description='夹爪控制')
parser.add_argument('--cmd', type=int, default=1, help='True: 关闭夹爪, False: 打开夹爪')
args = parser.parse_args()

ip = "192.168.1.198"

arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
time.sleep(1)

arm.set_tgpio_modbus_baudrate(baud=115200)
if args.cmd:
    code, ret = arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
    print('set_close_gripper_location, code={}, ret={}'.format(code, ret))
else:
    code, ret = arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
    print('set_open_gripper_location, code={}, ret={}'.format(code, ret))

code, ret = arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
print('set_gripper_move, code={}, ret={}'.format(code, ret))

arm.disconnect()