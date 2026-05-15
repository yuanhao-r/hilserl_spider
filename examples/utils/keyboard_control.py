#!/usr/bin/env python3
import sys
import os
import time
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sdk_parent_path = os.path.join(root_dir, "test_teleop", "demo")
if sdk_parent_path not in sys.path:
    sys.path.insert(0, sdk_parent_path)

try:
    from SDK_PYTHON.fx_robot import Marvin_Robot
except ImportError:
    print("Error: 找不到 SDK_PYTHON。")
    sys.exit(1)

from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara
from SDK_PYTHON.fx_robot import DCSS

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

ROBOT_IP = '10.10.13.10'
ARM_NAME = 'B'
ARM_TYPE = 0 if ARM_NAME == 'A' else 1
CONFIG_FILE = os.path.join(root_dir, 'DEMO_PYTHON', 'ccs_m6_40.MvKDCfg')
VEL_RATIO = 10
ACC_RATIO = 10
STEP_LINEAR = 1.0
STEP_ANGULAR = 1.0

# Gripper defaults
GRIPPER_COM_PORT = 2
GRIPPER_SLAVE_ID = 9


def getch():
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class GripperHandle:
    def __init__(self, robot):
        self.robot = robot
        self.is_activated = True

    def _calc_crc16(self, data: bytearray) -> int:
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for i in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc

    def _send_modbus_cmd(self, payload: list):
        data_bytes = bytearray(payload)
        crc = self._calc_crc16(data_bytes)
        data_bytes.append(crc & 0xFF)
        data_bytes.append((crc >> 8) & 0xFF)
        hex_str = " ".join([f"{b:02X}" for b in data_bytes])
        byte_len = len(data_bytes)
        self.robot.set_485_data(ARM_NAME, hex_str, byte_len, GRIPPER_COM_PORT)
        time.sleep(0.05)

    def activate(self):
        logger.info("激活夹爪...")
        cmd_reset = [GRIPPER_SLAVE_ID, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x00, 0x00]
        self._send_modbus_cmd(cmd_reset)
        time.sleep(0.5)
        cmd_enable = [GRIPPER_SLAVE_ID, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x00, 0x01]
        self._send_modbus_cmd(cmd_enable)
        self.is_activated = True
        time.sleep(2.0)

    def move(self, position, speed=255, force=255):
        pos = int(position) & 0xFF
        spd = int(speed) & 0xFF
        frc = int(force) & 0xFF
        payload = [
            GRIPPER_SLAVE_ID, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
            0x00, 0x09, pos, 0x00, frc, spd
        ]
        self._send_modbus_cmd(payload)

    def open(self):
        if not self.is_activated:
            self.activate()
        self.move(position=0)

    def close(self):
        if not self.is_activated:
            self.activate()
        self.move(position=255)


class KeyboardController:
    def __init__(self):
        self.robot = Marvin_Robot()
        self.kine = Marvin_Kine()
        self.dcss = DCSS()
        self.gripper = GripperHandle(self.robot)
        self.arm_idx = 0 if ARM_NAME == 'A' else 1
        self.connected = False
        self.control_mode_set = False

    def connect(self):
        logger.info("连接机器人...")
        if self.robot.connect(ROBOT_IP) == 0:
            logger.error("连接失败！")
            return False

        self.robot.log_switch('0')
        self.robot.local_log_switch('0')

        time.sleep(0.2)
        self.robot.clear_set()
        self.robot.clear_error('A')
        self.robot.clear_error('B')
        self.robot.send_cmd()
        time.sleep(0.2)


        for i in range(5):
            sub_data = self.robot.subscribe(self.dcss)
            if sub_data['outputs'][0]['frame_serial'] != 0:
                break
            time.sleep(0.1)
        else:
            logger.error("连接验证失败")
            return False

        logger.info("加载运动学配置...")
        if not os.path.exists(CONFIG_FILE):
            logger.error(f'配置文件不存在: {CONFIG_FILE}')
            return False

        ini_result = self.kine.load_config(arm_type=ARM_TYPE, config_path=CONFIG_FILE)
        if not ini_result:
            logger.error('加载配置文件失败！')
            return False

        self.kine.initial_kine(
            robot_type=ini_result['TYPE'][0],
            dh=ini_result['DH'][0],
            pnva=ini_result['PNVA'][0],
            j67=ini_result['BD'][0]
        )
        self.kine.log_switch(0)

        logger.info("设置控制模式...")
        self.robot.clear_set()
        self.robot.set_state(arm=ARM_NAME, state=1)
        self.robot.set_vel_acc(arm=ARM_NAME, velRatio=VEL_RATIO, AccRatio=ACC_RATIO)
        self.robot.send_cmd()
        self.control_mode_set = True
        time.sleep(0.2)

        self.robot.clear_set()
        self.robot.set_state(arm=ARM_NAME, state=3)  # 扭矩模式
        self.robot.set_joint_kd_params(
            arm=ARM_NAME,
            K=[15, 15, 15, 15, 15, 15, 15],
            D=[0.84, 0.84, 0.95, 0.68, 0.58, 0.58, 0.52],
        )
        self.robot.set_cart_kd_params(
            arm=ARM_NAME,
            K=[2800, 2800, 4300, 78, 78, 78, 32],
            D=[0.62, 0.62, 0.78, 0.92, 0.92, 0.92, 2.2],
            type=2,
        )
        self.robot.send_cmd()
        time.sleep(0.2)

        self.robot.clear_set()
        self.robot.set_impedance_type(arm=ARM_NAME, type=2)  # 1关节阻抗;2笛卡尔阻抗;3力控
        self.robot.send_cmd()
        time.sleep(0.1)

        # Clear 485 cache for gripper
        self.robot.clear_485_cache(ARM_NAME)

        self.connected = True
        self._update_state()
        logger.info("✓ 初始化完成")

        self._update_state()
        self._target_joints = np.copy(self.current_joints).tolist()
        self._target = self.current_xyzabc
        return True

    def _update_state(self):
        sub_data = self.robot.subscribe(self.dcss)
        self.current_joints = sub_data['outputs'][self.arm_idx]['fb_joint_pos']
        pose_mat = self.kine.fk(joints=self.current_joints)
        self.current_xyzabc = self.kine.mat4x4_to_xyzabc(pose_mat=pose_mat)

    def move_cartesian(self, delta):
        self._update_state()
        target = [self._target[i] + delta[i] if i < 3 else self._target[i] for i in range(6)]

        print('---')
        target_mat = np.array(self.kine.xyzabc_to_mat4x4(xyzabc=target))
        target_mat[:3, :3] = R.from_euler('xyz', delta[3:], degrees=True).as_matrix() @ target_mat[:3, :3]
        target_mat = target_mat.tolist()

        self._target = self.kine.mat4x4_to_xyzabc(target_mat)

        ik_para = FX_InvKineSolvePara()
        mat16 = self.kine.mat4x4_to_mat1x16(target_mat)
        ik_para.set_input_ik_target_tcp(mat16)
        ik_para.set_input_ik_ref_joint(self.current_joints)
        ik_para.set_input_ik_zsp_type(0)
        ik_result = self.kine.ik(structure_data=ik_para)

        if not ik_result or ik_result.m_Output_IsOutRange:
            logger.warning("目标不可达！")
            return False

        target_joints = ik_result.m_Output_RetJoint.to_list()
        # target_joints = (self._target_joints + (np.array(target_joints) - self.current_joints)).tolist()

        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm=ARM_NAME, joints=target_joints)
        self.robot.send_cmd()

        self._target_joints = target_joints
        return True

    def print_joints(self):
        self._update_state()
        print(f"\r关节角度: " + ",  ".join([f"{v:3.5f}" for i, v in enumerate(self.current_joints)]))
        print(f"\r关节角度: " + ",  ".join([f"{v:3.5f}" for i, v in enumerate(self._target_joints)]))

    def print_pose(self):
        self._update_state()
        print(f"\r末端位姿: X={self.current_xyzabc[0]:7.2f}  Y={self.current_xyzabc[1]:7.2f}  Z={self.current_xyzabc[2]:7.2f}  "
              f"A={self.current_xyzabc[3]:7.2f}  B={self.current_xyzabc[4]:7.2f}  C={self.current_xyzabc[5]:7.2f}")

    def run(self):
        print("=" * 60)
        print("  键盘控制 Marvin 机械臂")
        print("=" * 60)
        print("")
        print("  移动 (每按 {}mm):      旋转 (每按 {}°):      夹爪:".format(STEP_LINEAR, STEP_ANGULAR))
        print("    w: +X (前进)            u: +A                  g: 打开")
        print("    s: -X (后退)            j: -A                  h: 闭合")
        print("    a: +Y (左移)            i: +B")
        print("    d: -Y (右移)            k: -B")
        print("    q: +Z (上升)            o: +C")
        print("    e: -Z (下降)            l: -C")
        print("")
        print("  p: 打印关节角度    r: 打印末端位姿    ?: 帮助    x/ESC: 退出")
        print("  SHIFT + 方向键 = 5x步长")
        print("-" * 60)

        key_map = {
            'w': ( STEP_LINEAR, 0, 0, 0, 0, 0),
            's': (-STEP_LINEAR, 0, 0, 0, 0, 0),
            'e': (0,  STEP_LINEAR, 0, 0, 0, 0),
            'q': (0, -STEP_LINEAR, 0, 0, 0, 0),
            'd': (0, 0,  STEP_LINEAR, 0, 0, 0),
            'a': (0, 0, -STEP_LINEAR, 0, 0, 0),
            'u': (0, 0, 0,  STEP_ANGULAR, 0, 0),
            'j': (0, 0, 0, -STEP_ANGULAR, 0, 0),
            'i': (0, 0, 0, 0,  STEP_ANGULAR, 0),
            'k': (0, 0, 0, 0, -STEP_ANGULAR, 0),
            'o': (0, 0, 0, 0, 0,  STEP_ANGULAR),
            'l': (0, 0, 0, 0, 0, -STEP_ANGULAR),
        }

        try:
            while True:
                ch = getch()
                if ch in ('x', '\x1b'):
                    print("\n退出")
                    break
                elif ch == '?':
                    continue
                elif ch == 'p':
                    self.print_joints()
                elif ch == 'r':
                    self.print_pose()
                elif ch == 'g':
                    print("\n打开夹爪...")
                    self.gripper.open()
                    print("夹爪已打开")
                elif ch == 'h':
                    print("\n闭合夹爪...")
                    self.gripper.close()
                    print("夹爪已闭合")
                elif ch in key_map:
                    self.move_cartesian(key_map[ch])
                elif ch.upper() in key_map:
                    delta = key_map[ch.upper()]
                    delta = tuple(5 * d for d in delta)
                    self.move_cartesian(delta)
        except KeyboardInterrupt:
            print("\n中断")

    def shutdown(self):
        if self.connected and self.control_mode_set:
            try:
                self.robot.clear_set()
                self.robot.set_state(arm=ARM_NAME, state=0)
                self.robot.send_cmd()
            except Exception:
                pass
        if self.connected:
            try:
                self.robot.release_robot()
            except Exception:
                pass
        print("已安全关闭")


def main():
    ctrl = KeyboardController()
    if not ctrl.connect():
        return
    try:
        ctrl.run()
    finally:
        ctrl.shutdown()


if __name__ == '__main__':
    main()
