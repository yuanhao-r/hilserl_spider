#!/usr/bin/env python3
import argparse
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
teleop_python_path = os.path.join(root_dir, "test_teleop", "python")
if teleop_python_path not in sys.path:
    sys.path.insert(0, teleop_python_path)
serl_robot_infra_path = os.path.join(root_dir, "hilserl_spider", "serl_robot_infra")
if serl_robot_infra_path not in sys.path:
    sys.path.insert(0, serl_robot_infra_path)

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
DEG_TO_RAD = np.pi / 180.0
MM_TO_M = 0.001
DEFAULT_IK_SERVER_URL = "http://127.0.0.1:8765"
DEFAULT_IK_SERVER_TIMEOUT = 1.0
DEFAULT_IK_BACKEND = os.environ.get("HILSERL_KEYBOARD_IK_BACKEND", "server")
DEFAULT_SERVER_ARM_ROLE = os.environ.get(
    "HILSERL_KEYBOARD_SERVER_ARM_ROLE",
    "left" if ARM_NAME == "A" else "right",
)
DEFAULT_TOOL_Z_OFFSET = float(os.environ.get("HILSERL_KEYBOARD_TOOL_Z_OFFSET", "0.2"))
HEAD_Z_OFFSET = float(os.environ.get("HILSERL_KEYBOARD_HEAD_Z_OFFSET", "0.2"))
RIGHT_ARM_BASE_POSE_WXYZXYZ = np.array(
    [0.707105, 0.707108, -0.000005, 0.000005, 0.319484, -0.012501, 1.127505],
    dtype=np.float64,
)
LEFT_ARM_BASE_POSE_WXYZXYZ = np.array(
    [0.707108, -0.707105, -0.000005, -0.000005, 0.319484, 0.012499, 1.127505],
    dtype=np.float64,
)
HEAD_BASE_POSE_WXYZXYZ = np.array(
    [0.354649, -0.045577, 1.239027, -0.313173, 0.3, 0.0, 1.1],
    dtype=np.float64,
)


def wxyzxyz_to_transform(tq):
    qw, qx, qy, qz, x, y, z = tq
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    tf[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return tf


def jointsTorque2Eef(jacobin:list,joints: list, tau: list):
    if len(joints) != 7:
        raise ValueError("joints must be (7,)")
    if len(tau) != 7:
        raise ValueError("tau must be (7,)")
    J = jacobin

    JJt = [[0.0 for _ in range(6)] for _ in range(6)]
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(7):
                s += float(J[i][k]) * float(J[j][k])
            JJt[i][j] = s
    rhs = [0.0 for _ in range(6)]
    for i in range(6):
        s = 0.0
        for k in range(7):
            s += float(J[i][k]) * float(tau[k])
        rhs[i] = s

    n = 6
    M = [JJt[i][:] + [rhs[i]] for i in range(n)]  # 增广矩阵 6x7

    for col in range(n):
        pivot_row = col
        max_abs = abs(M[col][col])
        for r in range(col + 1, n):
            v = abs(M[r][col])
            if v > max_abs:
                max_abs = v
                pivot_row = r

        if max_abs < 1e-12:
            return False

        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= pivot
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if abs(factor) < 1e-18:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    F = [M[i][n] for i in range(n)]
    return F


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
    def __init__(
        self,
        ik_backend=DEFAULT_IK_BACKEND,
        ik_server_url=None,
        ik_server_timeout=DEFAULT_IK_SERVER_TIMEOUT,
        server_arm_role=DEFAULT_SERVER_ARM_ROLE,
        tool_z_offset=DEFAULT_TOOL_Z_OFFSET,
    ):
        self.robot = Marvin_Robot()
        self.kine = Marvin_Kine()
        self.dcss = DCSS()
        self.gripper = GripperHandle(self.robot)
        self.arm_idx = 0 if ARM_NAME == 'A' else 1
        self.connected = False
        self.control_mode_set = False
        self.ik_backend = ik_backend.lower()
        if self.ik_backend not in ("sdk", "server"):
            raise ValueError("ik_backend must be 'sdk' or 'server'")
        self.ik_server_url = (ik_server_url or os.environ.get("HILSERL_TIANJI_IK_URL") or DEFAULT_IK_SERVER_URL).strip()
        self.ik_server_timeout = float(ik_server_timeout)
        self.server_arm_role = server_arm_role.lower()
        if self.server_arm_role not in ("left", "right"):
            raise ValueError("server_arm_role must be 'left' or 'right'")
        self.ik_client = None
        self.server_kines = None
        self.base_left_tf = wxyzxyz_to_transform(LEFT_ARM_BASE_POSE_WXYZXYZ)
        self.base_right_tf = wxyzxyz_to_transform(RIGHT_ARM_BASE_POSE_WXYZXYZ)
        self.head_target_pose = wxyzxyz_to_transform(HEAD_BASE_POSE_WXYZXYZ)
        self.body_waist_q = np.zeros(2, dtype=np.float64)
        self.tool_tf = np.eye(4, dtype=np.float64)
        self.tool_tf[2, 3] = float(tool_z_offset)

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

        if self.ik_backend == "server":
            self._init_server_ik()

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

    def _init_server_ik(self):
        try:
            from franka_env.envs.tianji_ik_client import TianjiIKClient
        except ImportError as exc:
            raise ImportError(
                "无法导入 TianjiIKClient。请确认 hilserl_spider/serl_robot_infra 在 PYTHONPATH 中。"
            ) from exc

        self.ik_client = TianjiIKClient(
            self.ik_server_url,
            timeout=self.ik_server_timeout,
        )
        self.server_kines = []
        for arm_type in (0, 1):
            kine = Marvin_Kine()
            ini_result = kine.load_config(arm_type=arm_type, config_path=CONFIG_FILE)
            if not ini_result:
                raise RuntimeError(f"server IK: 加载 arm_type={arm_type} 运动学配置失败")
            kine.initial_kine(
                robot_type=ini_result['TYPE'][0],
                dh=ini_result['DH'][0],
                pnva=ini_result['PNVA'][0],
                j67=ini_result['BD'][0]
            )
            kine.log_switch(0)
            self.server_kines.append(kine)
        logger.info(f"使用 Tianji IK server: {self.ik_server_url} ({self.server_arm_role} arm)")

    def _update_state(self):
        sub_data = self.robot.subscribe(self.dcss)
        self.latest_sub_data = sub_data
        self.all_current_joints = [
            sub_data['outputs'][0]['fb_joint_pos'],
            sub_data['outputs'][1]['fb_joint_pos'],
        ]
        self.current_joints = sub_data['outputs'][self.arm_idx]['fb_joint_pos']
        pose_mat = self.kine.fk(joints=self.current_joints)
        self.current_xyzabc = self.kine.mat4x4_to_xyzabc(pose_mat=pose_mat)

        current_torque = sub_data['outputs'][self.arm_idx]['est_joint_force']

        jacobin_mat = self.kine.joints2JacobMatrix(joints=self.current_joints)
        if not jacobin_mat:
            return

        eef_torque = jointsTorque2Eef(
            jacobin=jacobin_mat,
            joints=self.current_joints,
            tau=current_torque
        )
        fx, fy, fz, nx, ny, nz = eef_torque
        print(
            f"[Eef torque] "
            f"Fx={fx:8.3f}  Fy={fy:8.3f}  Fz={fz:8.3f}  |  "
            f"Nx={nx:8.3f}  Ny={ny:8.3f}  Nz={nz:8.3f}"
        )

    def _solve_ik_with_sdk(self, target_mat):
        ik_para = FX_InvKineSolvePara()
        mat16 = self.kine.mat4x4_to_mat1x16(target_mat)
        ik_para.set_input_ik_target_tcp(mat16)
        ik_para.set_input_ik_ref_joint(self.current_joints)
        ik_para.set_input_ik_zsp_type(0)
        ik_result = self.kine.ik(structure_data=ik_para)

        if not ik_result or ik_result.m_Output_IsOutRange:
            logger.warning("目标不可达！")
            return None

        return ik_result.m_Output_RetJoint.to_list()

    def _local_fk_m(self, arm_idx, joints_deg):
        if self.server_kines is None:
            raise RuntimeError("server IK kinematics are not initialized")
        pose_mat = np.array(self.server_kines[arm_idx].fk(joints=joints_deg), dtype=np.float64)
        pose_mat[:3, 3] *= MM_TO_M
        return pose_mat

    def _target_mat_mm_to_server_pose(self, target_mat):
        local_target = np.array(target_mat, dtype=np.float64)
        local_target[:3, 3] *= MM_TO_M
        base_tf = self.base_left_tf if self.server_arm_role == "left" else self.base_right_tf
        return base_tf @ local_target @ self.tool_tf

    def _current_server_arm_poses(self):
        left_local = self._local_fk_m(0, self.all_current_joints[0])
        right_local = self._local_fk_m(1, self.all_current_joints[1])
        left_pose = self.base_left_tf @ left_local @ self.tool_tf
        right_pose = self.base_right_tf @ right_local @ self.tool_tf
        return left_pose, right_pose

    def _solve_ik_with_server(self, target_mat):
        if self.ik_client is None:
            raise RuntimeError("IK backend is server but IK client is not initialized")

        current_ql_rad = np.array(self.all_current_joints[0], dtype=np.float64) * DEG_TO_RAD
        current_qr_rad = np.array(self.all_current_joints[1], dtype=np.float64) * DEG_TO_RAD
        left_target_pose, right_target_pose = self._current_server_arm_poses()
        active_target_pose = self._target_mat_mm_to_server_pose(target_mat)
        if self.server_arm_role == "left":
            left_target_pose = active_target_pose
        else:
            right_target_pose = active_target_pose

        try:
            result = self.ik_client.solve(
                current_ql_rad=current_ql_rad,
                current_qr_rad=current_qr_rad,
                left_target_pose=left_target_pose,
                right_target_pose=right_target_pose,
                head_target_pose=self.head_target_pose,
                body_waist_q=self.body_waist_q,
                head_z_offset=HEAD_Z_OFFSET,
                force_seed_from_current=False,
            )
        except Exception as exc:
            logger.warning(f"IK server 求解失败: {exc}")
            return None

        key = "ql_target_rad" if self.server_arm_role == "left" else "qr_target_rad"
        return (np.array(result[key], dtype=np.float64) / DEG_TO_RAD).tolist()

    def _solve_target_joints(self, target_mat):
        if self.ik_backend == "server":
            return self._solve_ik_with_server(target_mat)
        return self._solve_ik_with_sdk(target_mat)

    def move_cartesian(self, delta):
        self._update_state()
        target = [self._target[i] + delta[i] if i < 3 else self._target[i] for i in range(6)]

        print('---')
        target_mat = np.array(self.kine.xyzabc_to_mat4x4(xyzabc=target))
        target_mat[:3, :3] = R.from_euler('xyz', delta[3:], degrees=True).as_matrix() @ target_mat[:3, :3]
        target_mat = target_mat.tolist()

        self._target = self.kine.mat4x4_to_xyzabc(target_mat)

        target_joints = self._solve_target_joints(target_mat)
        if target_joints is None:
            logger.warning("目标不可达！")
            return False
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
    parser = argparse.ArgumentParser(description="Keyboard control for Tianji/Marvin arm.")
    parser.add_argument(
        "--ik-backend",
        choices=("sdk", "server"),
        default=DEFAULT_IK_BACKEND,
        help="IK solver backend. Default keeps the SDK local IK path.",
    )
    parser.add_argument(
        "--ik-server-url",
        default=os.environ.get("HILSERL_TIANJI_IK_URL", DEFAULT_IK_SERVER_URL),
        help="Tianji IK server URL used when --ik-backend server.",
    )
    parser.add_argument(
        "--ik-server-timeout",
        type=float,
        default=float(os.environ.get("HILSERL_TIANJI_IK_TIMEOUT", DEFAULT_IK_SERVER_TIMEOUT)),
        help="HTTP timeout for Tianji IK server requests.",
    )
    parser.add_argument(
        "--server-arm-role",
        choices=("left", "right"),
        default=DEFAULT_SERVER_ARM_ROLE,
        help="Which full-body IK arm corresponds to ARM_NAME in this script.",
    )
    parser.add_argument(
        "--tool-z-offset",
        type=float,
        default=DEFAULT_TOOL_Z_OFFSET,
        help="Tool offset in meters appended to the arm flange pose for server IK.",
    )
    args = parser.parse_args()

    ctrl = KeyboardController(
        ik_backend=args.ik_backend,
        ik_server_url=args.ik_server_url,
        ik_server_timeout=args.ik_server_timeout,
        server_arm_role=args.server_arm_role,
        tool_z_offset=args.tool_z_offset,
    )
    if not ctrl.connect():
        return
    try:
        ctrl.run()
    finally:
        ctrl.shutdown()


if __name__ == '__main__':
    main()
