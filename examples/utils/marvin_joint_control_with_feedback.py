#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关节角度控制示例：设置关节角度控制机械臂运动，并实时输出末端笛卡尔位姿

功能：
1. 连接机器人
2. 可选：设置目标关节角度并控制机械臂运动
3. 实时输出当前关节角度和末端笛卡尔位姿

使用前请修改：
1. ROBOT_IP: 机器人IP地址
2. CONFIG_FILE: 根据实际机型选择配置文件
3. TARGET_JOINTS: 目标关节角度（根据实际情况设置）
4. MONITOR_ONLY: True 时仅监控，不发送控制指令
"""

import sys
import os
import time
import logging
import threading

# 1. 获取当前文件所在目录: .../hilserl_spider/examples/utils
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 向上跳三级，到达共同根目录 /home/ubuntu/teleop_tianji
# 第一级回退到 examples, 第二级回退到 hilserl_spider, 第三级回退到 teleop_tianji
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))

# 3. 拼接出 SDK 所在的 demo 目录: /home/ubuntu/teleop_tianji/test_teleop/demo
sdk_parent_path = os.path.join(root_dir, "test_teleop", "demo")

# 4. 将该路径加入搜索路径
if sdk_parent_path not in sys.path:
    sys.path.insert(0, sdk_parent_path)

# 引入天机机器人SDK
try:
    from SDK_PYTHON.fx_robot import Marvin_Robot
    # print(f"成功加载 SDK，路径来自于: {sdk_parent_path}")
except ImportError:
    print(f"Error: 找不到 SDK_PYTHON。")
    print(f"请检查路径是否存在: {sdk_parent_path}")
    print(f"当前 sys.path 为: {sys.path}")
    sys.exit(1)

from SDK_PYTHON.fx_kine import Marvin_Kine
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS

# 配置日志
logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================
ROBOT_IP = '10.10.13.10'  # 请修改为实际IP
ARM_NAME = 'B'  # 'A' 左臂 或 'B' 右臂
ARM_TYPE = 0 if ARM_NAME == 'A' else 1  # 0: 左臂, 1: 右臂

# 根据实际机型选择配置文件
CONFIG_FILE = os.path.join(root_dir, 'DEMO_PYTHON', 'ccs_m6_40.MvKDCfg')

# 目标关节角度 [J1, J2, J3, J4, J5, J6, J7] (单位：度)
# 请根据实际情况修改目标关节角度
# TARGET_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# TARGET_JOINTS = [91.45, 39.04, -7.91, 56.91, -83.32, 4.32, 85.93]
TARGET_JOINTS = [-5.949056,-66.850751,-82.187302,-59.176090,66.628048,-10.041069,21.245578]


VEL_RATIO = 10  # 速度百分比（安全起见，调试时使用较小值）
ACC_RATIO = 10  # 加速度百分比

# 只读监控模式：True 时不发送 set_state/set_joint_cmd_pose/send_cmd 控制指令
MONITOR_ONLY = True

# 实时反馈参数
FEEDBACK_INTERVAL = 0.1  # 反馈间隔（秒），0.1秒 = 10Hz
FEEDBACK_ENABLED = True  # 是否启用实时反馈

# ==================== 实时反馈线程 ====================

class PoseFeedbackThread:
    """实时反馈线程：持续输出当前关节角度和笛卡尔位姿"""
    
    def __init__(self, robot, kine, dcss, arm_name, interval=0.1):
        self.robot = robot
        self.kine = kine
        self.dcss = dcss
        self.arm_name = arm_name
        self.arm_idx = 0 if arm_name == 'A' else 1
        self.interval = interval
        self.running = False
        self.thread = None
        
    def start(self):
        """启动反馈线程"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self.thread.start()
        logger.info(f"实时反馈线程已启动 (频率: {1.0/self.interval:.1f}Hz)")
        
    def stop(self):
        """停止反馈线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("实时反馈线程已停止")
        
    def _feedback_loop(self):
        """反馈循环"""
        while self.running:
            try:
                # 获取当前关节角度
                sub_data = self.robot.subscribe(self.dcss)
                current_joints = sub_data['outputs'][self.arm_idx]['fb_joint_pos']
                
                # 计算当前笛卡尔位姿
                current_pose_mat = self.kine.fk(joints=current_joints)
                if current_pose_mat:
                    current_xyzabc = self.kine.mat4x4_to_xyzabc(pose_mat=current_pose_mat)
                    
                    # 输出信息（使用 \r 实现同一行更新）
                    sys.stdout.write(f"\r[实时反馈] "
                                   f"关节: [{current_joints[0]:7.2f}, {current_joints[1]:7.2f}, "
                                   f"{current_joints[2]:7.2f}, {current_joints[3]:7.2f}, "
                                   f"{current_joints[4]:7.2f}, {current_joints[5]:7.2f}, "
                                   f"{current_joints[6]:7.2f}]° | "
                                   f"位姿: X={current_xyzabc[0]:7.2f}, Y={current_xyzabc[1]:7.2f}, "
                                   f"Z={current_xyzabc[2]:7.2f}, A={current_xyzabc[3]:7.2f}°, "
                                   f"B={current_xyzabc[4]:7.2f}°, C={current_xyzabc[5]:7.2f}°  "
                                   f"位姿：{current_xyzabc[0]:7.2f},{current_xyzabc[1]:7.2f},{current_xyzabc[2]:7.2f},{current_xyzabc[3]:7.2f},{current_xyzabc[4]:7.2f},{current_xyzabc[5]:7.2f}   ")
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\r[实时反馈] 正运动学计算失败")
                    sys.stdout.flush()
                    
            except Exception as e:
                sys.stdout.write(f"\r[实时反馈] 错误: {e}")
                sys.stdout.flush()
                
            time.sleep(self.interval)

# ==================== 主程序 ====================

def main():
    """主函数"""

    # 1. 初始化
    logger.info("=" * 80)
    if MONITOR_ONLY:
        logger.info("关节角度状态监控示例 - 只读模式（不运动）")
    else:
        logger.info("关节角度控制示例 - 实时反馈版本")
    logger.info("=" * 80)

    dcss = DCSS()
    robot = Marvin_Robot()
    kine = Marvin_Kine()
    kine.log_switch(0)  # 关闭运动学日志

    feedback_thread = None
    arm_idx = 0 if ARM_NAME == 'A' else 1
    connected = False
    control_mode_set = False
    interrupted = False

    try:
        # 2. 连接机器人
        logger.info("连接机器人...")
        init = robot.connect(ROBOT_IP)
        if init == 0:
            logger.error('连接失败！')
            return
        connected = True

        # 清错（只在控制模式下执行）
        if not MONITOR_ONLY:
            time.sleep(0.5)
            robot.clear_set()
            robot.clear_error('A')
            robot.clear_error('B')
            robot.send_cmd()
            time.sleep(0.5)
        else:
            logger.info("只读模式：跳过清错和控制命令下发")

        # 验证连接
        for i in range(5):
            sub_data = robot.subscribe(dcss)
            if sub_data['outputs'][0]['frame_serial'] != 0:
                logger.info("✓ 连接成功")
                break
            time.sleep(0.1)
        else:
            logger.error("✗ 连接失败")
            return

        # 3. 加载运动学配置
        logger.info("加载运动学配置...")
        if not os.path.exists(CONFIG_FILE):
            logger.error(f'配置文件不存在: {CONFIG_FILE}')
            return

        ini_result = kine.load_config(arm_type=ARM_TYPE, config_path=CONFIG_FILE)
        if not ini_result:
            logger.error('加载配置文件失败！')
            return

        # 初始化运动学
        kine.initial_kine(
            robot_type=ini_result['TYPE'][0],
            dh=ini_result['DH'][0],
            pnva=ini_result['PNVA'][0],
            j67=ini_result['BD'][0]
        )
        logger.info("✓ 运动学初始化完成")

        # 4. 设置控制模式（只在控制模式下执行）
        if not MONITOR_ONLY:
            logger.info("设置控制模式...")
            robot.clear_set()
            robot.set_state(arm=ARM_NAME, state=1)  # 位置跟随模式
            robot.set_vel_acc(arm=ARM_NAME, velRatio=VEL_RATIO, AccRatio=ACC_RATIO)
            robot.send_cmd()
            time.sleep(0.5)
            control_mode_set = True
        else:
            logger.info("只读模式：跳过控制模式设置")

        # 5. 获取初始状态
        logger.info("=" * 80)
        logger.info("获取初始状态...")
        logger.info("=" * 80)

        sub_data = robot.subscribe(dcss)
        initial_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']

        # 正运动学：关节角度 -> 末端位姿
        initial_pose_mat = kine.fk(joints=initial_joints)
        initial_xyzabc = kine.mat4x4_to_xyzabc(pose_mat=initial_pose_mat)

        logger.info(f"初始关节角度: {initial_joints}")
        logger.info(f"初始末端位姿: X={initial_xyzabc[0]:.2f}, Y={initial_xyzabc[1]:.2f}, "
                    f"Z={initial_xyzabc[2]:.2f}, A={initial_xyzabc[3]:.2f}°, "
                    f"B={initial_xyzabc[4]:.2f}°, C={initial_xyzabc[5]:.2f}°")

        # 6. 启动实时反馈线程
        if FEEDBACK_ENABLED:
            feedback_thread = PoseFeedbackThread(
                robot=robot,
                kine=kine,
                dcss=dcss,
                arm_name=ARM_NAME,
                interval=FEEDBACK_INTERVAL
            )
            feedback_thread.start()
            time.sleep(0.5)  # 等待反馈线程启动

        # 7. 控制机械臂运动到目标关节角度（可选）
        if not MONITOR_ONLY:
            logger.info("=" * 80)
            logger.info("控制机械臂运动到目标关节角度...")
            logger.info("=" * 80)
            logger.info(f"目标关节角度: {TARGET_JOINTS}")

            robot.clear_set()
            robot.set_state(arm=ARM_NAME, state=1)
            robot.set_vel_acc(arm=ARM_NAME, velRatio=VEL_RATIO, AccRatio=ACC_RATIO)
            robot.set_joint_cmd_pose(arm=ARM_NAME, joints=TARGET_JOINTS)
            robot.send_cmd()

            logger.info("等待机械臂运动到位...")
            logger.info("(实时反馈将持续显示当前状态，按 Ctrl+C 可提前停止)")
            time.sleep(5)  # 等待运动完成
        else:
            logger.info("=" * 80)
            logger.info("只读监控模式：不发送运动指令，仅采集当前状态")
            logger.info("=" * 80)
            time.sleep(2)  # 让实时反馈先输出几帧状态

        # 停止实时反馈
        if feedback_thread and feedback_thread.running:
            feedback_thread.stop()
            print()  # 换行，避免覆盖反馈信息

        # 8. 输出当前状态（控制模式下同时验证运动结果）
        logger.info("=" * 80)
        if MONITOR_ONLY:
            logger.info("当前状态快照...")
        else:
            logger.info("验证运动结果...")
        logger.info("=" * 80)

        sub_data = robot.subscribe(dcss)
        actual_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        actual_pose_mat = kine.fk(joints=actual_joints)
        actual_xyzabc = kine.mat4x4_to_xyzabc(pose_mat=actual_pose_mat)

        logger.info(f"当前关节角度: {actual_joints}")
        logger.info(f"当前末端位姿: X={actual_xyzabc[0]:.2f}, Y={actual_xyzabc[1]:.2f}, "
                    f"Z={actual_xyzabc[2]:.2f}, A={actual_xyzabc[3]:.2f}°, "
                    f"B={actual_xyzabc[4]:.2f}°, C={actual_xyzabc[5]:.2f}°")

        if not MONITOR_ONLY:
            # 计算误差
            joint_error = [
                abs(actual_joints[i] - TARGET_JOINTS[i]) for i in range(7)
            ]
            logger.info(f"关节角度误差: {[f'{e:.2f}°' for e in joint_error]}")

        # 9. 可选：持续反馈模式
        if FEEDBACK_ENABLED and feedback_thread:
            logger.info("=" * 80)
            if MONITOR_ONLY:
                logger.info("进入持续只读反馈模式（按 Enter 键停止）...")
            else:
                logger.info("进入持续反馈模式（按 Enter 键停止）...")
            logger.info("=" * 80)

            feedback_thread.start()
            try:
                input()  # 等待用户按 Enter
            except KeyboardInterrupt:
                interrupted = True
                logger.info("\n用户中断")
            finally:
                if feedback_thread.running:
                    feedback_thread.stop()
                    print()  # 换行

    except KeyboardInterrupt:
        interrupted = True
        logger.info("\n用户中断，准备安全退出...")

    finally:
        # 兜底停止实时反馈
        if feedback_thread and feedback_thread.running:
            feedback_thread.stop()
            print()

        # 10. 退出前处理
        if connected:
            if not MONITOR_ONLY and control_mode_set:
                logger.info("=" * 80)
                if interrupted:
                    logger.info("中断退出，下使能...")
                else:
                    logger.info("任务完成，下使能...")
                logger.info("=" * 80)
                try:
                    robot.clear_set()
                    robot.set_state(arm=ARM_NAME, state=0)
                    robot.send_cmd()
                    try:
                        time.sleep(0.2)
                    except KeyboardInterrupt:
                        pass
                except Exception as e:
                    logger.warning(f"下使能失败: {e}")
            elif MONITOR_ONLY:
                logger.info("=" * 80)
                if interrupted:
                    logger.info("只读模式中断退出，未下发任何运动控制指令")
                else:
                    logger.info("只读模式任务完成，未下发任何运动控制指令")
                logger.info("=" * 80)

            try:
                robot.release_robot()
            except Exception as e:
                logger.warning(f"释放机器人连接失败: {e}")

        logger.info("✓ 程序执行完成")

if __name__ == '__main__':
    main()
