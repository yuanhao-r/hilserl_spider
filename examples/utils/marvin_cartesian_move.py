#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单示例：使用笛卡尔位姿控制机械臂移动到指定点

这是一个简化版本，专注于演示核心功能：
1. 定义目标笛卡尔位姿（XYZABC）
2. 使用逆运动学转换为关节角度
3. 控制机械臂运动到目标位姿

使用前请修改：
1. ROBOT_IP: 机器人IP地址
2. CONFIG_FILE: 根据实际机型选择配置文件
3. TARGET_XYZABC: 目标位姿（根据实际情况设置）
"""

import sys
import os
import time
import logging

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

from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara
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

# 目标笛卡尔位姿 [X, Y, Z, A, B, C]
# X, Y, Z: 位置（单位：毫米 mm）
# A, B, C: 欧拉角（单位：度 degree）
# 请根据实际情况修改目标位姿
TARGET_XYZABC = [342.22, -12.96+10+100, 562.81, 112.18,  89.49, 108.80 ]
# TARGET_XYZABC = [0.0, 0.0, 870.49, 0.0, 0.0, 0.0]

VEL_RATIO = 10  # 速度百分比（安全起见，调试时使用较小值）
ACC_RATIO = 10  # 加速度百分比

# 到位判定与等待参数（替代固定 sleep）
MOVE_TIMEOUT_SEC = 8.0
POLL_INTERVAL_SEC = 0.05
POS_TOL_MM = 2.0
ORI_TOL_DEG = 3.0

# ==================== 主程序 ====================

def _angle_error_deg(a, b):
    """计算两个角度的最小差值（单位：度）"""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def _wait_until_target(robot, dcss, kine, arm_idx, target_xyzabc, timeout_sec, poll_sec):
    """
    等待到位：达到阈值即提前返回，不再固定等待5秒
    """
    start = time.time()
    last_joints = None
    last_xyzabc = None

    while time.time() - start < timeout_sec:
        sub_data = robot.subscribe(dcss)
        last_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        pose_mat = kine.fk(joints=last_joints)
        if pose_mat:
            last_xyzabc = kine.mat4x4_to_xyzabc(pose_mat=pose_mat)
            pos_ok = (
                abs(last_xyzabc[0] - target_xyzabc[0]) <= POS_TOL_MM and
                abs(last_xyzabc[1] - target_xyzabc[1]) <= POS_TOL_MM and
                abs(last_xyzabc[2] - target_xyzabc[2]) <= POS_TOL_MM
            )
            ori_ok = (
                _angle_error_deg(last_xyzabc[3], target_xyzabc[3]) <= ORI_TOL_DEG and
                _angle_error_deg(last_xyzabc[4], target_xyzabc[4]) <= ORI_TOL_DEG and
                _angle_error_deg(last_xyzabc[5], target_xyzabc[5]) <= ORI_TOL_DEG
            )
            if pos_ok and ori_ok:
                return True, last_joints, last_xyzabc
        time.sleep(poll_sec)

    return False, last_joints, last_xyzabc

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("初始化...")
    logger.info("=" * 60)

    dcss = DCSS()
    robot = Marvin_Robot()
    kine = Marvin_Kine()
    kine.log_switch(0)  # 关闭运动学日志

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

        # 清错
        time.sleep(0.2)
        robot.clear_set()
        robot.clear_error('A')
        robot.clear_error('B')
        robot.send_cmd()
        time.sleep(0.2)

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

        # 4. 设置控制模式
        logger.info("设置控制模式...")
        robot.clear_set()
        robot.set_state(arm=ARM_NAME, state=1)  # 位置跟随模式
        robot.set_vel_acc(arm=ARM_NAME, velRatio=VEL_RATIO, AccRatio=ACC_RATIO)
        robot.send_cmd()
        control_mode_set = True
        time.sleep(0.2)

        # 5. 获取当前位姿
        logger.info("获取当前位姿...")
        sub_data = robot.subscribe(dcss)
        current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']

        # 正运动学：关节角度 -> 末端位姿
        current_pose_mat = kine.fk(joints=current_joints)
        current_xyzabc = kine.mat4x4_to_xyzabc(pose_mat=current_pose_mat)

        logger.info(f"当前位姿: X={current_xyzabc[0]:.2f}, Y={current_xyzabc[1]:.2f}, "
                    f"Z={current_xyzabc[2]:.2f}, A={current_xyzabc[3]:.2f}, "
                    f"B={current_xyzabc[4]:.2f}, C={current_xyzabc[5]:.2f}")

        # 6. 计算目标关节角度
        logger.info("=" * 60)
        logger.info("计算目标关节角度...")
        logger.info("=" * 60)
        logger.info(f"目标位姿: X={TARGET_XYZABC[0]:.2f}, Y={TARGET_XYZABC[1]:.2f}, "
                    f"Z={TARGET_XYZABC[2]:.2f}, A={TARGET_XYZABC[3]:.2f}, "
                    f"B={TARGET_XYZABC[4]:.2f}, C={TARGET_XYZABC[5]:.2f}")

        # XYZABC -> 4x4位姿矩阵
        target_pose_mat = kine.xyzabc_to_mat4x4(xyzabc=TARGET_XYZABC)

        # 逆运动学：末端位姿 -> 关节角度
        ik_para = FX_InvKineSolvePara()
        mat16 = kine.mat4x4_to_mat1x16(target_pose_mat)
        ik_para.set_input_ik_target_tcp(mat16)
        ik_para.set_input_ik_ref_joint(current_joints)  # 使用当前关节作为参考
        ik_para.set_input_ik_zsp_type(0)

        ik_result = kine.ik(structure_data=ik_para)

        if not ik_result:
            logger.error("逆运动学计算失败！")
            logger.error("可能原因：目标位姿超出机器人可达空间")
            return

        # 检查逆解结果 - 严格检查解的合法性
        if ik_result.m_Output_IsOutRange:
            logger.error("✗ 错误：目标位姿超出机器人可达空间！")
            logger.error("请调整目标位姿，使其在机器人工作空间内")
            return

        if ik_result.m_Output_IsJntExd:
            logger.error("✗ 错误：计算出的关节角度超出限位！")
            logger.error("推荐解的关节角度:")
            ret_joints = ik_result.m_Output_RetJoint.to_list()
            exd_tags = ik_result.m_Output_JntExdTags[:]
            run_lmt_p = ik_result.m_Output_RunLmtP.to_list()
            run_lmt_n = ik_result.m_Output_RunLmtN.to_list()

            for i in range(7):
                joint_val = ret_joints[i]
                if exd_tags[i]:
                    logger.error(f"  关节 {i+1}: {joint_val:.2f}° (超出限位! 限位范围: [{run_lmt_n[i]:.2f}, {run_lmt_p[i]:.2f}])")
                else:
                    logger.info(f"  关节 {i+1}: {joint_val:.2f}° (正常)")

            return

        if any(ik_result.m_Output_IsDeg[:]):
            deg_joints = [i+1 for i, is_deg in enumerate(ik_result.m_Output_IsDeg) if is_deg]
            logger.warning(f"⚠ 警告：关节 {deg_joints} 存在奇异！")
            logger.warning("建议调整目标位姿或参考关节角度以避免奇异")

        target_joints = ik_result.m_Output_RetJoint.to_list()
        logger.info(f"✓ 解有效，目标关节角度: {target_joints}")

        # 7. 控制机械臂运动
        logger.info("=" * 60)
        logger.info("控制机械臂运动到目标位姿...")
        logger.info("=" * 60)

        robot.clear_set()
        robot.set_joint_cmd_pose(arm=ARM_NAME, joints=target_joints)
        robot.send_cmd()

        logger.info("等待机械臂到位（到位即继续，不固定长等待）...")
        reached, actual_joints, actual_xyzabc = _wait_until_target(
            robot=robot,
            dcss=dcss,
            kine=kine,
            arm_idx=arm_idx,
            target_xyzabc=TARGET_XYZABC,
            timeout_sec=MOVE_TIMEOUT_SEC,
            poll_sec=POLL_INTERVAL_SEC,
        )
        if reached:
            logger.info("✓ 已到达目标附近，立即进入收尾")
        else:
            logger.warning("⚠ 到位等待超时，按当前状态收尾")
            if actual_xyzabc is None:
                sub_data = robot.subscribe(dcss)
                actual_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
                actual_pose_mat = kine.fk(joints=actual_joints)
                actual_xyzabc = kine.mat4x4_to_xyzabc(pose_mat=actual_pose_mat)

        # 8. 验证结果
        logger.info("验证运动结果...")
        logger.info(f"实际位姿: X={actual_xyzabc[0]:.2f}, Y={actual_xyzabc[1]:.2f}, "
                    f"Z={actual_xyzabc[2]:.2f}, A={actual_xyzabc[3]:.2f}, "
                    f"B={actual_xyzabc[4]:.2f}, C={actual_xyzabc[5]:.2f}")

        # 计算误差
        pos_error = [
            abs(actual_xyzabc[0] - TARGET_XYZABC[0]),
            abs(actual_xyzabc[1] - TARGET_XYZABC[1]),
            abs(actual_xyzabc[2] - TARGET_XYZABC[2])
        ]
        ori_error = [
            _angle_error_deg(actual_xyzabc[3], TARGET_XYZABC[3]),
            _angle_error_deg(actual_xyzabc[4], TARGET_XYZABC[4]),
            _angle_error_deg(actual_xyzabc[5], TARGET_XYZABC[5])
        ]

        logger.info(f"位置误差: X={pos_error[0]:.2f}mm, Y={pos_error[1]:.2f}mm, Z={pos_error[2]:.2f}mm")
        logger.info(f"姿态误差: A={ori_error[0]:.2f}°, B={ori_error[1]:.2f}°, C={ori_error[2]:.2f}°")

    except KeyboardInterrupt:
        interrupted = True
        logger.info("\n收到 Ctrl+C，正在停止运动并安全退出...")

    finally:
        # 无论正常结束还是中断，都做统一收尾
        if connected and control_mode_set:
            logger.info("=" * 60)
            if interrupted:
                logger.info("中断收尾：下使能...")
            else:
                logger.info("任务完成：下使能...")
            logger.info("=" * 60)
            try:
                robot.clear_set()
                robot.set_state(arm=ARM_NAME, state=0)
                robot.send_cmd()
            except Exception as e:
                logger.warning(f"下使能失败: {e}")

        if connected:
            try:
                robot.release_robot()
            except Exception as e:
                logger.warning(f"释放连接失败: {e}")

        logger.info("✓ 程序执行完成")

if __name__ == '__main__':
    main()
