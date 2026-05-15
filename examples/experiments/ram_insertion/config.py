import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    HumanClassifierWrapper,
    ReplayIntervention,
    SleepEnv,
)
from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.ram_insertion.wrapper import RAMEnv

ROBOT_BACKEND = os.environ.get("HILSERL_ARM_BACKEND", "tianji").lower()


if ROBOT_BACKEND in {"tianji", "marvin"}:
    from franka_env.envs.tianji_env import DefaultTianjiEnvConfig

    _WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
    _TELEOP_ROOT = _WORKSPACE_ROOT / "test_teleop"

    class EnvConfig(DefaultTianjiEnvConfig):
        SERVER_URL = "10.10.13.10"
        ROBOT_IP = "10.10.13.10"
        DUAL_IK_URDF_PATH = str(
            (_TELEOP_ROOT / "models/spiderrobot/robot_marvin_arms.urdf").resolve()
        )
        MARVIN_URDF_PATH = str(
            (
                _TELEOP_ROOT
                / "demo/urdf/MarvinCCS/urdf/Marvin M6-S-L-CCS-696-V3.1 urdf.urdf"
            ).resolve()
        )
        REALSENSE_CAMERAS = {
            # "wrist_1": {
            #     "camera_type": "orbbec",
            #     # 多台 Orbbec 时建议固定 serial_number，避免枚举顺序变化。
            #     "serial_number": "CPCV5530014V",
            #     # 也可用 device_index（0/1/2...），但不如 serial 稳定。
            #     # "device_index": 0,
            #     "dim": (1280, 720),
            # },
            # "wrist_1": {
            #     "camera_index": 0,
            #     "dim": (1280, 720),
            # },
            "wrist_2": {
                "camera_index": 0,
                "dim": (1280, 720),
            },
        }
        IMAGE_CROP = {
            # "wrist_1": lambda img: img[5:195, 600:835],
            # "wrist_2": lambda img: img[40:360, 520:840],
            "wrist_2": lambda img: img[278:684, 600:1066]#,[191:656, 295:787],
        }
        WOWSKIN_PORT = None
        # Tianji task poses use [x, y, z, rx, ry, rz], where xyz are in mm here
        # and converted to meters below (same style as xarm).
        # Leave zeros to auto-resolve from current arm pose at env startup.
        TARGET_POSE_MM = np.zeros((6,), dtype=np.float64)
        GRASP_POSE_MM = np.zeros((6,), dtype=np.float64)
        RESET_POSE_MM = np.zeros((6,), dtype=np.float64)

        # 1. 抓取点 / 插入完成点 (使用你测试过的坐标)
        # TARGET_JOINTS = np.array([ 75.05,   49.79,  -88.83, -115.96,  130.50,  -17.64,   -0.84], dtype=np.float64)
        # TARGET_JOINTS = np.array([ 78.61602,  26.68312,  -100.83729,  -117.58612,  149.66961,  -29.96585,  7.68013], dtype=np.float64)
        # 260509
        # TARGET_JOINTS = np.array([94.03467,  45.92141,  -108.42716,  -113.40481,  129.57779,  -29.59668,  -11.24881], dtype=np.float64)
        #TARGET_JOINTS = np.array([85.42277,  56.03439,  -81.79520,  -104.54444,  124.36304,  0.72558,  -11.55420], dtype=np.float64)
        # 260511
        TARGET_JOINTS = np.array([85.74012,  83.62775,  -78.57943,  -114.32326,  102.61940,  11.99046,  -21.32428], dtype=np.float64)
        
        # GRASP_JOINTS = np.array([-8.655314,-70.119028,-80.595401,-48.706882,49.187142,-19.077660,26.062595], dtype=np.float64)

        # 2. 抓取点正上方 (请务必用示教器把机械臂提起到内存槽正上方，并把那时的关节角填到这里！)
        # (这里暂时填的复位点做示范，请一定修改为你实际的正上方安全点)
        # TOP_JOINTS = np.array([ 75.87,   36.76,  -89.20, -121.87,  141.95,  -26.66,   -1.08], dtype=np.float64)
        # 260509:
        # TOP_JOINTS = np.array([98.72443,  40.75554,  -108.88324,  -118.11645,  133.35419,  -33.57391,  -16.68520], dtype=np.float64)
        #TOP_JOINTS = np.array([87.08540,  52.91363,  -80.32155,  -109.01091,  127.76875,  -1.75918,  -17.52225], dtype=np.float64)
        # 260511
        TOP_JOINTS = np.array([86.02954,  76.22053,  -80.41872,  -119.65581,  109.78898,  6.75465,  -25.18097], dtype=np.float64)

        # 3. 初始复位待命点
        # RESET_JOINTS = np.array([ -6.09,   23.71,  -16.85, -103.77,   -5.87,   -8.51,  -24.22], dtype=np.float64)
        # RESET_JOINTS = np.array([ 75.87,   36.76,  -89.20, -121.87,  141.95,  -26.66,   -1.08], dtype=np.float64)
        # 260509:
        # RESET_JOINTS = np.array([98.72443,  40.75554,  -108.88324,  -118.11645,  133.35419,  -33.57391,  -16.68520], dtype=np.float64)
        # RESET_JOINTS = np.array([87.08540,  52.91363,  -80.32155,  -109.01091,  127.76875,  -1.75918,  -17.52225], dtype=np.float64)
        # 260511
        RESET_JOINTS = np.array([86.02954,  76.22053,  -80.41872,  -119.65581,  109.78898,  6.75465,  -25.18097], dtype=np.float64)

        CONTROL_HZ = 10
        CONTROL_TIME = 1 / CONTROL_HZ

        TARGET_POSE = TARGET_POSE_MM.copy()
        GRASP_POSE = GRASP_POSE_MM.copy()
        RESET_POSE = RESET_POSE_MM.copy()
        TARGET_POSE[:3] /= 1000.0
        GRASP_POSE[:3] /= 1000.0
        RESET_POSE[:3] /= 1000.0
        # 安全工作区（会在 step() 和随机化阶段裁剪目标位姿）
        # 注意：若 RESET_JOINTS 正解位姿不在此范围内，系统会把目标“吸”到边界，表现为
        # 1) 看起来随机偏移远大于 RANDOM_* 设定
        # 2) 姿态被夹到边界后出现倾斜
        # 3) SpaceMouse 往回拉时被边界限制
        ABS_POSE_LIMIT_LOW = np.array(
            [0.7505, -0.1617, 0.5230, -np.pi, -np.pi, -np.pi], dtype=np.float64
        )
        ABS_POSE_LIMIT_HIGH = np.array(
            [1.8005, -0.08217, 1.6230, np.pi, np.pi, np.pi], dtype=np.float64
        )
        REWARD_THRESHOLD = 0.001
        # 开关随机范围
        RANDOM_RESET = True
        # 兼容旧参数：若未配置 RANDOM_X_RANGE/Y_RANGE，则沿用 RANDOM_XY_RANGE
        RANDOM_XY_RANGE = 0.03
        # 推荐使用按轴独立随机范围，便于避开“前方柱子”
        RANDOM_X_RANGE = 0.02
        RANDOM_Y_RANGE = 0.01
        RANDOM_Z_RANGE = 0.0
        # 可选：定向限制（单位 m）
        # 例如若“向前”为 +x，可把 RANDOM_DX_MAX 设小一些，减少前探碰撞
        RANDOM_DX_MIN = -0.025
        RANDOM_DX_MAX = 0.025
        RANDOM_DY_MIN = -0.025
        RANDOM_DY_MAX = 0.025
        RANDOM_DZ_MIN = 0.0
        RANDOM_DZ_MAX = 0.0

        # RANDOM for GRASP pose
        RANDOM_TARGET_DX_RANGE = 0.001
        RANDOM_TARGET_DY_RANGE = 0.0
        RANDOM_TARGET_DZ_RANGE = 0.0005

        # Orientation 
        RANDOM_TARGET_RX_RANGE = 0.05
        RANDOM_TARGET_RY_RANGE = 0.05
        RANDOM_TARGET_RZ_RANGE = 0.1

        # 可选：整体偏置（单位 m）
        RANDOM_X_BIAS = 0.0
        RANDOM_Y_BIAS = 0.0
        RANDOM_Z_BIAS = 0.0
        RANDOM_RZ_RANGE = 0.0
        RANDOM_KEEP_TOOL_ORIENTATION = True
        RANDOM_RESET_TIMEOUT = 0.5
        AUTO_QUICK_REGRASP = True
        # RL 探索速度（平移、旋转、夹爪）
        ACTION_SCALE = (0.01, 0.0, 1)
        # SpaceMouse 介入时的额外缩放（会叠加到 ACTION_SCALE 上）
        SPACEMOUSE_LINEAR_SCALE = 1.0
        SPACEMOUSE_ANGULAR_SCALE = 1.0
        # reset 结束后前几帧若尚未检测到 SpaceMouse 介入，则先发零动作，避免交接瞬间掉一下
        SPACEMOUSE_POST_RESET_HOLD_STEPS = 0
        # 调试日志开关（强制关闭 TWITCH 日志）
        DEBUG_TWITCH = False
        DEBUG_TWITCH_RING = 80
        # 打印 SpaceMouse 轴映射（只打印一次）
        DEBUG_SPACEMOUSE_AXIS_MAP = True
        # 夹爪时序：第一轮等待更久，避免未完全张开就下探
        GRIPPER_OPEN_WAIT_SEC = 1.0
        FIRST_ROUND_GRIPPER_OPEN_WAIT_SEC = 3.2
        # 抓取后从 TARGET_JOINTS 提起到 TOP_JOINTS 时，使用笛卡尔直线插值
        LINEAR_LIFT_TARGET_TO_TOP = True
        LINEAR_LIFT_TIMEOUT = 1.0
        # 从 TOP_JOINTS 下降到 TARGET_JOINTS 也使用笛卡尔直线
        LINEAR_DROP_TOP_TO_TARGET = True
        LINEAR_DROP_TIMEOUT = 1.0
        # 首轮通常载荷/状态突变更大，放慢动作避免冲击反弹
        FIRST_ROUND_MOTION_TIMEOUT_SCALE = 2.2
        # reset 轨迹平滑参数（不影响 RL step 主频）
        INTERPOLATE_HZ = 60.0
        INTERPOLATE_MAX_STEP_DEG = 1.2
        INTERPOLATE_MAX_POS_STEP_M = 0.0025
        INTERPOLATE_MAX_ROT_STEP_RAD = 0.025
        INTERPOLATE_EASE = True
        # 关键点到达后继续短暂闭环，抑制“到点松一下”的下垂
        INTERPOLATE_SETTLE_TIMEOUT = 0.2
        INTERPOLATE_SETTLE_TOL_DEG = 0.15
        INTERPOLATE_SETTLE_HZ = 80.0
        # reset/quick_regrasp 阶段：sleep 改为持续 hold 指令
        RESET_HOLD_DURING_SLEEP = True
        RESET_HOLD_HZ = 50.0
        # 连续 reset：中间关键点不收敛停顿，只在最终点短暂收敛
        RESET_CONTINUOUS_MODE = True
        RESET_INTERMEDIATE_SETTLE = False
        RESET_FINAL_SETTLE_TIMEOUT = 0.12
        RESET_HANDOFF_HOLD_SEC = 0.18
        RESET_ASYNC_HANDOFF_HOLD = True
        RESET_HANDOFF_MAX_SEC = 2.2
        RESET_HANDOFF_HZ = 160.0
        RESET_HANDOFF_ZERO_ACTION_EPS = 1e-6
        RESET_HANDOFF_UNLOCK_ACTION_EPS = 0.002
        RESET_HANDOFF_DROP_FIRST_CONTROL_STEP = True
        # 将 TOP->RESET->RANDOM 作为一条连续 waypoint 轨迹下发，减少中间点顿挫/下垂
        RESET_CHAINED_WAYPOINT_MOTION = True
        RESET_CHAINED_TIMEOUT = 1.0
        # 夹爪闭合后从 TARGET 连续衔接到 TOP->RESET->RANDOM，避免 TARGET->TOP 到点停顿
        CHAIN_FROM_TARGET_AFTER_GRASP = True
        RESET_CHAINED_FROM_TARGET_TIMEOUT_SCALE = 1.0
        # quick_regrasp 结束后若已在 TOP 附近，则 reset 流程不再重复经过 TOP
        TOP_REVISIT_THRESHOLD_M = 0.004
        # 宏观复位阶段优先用位置模式，结束后恢复阻抗模式供 RL 控制
        RESET_USE_POSITION_MODE = False
        RESET_POSITION_MODE_VEL_RATIO = 14
        RESET_POSITION_MODE_ACC_RATIO = 14
        RESET_RESTORE_IMPEDANCE_ON_EXIT = False
        RESET_IMPEDANCE_VEL_RATIO = 90
        RESET_IMPEDANCE_ACC_RATIO = 90
        # TOP_JOINTS -> RESET_JOINTS 是否改为笛卡尔直线
        RESET_TOP_TO_RESET_LINEAR = False
        RESET_TOP_TO_RESET_TIMEOUT = 2.0
        # 关键点之间额外停顿（建议保持 0，避免“到点下垂”）
        RESET_REPLAY_ALIGN_DWELL_SEC = 0.0
        RESET_AFTER_RESET_DWELL_SEC = 1.0
        RESET_AFTER_RANDOM_DWELL_SEC = 1.0
        QUICK_REGRASP_DWELL_TOP_SEC = 0.0
        QUICK_REGRASP_DWELL_TARGET_SEC = 0.0
        QUICK_REGRASP_DWELL_FINAL_TOP_SEC = 0.0
        # 右臂(B)抗下坠阻抗增强（负载抓取工况）
        RESET_APPLY_CUSTOM_IMPEDANCE = True
        RESET_IMPEDANCE_TARGET_ARM = "B"
        RESET_IMPEDANCE_JOINT_K = [2.9, 2.9, 3.3, 2.2, 1.65, 1.65, 1.45]
        RESET_IMPEDANCE_JOINT_D = [0.84, 0.84, 0.95, 0.68, 0.58, 0.58, 0.52]
        RESET_IMPEDANCE_CART_K = [2800, 2800, 4300, 78, 78, 78, 32]
        RESET_IMPEDANCE_CART_D = [0.62, 0.62, 0.78, 0.92, 0.92, 0.92, 2.2]
        RESET_IMPEDANCE_CART_D_SECONDARY = [2.2, 2.2, 2.6, 1.8, 1.8, 1.8, 1.8]
        DISPLAY_IMAGE = True
        MAX_EPISODE_LENGTH = 200
        BASIC_JOINT_RESET = np.array(
            [45.0, -60.0, -8.0, -57.0, 5.0, -5.0, 5.0], dtype=np.float64
        )
else:
    from franka_env.envs.xarm_env import DefaultXArmEnvConfig

    class EnvConfig(DefaultXArmEnvConfig):
        SERVER_URL = "192.168.1.232"
        REALSENSE_CAMERAS = {
            "wrist_1": {
                "camera_index": 0,
                "dim": (1280, 720),
            },
            "wrist_2": {
                "camera_index": 2,
                "dim": (1280, 720),
            },
        }
        IMAGE_CROP = {
            "wrist_1": lambda img: img[340:660, 480:800],
            "wrist_2": lambda img: img[40:360, 520:840],
        }
        WOWSKIN_PORT = None
        TARGET_POSE = np.array(
            [552.452942, 13.827323, -83.374962, 3.137447, -0.023693, -0.907792]
        )
        GRASP_POSE = np.array(
            [552.452942, 13.827323, -83.374962, 3.137447, -0.023693, -0.907792]
        )
        RESET_POSE = np.array(
            [552.452942 - 30, 13.827323 + 30, 100 - 83.374962, 3.137447, -0.023693, -0.907792]
        )
        TARGET_POSE[:3] /= 1000.0
        GRASP_POSE[:3] /= 1000.0
        RESET_POSE[:3] /= 1000.0

        ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.03, 0.00, 0.02, 0.02, 0.1]) * 10
        ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.03, 0.01, 0.02, 0.02, 0.1]) * 10
        REWARD_THRESHOLD = 0.001
        RANDOM_RESET = True
        RANDOM_XY_RANGE = 0.03
        RANDOM_RZ_RANGE = 0.0
        AUTO_QUICK_REGRASP = True
        ACTION_SCALE = (0.01, 0.0, 1)
        SPACEMOUSE_LINEAR_SCALE = 3.0
        SPACEMOUSE_ANGULAR_SCALE = 1.0
        DISPLAY_IMAGE = True
        MAX_EPISODE_LENGTH = 1000
        BASIC_JOINT_RESET = np.array(
            [-46.12577010671351, -44.29478763552985, -82.5851081603773, -180.04526184239683, 37.780894280696636, 191.0232473104399]
        )
        COMPLIANCE_PARAM = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
            "translational_Ki": 0,
            "translational_clip_x": 0.0075,
            "translational_clip_y": 0.0016,
            "translational_clip_z": 0.0055,
            "translational_clip_neg_x": 0.002,
            "translational_clip_neg_y": 0.0016,
            "translational_clip_neg_z": 0.005,
            "rotational_clip_x": 0.01,
            "rotational_clip_y": 0.025,
            "rotational_clip_z": 0.005,
            "rotational_clip_neg_x": 0.01,
            "rotational_clip_neg_y": 0.025,
            "rotational_clip_neg_z": 0.005,
            "rotational_Ki": 0,
        }
        PRECISION_PARAM = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 250,
            "rotational_damping": 9,
            "translational_Ki": 0.0,
            "translational_clip_x": 0.1,
            "translational_clip_y": 0.1,
            "translational_clip_z": 0.1,
            "translational_clip_neg_x": 0.1,
            "translational_clip_neg_y": 0.1,
            "translational_clip_neg_z": 0.1,
            "rotational_clip_x": 0.5,
            "rotational_clip_y": 0.5,
            "rotational_clip_z": 0.5,
            "rotational_clip_neg_x": 0.5,
            "rotational_clip_neg_y": 0.5,
            "rotational_clip_neg_z": 0.5,
            "rotational_Ki": 0.0,
        }


class TrainConfig(DefaultTrainingConfig):
    # 相机键与 REALSENSE_CAMERAS 自动同步；只改 REALSENSE_CAMERAS 即可切单/双相机
    image_keys = list(EnvConfig.REALSENSE_CAMERAS.keys())
    # image_keys = ["wrist_2"]
    classifier_keys = list(EnvConfig.REALSENSE_CAMERAS.keys())
    proprio_keys = ["tcp_pose",] # ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 2000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    # resnet_param_fixed: bool = False
    resnet_param_fixed: bool = True
    setup_mode = "single-arm-fixed-gripper"

    
    def get_environment(self, fake_env=False, save_video=False, classifier=False, replay_intervention_files=None):
        env_config = EnvConfig()
        self.image_keys = list(env_config.REALSENSE_CAMERAS.keys())
        self.classifier_keys = list(env_config.REALSENSE_CAMERAS.keys())
        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        # env = SleepEnv(env, control_time=env_config.CONTROL_TIME)
        env = GripperCloseEnv(env)
        if not fake_env:
            if replay_intervention_files is not None:
                env = ReplayIntervention(env, replay_file_list=replay_intervention_files, action_scale=env_config.ACTION_SCALE)
            env = SpacemouseIntervention(
                env,
                gripper_control=False,
                deadband=0.002,
                expert_linear_scale=getattr(env_config, "SPACEMOUSE_LINEAR_SCALE", 1.0),
                expert_angular_scale=getattr(env_config, "SPACEMOUSE_ANGULAR_SCALE", 1.0),
            )
        env = SleepEnv(env, control_time=env_config.CONTROL_TIME)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            # classifier = load_classifier_func(
            #     key=jax.random.PRNGKey(0),
            #     sample=env.observation_space.sample(),
            #     image_keys=self.classifier_keys,
            #     checkpoint_path=os.path.abspath("classifier_ckpt/"),
            # )

            # def reward_func(obs):
            #     sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
            #     # added check for z position to further robustify classifier, but should work without as well
            #     # print(f"Classifier: {sigmoid(classifier(obs))}")
            #     return int(sigmoid(classifier(obs))[0] > 0.85)

            # env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
            
            env = HumanClassifierWrapper(env)
        return env
