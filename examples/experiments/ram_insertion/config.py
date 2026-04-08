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
        # Tianji task poses use [x, y, z, rx, ry, rz], where xyz are in mm here
        # and converted to meters below (same style as xarm).
        # Leave zeros to auto-resolve from current arm pose at env startup.
        TARGET_POSE_MM = np.zeros((6,), dtype=np.float64)
        GRASP_POSE_MM = np.zeros((6,), dtype=np.float64)
        RESET_POSE_MM = np.zeros((6,), dtype=np.float64)

        TARGET_POSE = TARGET_POSE_MM.copy()
        GRASP_POSE = GRASP_POSE_MM.copy()
        RESET_POSE = RESET_POSE_MM.copy()
        TARGET_POSE[:3] /= 1000.0
        GRASP_POSE[:3] /= 1000.0
        RESET_POSE[:3] /= 1000.0
        ABS_POSE_LIMIT_LOW = np.array(
            [0.15, -0.30, 0.30, -np.pi, -1.5, -np.pi], dtype=np.float64
        )
        ABS_POSE_LIMIT_HIGH = np.array(
            [0.80, 0.30, 1.35, np.pi, 1.5, np.pi], dtype=np.float64
        )
        REWARD_THRESHOLD = 0.001
        RANDOM_RESET = True
        RANDOM_XY_RANGE = 0.03
        RANDOM_RZ_RANGE = 0.0
        AUTO_QUICK_REGRASP = False
        ACTION_SCALE = (0.01, 0.0, 1)
        DISPLAY_IMAGE = True
        MAX_EPISODE_LENGTH = 1000
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
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose",] # ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 2000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    resnet_param_fixed: bool = False
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, replay_intervention_files=None):
        env_config = EnvConfig()
        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            if replay_intervention_files is not None:
                env = ReplayIntervention(env, replay_file_list=replay_intervention_files, action_scale=env_config.ACTION_SCALE)
            env = SpacemouseIntervention(env, gripper_control=False, deadband=0.05)
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
