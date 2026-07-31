import os
from pathlib import Path

import numpy as np

from experiments.config import DefaultTrainingConfig
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.tianji.env import DefaultTianjiEnvConfig, TianjiEnv
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    ReplayIntervention,
    SpacemouseIntervention,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper


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
            "camera_index": 2,
            "dim": (1280, 720),
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[280:, 320:960],
    }
    DISPLAY_IMAGE = False

    # Fill this with the fixed reset/right-arm standby joint angles you want.
    # Unit: degrees, same as ram_insertion's Tianji config.
    RESET_JOINTS = np.array(
        [86.02954, 76.22053, -80.41872, -119.65581, 109.78898, 6.75465, -25.18097],
        dtype=np.float64,
    )
    RESET_POSE = np.array(
        [ 0.81554249, -0.25109333,0.94290184 , 1.69872075, -1.4762164,   1.55535381],
        dtype=np.float64,
    )
    CONTROL_HZ = 10
    CONTROL_TIME = 1.0 / CONTROL_HZ
    ACTION_SCALE = np.array([0.005, 0.005, 1.0], dtype=np.float64)
    SPACEMOUSE_LINEAR_SCALE = 1.0
    SPACEMOUSE_ANGULAR_SCALE = 1.0

    ABS_POSE_LIMIT_LOW = np.array(
        [0.7505, -0.1617, 0.5230, -np.pi, -np.pi, -np.pi],
        dtype=np.float64,
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [1.8005, -0.08217, 1.6230, np.pi, np.pi, np.pi],
        dtype=np.float64,
    )

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.0
    RANDOM_DX_MIN = -0.015
    RANDOM_DX_MAX = 0.015
    RANDOM_DY_MIN = -0.015
    RANDOM_DY_MAX = 0.015
    RANDOM_DZ_MIN = 0.0
    RANDOM_DZ_MAX = 0.0

    RESET_WAIT_SEC = 2.0
    RANDOM_RESET_WAIT_SEC = 2.0
    RESET_HOLD_HZ = 40.0
    GRIPPER_SLEEP = 0.6
    MAX_EPISODE_LENGTH = 200


class TrainConfig(DefaultTrainingConfig):
    image_keys = list(EnvConfig.REALSENSE_CAMERAS.keys())
    classifier_keys = list(EnvConfig.REALSENSE_CAMERAS.keys())
    proprio_keys = ["tcp_pose"]
    max_traj_length = EnvConfig.MAX_EPISODE_LENGTH
    buffer_period = 1000
    checkpoint_period = 2000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    resnet_param_fixed = True
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        replay_intervention_files=None,
    ):
        del classifier
        env_config = EnvConfig()
        self.image_keys = list(env_config.realsense_cameras.keys())
        self.classifier_keys = list(env_config.realsense_cameras.keys())

        env = TianjiEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            if replay_intervention_files is not None:
                env = ReplayIntervention(
                    env,
                    replay_file_list=replay_intervention_files,
                    action_scale=env_config.action_scale,
                )
            env = SpacemouseIntervention(
                env,
                gripper_control=False,
                # deadband=float(os.environ.get("HILSERL_SPACEMOUSE_DEADBAND", "0.08")),
                deadband = 0.08,
                expert_linear_scale=getattr(env_config, "SPACEMOUSE_LINEAR_SCALE", 1.0),
                expert_angular_scale=getattr(env_config, "SPACEMOUSE_ANGULAR_SCALE", 1.0),
            )
        env = RelativeFrame(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

    def process_demos(self, demo):
        return demo
