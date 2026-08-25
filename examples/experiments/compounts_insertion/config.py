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

from experiments.compounts_insertion.wrapper import COMPONENTEnv, TiltObsWrapper, FailureOnTiltWrapper


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
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[280:, 320:960],
    }
    DISPLAY_IMAGE = True

    # Fill this with the fixed reset/right-arm standby joint angles you want.
    # Unit: degrees, same as ram_insertion's Tianji config.
    # RESET_JOINTS = np.array(
    #     [  94.036583,   83.514676,  -87.355121, -108.510546,   95.927145,
    # 0.781873,  -21.707728],
    #     dtype=np.float64,
    # )
    # TOP_JOINTS = np.array([94.82210867, 78.68538263, -85.90459355, -109.36515261, 101.50909910, 2.13432508, -29.02953694])
    # TOP_JOINTS = np.array( [100.52533056, 75.83623540, -85.95455547, -111.59229686, 104.65297582, 2.33508949, -34.37380078], dtype=np.float64)
    TOP_JOINTS = np.array([ 107.519233,  84.583458, -92.759795, -95.491898,  96.455877,   0.936127,
 -21.966858])
    RESET_JOINTS = TOP_JOINTS
    # TOP_JOINTS = np.array([  97.436458,   83.435718,  -85.466166, -108.900828,  100.470269,
    # 2.845767,  -29.746135])
    # TARGET_JOINTS = np.array([95.03123827, 82.85038473, -79.78311246, -110.53914313, 103.97986500, 10.32664752, -31.54052448], dtype=np.float64)
    # TARGET_JOINTS = np.array([ 91.39966 ,  66.054413, -81.307066, -94.894359, 115.00575 , 5.105512, -6.316172])
    # TARGET_JOINTS = np.array([91.38258573, 79.93082098, -86.75887362, -105.77373856, 99.65964863, 2.11324024, -19.66660443])
    # 这个抓的有点紧，会卡着放不进去，平行了
    # TARGET_JOINTS = np.array([ 90.136975,  83.964157, -87.215177, -99.153841,  96.565333,   1.84882 ,
#   -7.734586])

    TARGET_JOINTS = np.array([ 105.004142,  84.752318, -92.956214, -92.499624,  96.108915,   1.051034,
 -18.474222])
    CONTROL_HZ = 10
    CONTROL_TIME = 1.0 / CONTROL_HZ
    ACTION_SCALE = np.array([0.005, 0, 1.0], dtype=np.float64)
    SPACEMOUSE_LINEAR_SCALE = 1.0
    SPACEMOUSE_ANGULAR_SCALE = 1.0
    SPACEMOUSE_DEADBAND = 0.02

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
    RANDOM_DX_MIN = -0.015*0
    RANDOM_DX_MAX = 0.015*0
    RANDOM_DY_MIN = -0.015*0
    RANDOM_DY_MAX = 0.015*0
    RANDOM_DZ_MIN = 0.0
    RANDOM_DZ_MAX = 0.0

    RESET_WAIT_SEC = 2.0
    RANDOM_RESET_WAIT_SEC = 2.0
    RESET_HOLD_HZ = 40.0
    GRIPPER_SLEEP = 0.6
    MAX_EPISODE_LENGTH = 200

    MAX_TILT_DEGREE = 5.0




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

        env = COMPONENTEnv(
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
                deadband=getattr(env_config, "SPACEMOUSE_DEADBAND", 0.02),
                expert_linear_scale=getattr(env_config, "SPACEMOUSE_LINEAR_SCALE", 1.0),
                expert_angular_scale=getattr(env_config, "SPACEMOUSE_ANGULAR_SCALE", 1.0),
            )
        env = TiltObsWrapper(env)
        env = FailureOnTiltWrapper(env, max_tilt_deg=getattr(env_config, "MAX_TILT_DEG", 5.0))
        
        env = RelativeFrame(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

    def process_demos(self, demo):
        return demo
