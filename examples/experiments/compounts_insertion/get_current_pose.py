#!/usr/bin/env python3
"""Print current Tianji right TCP pose from live joint feedback."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent

for path in (
    REPO_ROOT / "serl_robot_infra",
    REPO_ROOT / "serl_launcher",
    REPO_ROOT / "examples",
    WORKSPACE_ROOT / "test_teleop" / "python",
    WORKSPACE_ROOT / "test_teleop" / "demo",
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.compounts_insertion.config import EnvConfig
from franka_env.envs.tianji.backend import TianjiTlopBackend
from franka_env.envs.tianji.kinematics import TianjiKinematics, pose6_to_transform


def fmt(arr: np.ndarray, precision: int = 6) -> str:
    return np.array2string(
        np.asarray(arr, dtype=np.float64),
        precision=precision,
        suppress_small=False,
        separator=", ",
    )


def main() -> None:
    config = EnvConfig()
    config.__post_init__()

    backend = TianjiTlopBackend(config.backend)
    kinematics = None
    try:
        backend.connect()
        kinematics = TianjiKinematics(config.kinematics)

        left_rad, right_rad = backend.get_joints_rad()
        pose6 = kinematics.fk_right_pose6(right_rad)

        # Keep IK internals aligned with the same side effects as TianjiEnv._update_currpos().
        kinematics.update_locked_left_head(left_rad)
        kinematics.right_target_pose = pose6_to_transform(pose6)

        print("left_joints_rad:")
        print(fmt(left_rad))
        print("right_joints_rad:")
        print(fmt(right_rad))
        print("right_joints_deg:")
        print(fmt(np.rad2deg(right_rad)))
        print("current_right_tcp_pose6 [x, y, z, roll, pitch, yaw]:")
        print(fmt(pose6))
    finally:
        if kinematics is not None:
            kinematics.close()
        backend.close()


if __name__ == "__main__":
    main()
