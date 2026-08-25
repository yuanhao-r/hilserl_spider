#!/usr/bin/env python3
"""Compute IK for the current Tianji right TCP pose with a z offset."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read current Tianji joint feedback, compute the current right TCP "
            "6D pose, add a z offset, and solve IK for the modified pose."
        )
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=0.05,
        help="Offset added to current TCP z. Unit is the same as pose6 xyz, normally meters.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of digits printed for arrays.",
    )
    parser.add_argument(
        "--no-force-seed-from-current",
        action="store_true",
        help="Use the current joints as IK seed without forcing the IK solver to resync first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = EnvConfig()
    config.__post_init__()

    backend = TianjiTlopBackend(config.backend)
    kinematics = None
    try:
        backend.connect()
        kinematics = TianjiKinematics(config.kinematics)

        left_rad, right_rad = backend.get_joints_rad()
        current_pose6 = kinematics.fk_right_pose6(right_rad)
        target_pose6 = current_pose6.copy()
        target_pose6[2] += float(args.dz)

        # Match TianjiEnv._update_currpos() before solving IK.
        kinematics.update_locked_left_head(left_rad)
        kinematics.right_target_pose = pose6_to_transform(target_pose6)

        left_target_rad, right_target_rad, ik_error = kinematics.compute_full_body_ik(
            current_left_rad=left_rad,
            current_right_rad=right_rad,
            force_seed_from_current=not args.no_force_seed_from_current,
        )

        print("current_left_joints_rad:")
        print(fmt(left_rad, args.precision))
        print("current_right_joints_rad:")
        print(fmt(right_rad, args.precision))
        print("current_right_joints_deg:")
        print(fmt(np.rad2deg(right_rad), args.precision))
        print("current_right_tcp_pose6 [x, y, z, roll, pitch, yaw]:")
        print(fmt(current_pose6, args.precision))
        print(f"z_offset: {float(args.dz):.{args.precision}f}")
        print("target_right_tcp_pose6 [x, y, z, roll, pitch, yaw]:")
        print(fmt(target_pose6, args.precision))
        print(f"ik_error: {float(ik_error):.{args.precision}f}")

        if right_target_rad is None:
            print("target_right_joints_rad: IK failed")
            print("target_right_joints_deg: IK failed")
            raise SystemExit(1)

        print("target_left_joints_rad:")
        print(fmt(left_target_rad, args.precision))
        print("target_right_joints_rad:")
        print(fmt(right_target_rad, args.precision))
        print("target_right_joints_deg:")
        print(fmt(np.rad2deg(right_target_rad), args.precision))

        reached_pose6 = kinematics.fk_right_pose6(right_target_rad)
        print("fk_check_right_tcp_pose6 [x, y, z, roll, pitch, yaw]:")
        print(fmt(reached_pose6, args.precision))
        print("fk_check_pose_delta [x, y, z, roll, pitch, yaw]:")
        print(fmt(reached_pose6 - target_pose6, args.precision))
    finally:
        if kinematics is not None:
            kinematics.close()
        backend.close()


if __name__ == "__main__":
    main()
