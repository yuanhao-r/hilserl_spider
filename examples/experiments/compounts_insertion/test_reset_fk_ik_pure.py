#!/usr/bin/env python3
"""Pure right-arm kinematics check for RESET_JOINTS.

This script does not instantiate TianjiEnv, does not connect to tlop, and does
not send robot commands. It verifies only this closed loop:

    RESET_JOINTS -> FK reset pose -> right-arm IK -> FK reached pose
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TIANJI_DIR = REPO_ROOT / "serl_robot_infra/franka_env/envs/tianji"
DEFAULT_CONFIG = SCRIPT_DIR / "config.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_tianji_kinematics():
    """Load backend.py/kinematics.py without importing old franka_env.envs."""
    franka_env = types.ModuleType("franka_env")
    franka_env.__path__ = [str(REPO_ROOT / "serl_robot_infra/franka_env")]
    envs = types.ModuleType("franka_env.envs")
    envs.__path__ = [str(REPO_ROOT / "serl_robot_infra/franka_env/envs")]
    tianji = types.ModuleType("franka_env.envs.tianji")
    tianji.__path__ = [str(TIANJI_DIR)]

    sys.modules.setdefault("franka_env", franka_env)
    sys.modules.setdefault("franka_env.envs", envs)
    sys.modules.setdefault("franka_env.envs.tianji", tianji)

    _load_module("franka_env.envs.tianji.backend", TIANJI_DIR / "backend.py")
    return _load_module("franka_env.envs.tianji.kinematics", TIANJI_DIR / "kinematics.py")


def read_reset_joints_deg(config_path: Path) -> np.ndarray:
    tree = ast.parse(config_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EnvConfig":
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if not any(isinstance(t, ast.Name) and t.id == "RESET_JOINTS" for t in stmt.targets):
                    continue
                value = stmt.value
                if isinstance(value, ast.Call):
                    value = value.args[0]
                return np.asarray(ast.literal_eval(value), dtype=np.float64).reshape(7)
    raise ValueError(f"Cannot find EnvConfig.RESET_JOINTS in {config_path}")


def transform_error(kinematics_module, target_pose6: np.ndarray, reached_pose6: np.ndarray):
    target_tf = kinematics_module.pose6_to_transform(target_pose6)
    reached_tf = kinematics_module.pose6_to_transform(reached_pose6)
    xyz_err = reached_tf[:3, 3] - target_tf[:3, 3]
    rot_err_rad = Rotation.from_matrix(target_tf[:3, :3].T @ reached_tf[:3, :3]).magnitude()
    return xyz_err, rot_err_rad


def wrapped_joint_delta_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--xyz-threshold", type=float, default=0.01)
    parser.add_argument("--rot-threshold", type=float, default=0.05)
    args = parser.parse_args()

    kin_mod = load_tianji_kinematics()
    reset_joints_deg = read_reset_joints_deg(args.config)
    reset_joints_rad = np.deg2rad(reset_joints_deg)

    kin = kin_mod.TianjiKinematics()
    reset_pose = kin.pose_from_right_joints_deg(reset_joints_deg)
    print("RESET_JOINTS deg:", np.round(reset_joints_deg, 6).tolist())
    print("FK reset pose [x y z r p y]:", np.round(reset_pose, 6).tolist())
    reset_world_tool_tf = kin_mod.pose6_to_transform(reset_pose)
    reset_local_flange_tf = np.linalg.inv(kin.base_right_tf) @ reset_world_tool_tf @ np.linalg.inv(kin.tool_tf)
    reset_local_flange_wxyzxyz = kin_mod.transform_to_wxyzxyz(reset_local_flange_tf)
    reset_local_flange_xyzwxyz = np.concatenate(
        [reset_local_flange_wxyzxyz[1:4], reset_local_flange_wxyzxyz[0:1], reset_local_flange_wxyzxyz[4:7]]
    )
    reset_local_fk_tf = kin.controller.compute_fk(reset_joints_rad)
    reset_local_fk_wxyzxyz = kin_mod.transform_to_wxyzxyz(reset_local_fk_tf)
    reset_local_fk_xyzwxyz = np.concatenate(
        [reset_local_fk_wxyzxyz[1:4], reset_local_fk_wxyzxyz[0:1], reset_local_fk_wxyzxyz[4:7]]
    )

    print("right local FK xyzwxyz:", np.round(reset_local_fk_xyzwxyz, 6).tolist())
    print("right IK target xyzwxyz:", np.round(reset_local_flange_xyzwxyz, 6).tolist())
    print(
        "right local target diff:",
        np.round(reset_local_flange_xyzwxyz - reset_local_fk_xyzwxyz, 8).tolist(),
    )

    ik_right_rad = kin.controller.compute_ik(reset_joints_rad, reset_local_flange_xyzwxyz)
    if ik_right_rad is None:
        print("IK failed: solver returned None")
        return 2

    ik_pose = kin.fk_right_pose6(ik_right_rad)
    xyz_err, rot_err_rad = transform_error(kin_mod, reset_pose, ik_pose)
    joint_delta_rad = wrapped_joint_delta_rad(ik_right_rad, reset_joints_rad)

    print("config:", args.config)
    print("RESET_JOINTS deg:", np.round(reset_joints_deg, 6).tolist())
    print("FK reset pose [x y z r p y]:", np.round(reset_pose, 6).tolist())
    print("right-only IK target xyzwxyz:", np.round(reset_local_flange_xyzwxyz, 6).tolist())
    print("IK right joints deg:", np.round(np.rad2deg(ik_right_rad), 6).tolist())
    print("FK after IK pose [x y z r p y]:", np.round(ik_pose, 6).tolist())
    print("xyz error m:", np.round(xyz_err, 8).tolist(), "norm=", float(np.linalg.norm(xyz_err)))
    print("rot error rad:", float(rot_err_rad))
    print("joint delta deg:", np.round(np.rad2deg(joint_delta_rad), 6).tolist())

    ok = np.linalg.norm(xyz_err) <= args.xyz_threshold and rot_err_rad <= args.rot_threshold
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
