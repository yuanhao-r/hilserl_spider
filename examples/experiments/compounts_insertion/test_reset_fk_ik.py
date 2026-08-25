#!/usr/bin/env python3
"""Check that RESET_JOINTS FK and TianjiEnv reset IK agree.

This script uses the same env entry point as reset:

    self._send_pos_command(target_pose, force_seed_from_current=True, reset_stats=True)

It assumes backend.send_joints_rad has api.step commented out when you do not
want the robot to move.
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


def prepare_franka_env_namespace():
    """Avoid importing franka_env.envs.__init__, which imports the old Tianji env."""
    franka_env = types.ModuleType("franka_env")
    franka_env.__path__ = [str(REPO_ROOT / "serl_robot_infra/franka_env")]
    envs = types.ModuleType("franka_env.envs")
    envs.__path__ = [str(REPO_ROOT / "serl_robot_infra/franka_env/envs")]
    tianji = types.ModuleType("franka_env.envs.tianji")
    tianji.__path__ = [str(TIANJI_DIR)]

    sys.modules.setdefault("franka_env", franka_env)
    sys.modules.setdefault("franka_env.envs", envs)
    sys.modules.setdefault("franka_env.envs.tianji", tianji)


def load_tianji_env_module():
    prepare_franka_env_namespace()
    _load_module("franka_env.envs.tianji.backend", TIANJI_DIR / "backend.py")
    _load_module("franka_env.envs.tianji.kinematics", TIANJI_DIR / "kinematics.py")
    return _load_module("franka_env.envs.tianji.env", TIANJI_DIR / "env.py")


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

    env_mod = load_tianji_env_module()
    reset_joints_deg = read_reset_joints_deg(args.config)

    class TestConfig(env_mod.DefaultTianjiEnvConfig):
        RESET_JOINTS = reset_joints_deg
        RANDOM_RESET = False
        REALSENSE_CAMERAS = {}
        DISPLAY_IMAGE = False
        MAX_EPISODE_LENGTH = 1

    env = env_mod.TianjiEnv(
        fake_env=False,
        save_video=False,
        config=TestConfig(),
    )

    reset_pose = env.resetpos.copy()
    # reset_pose[2] += 0.25  # Move up by 10 cm
    print("input pose: ", reset_pose)

    for i in range(100000):
        ret = env._send_pos_command(
            reset_pose,
            force_seed_from_current=True,
            reset_stats=True,
        )
    if ret != 0:
        print(f"_send_pos_command failed with ret={ret}")
        return 2

    ik_right_rad = env._last_sent_right_rad
    if ik_right_rad is None:
        print("_send_pos_command did not produce _last_sent_right_rad")
        return 2

    ik_pose = env.kinematics.fk_right_pose6(ik_right_rad)
    xyz_err, rot_err_rad = transform_error(env_mod, reset_pose, ik_pose)
    joint_delta_rad = wrapped_joint_delta_rad(ik_right_rad, np.deg2rad(reset_joints_deg))

    print("config:", args.config)
    print("RESET_JOINTS deg:", np.round(reset_joints_deg, 6).tolist())
    print("FK reset pose [x y z r p y]:", np.round(reset_pose, 6).tolist())
    print("IK right joints deg:", np.round(np.rad2deg(ik_right_rad), 6).tolist())
    print("FK after IK pose [x y z r p y]:", np.round(ik_pose, 6).tolist())
    print("xyz error m:", np.round(xyz_err, 8).tolist(), "norm=", float(np.linalg.norm(xyz_err)))
    print("rot error rad:", float(rot_err_rad))
    print("joint delta deg:", np.round(np.rad2deg(joint_delta_rad), 6).tolist())

    ok = np.linalg.norm(xyz_err) <= args.xyz_threshold and rot_err_rad <= args.rot_threshold
    print("PASS" if ok else "FAIL")
    import time
    time.sleep(1111111111)


if __name__ == "__main__":
    raise SystemExit(main())
