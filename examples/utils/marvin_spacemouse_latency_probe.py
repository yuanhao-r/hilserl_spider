#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure SpaceMouse teleop probe for Tianji RAM env.

Purpose:
- Directly control the arm using SpaceMouse only (no policy / no replay).
- Print loop timing and simple input->motion latency events.

Usage example:
python marvin_spacemouse_latency_probe.py \
  --robot_ip 10.10.13.10 \
  --control_hz 50 \
  --linear_scale 3.0 \
  --deadband 0.002 \
  --unlock_eps 0.002 \
  --disable_cameras
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _setup_paths() -> None:
    """Prefer local project paths over global installs."""
    utils_dir = Path(__file__).resolve().parent
    repo_root = utils_dir.parents[1]  # .../hilserl_spider
    workspace_root = repo_root.parent  # .../teleop_tianji

    for p in [
        repo_root / "serl_robot_infra",
        repo_root / "serl_launcher",
        repo_root / "examples",
        workspace_root / "test_teleop" / "python",
        workspace_root / "test_teleop" / "demo",
    ]:
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)


def _apply_deadband(action: np.ndarray, deadband: float) -> np.ndarray:
    out = np.array(action, dtype=np.float64, copy=True)
    out[np.abs(out) < deadband] = 0.0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpaceMouse-only teleop latency probe (Tianji RAM env)."
    )
    parser.add_argument("--robot_ip", type=str, default="10.10.13.10")
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--max_sec", type=float, default=0.0, help="0 means run until Ctrl+C")
    parser.add_argument("--do_reset", action="store_true", help="Run env.reset() before teleop")
    parser.add_argument("--disable_cameras", action="store_true", help="Skip camera IO for cleaner latency")

    parser.add_argument("--linear_scale", type=float, default=None)
    parser.add_argument("--angular_scale", type=float, default=None)
    parser.add_argument("--deadband", type=float, default=None)
    parser.add_argument("--unlock_eps", type=float, default=None)

    parser.add_argument(
        "--intent_eps",
        type=float,
        default=0.01,
        help="Input threshold for a latency trigger event (in normalized action units).",
    )
    parser.add_argument(
        "--pose_move_eps_mm",
        type=float,
        default=0.4,
        help="TCP xyz motion threshold (mm) to mark response for one trigger.",
    )
    parser.add_argument(
        "--print_every_sec",
        type=float,
        default=1.0,
        help="Periodic stats print interval.",
    )
    return parser.parse_args()


def main() -> None:
    _setup_paths()

    # Import after path setup.
    from experiments.ram_insertion.config import EnvConfig
    from experiments.ram_insertion.wrapper import RAMEnv
    from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert

    args = parse_args()
    os.environ.setdefault("HILSERL_ARM_BACKEND", "tianji")

    cfg = EnvConfig()
    cfg.ROBOT_IP = args.robot_ip
    cfg.SERVER_URL = args.robot_ip

    if args.disable_cameras:
        cfg.REALSENSE_CAMERAS = {}
        cfg.IMAGE_CROP = {}
        cfg.DISPLAY_IMAGE = False

    if args.linear_scale is not None:
        cfg.SPACEMOUSE_LINEAR_SCALE = float(args.linear_scale)
    if args.angular_scale is not None:
        cfg.SPACEMOUSE_ANGULAR_SCALE = float(args.angular_scale)
    if args.deadband is not None:
        # This script applies deadband itself.
        pass
    if args.unlock_eps is not None:
        cfg.RESET_HANDOFF_UNLOCK_ACTION_EPS = float(args.unlock_eps)

    linear_scale = float(getattr(cfg, "SPACEMOUSE_LINEAR_SCALE", 1.0))
    angular_scale = float(getattr(cfg, "SPACEMOUSE_ANGULAR_SCALE", 1.0))
    deadband = float(0.02 if args.deadband is None else args.deadband)

    env = None
    spacemouse = None
    try:
        print("[INIT] creating RAMEnv ...", flush=True)
        env = RAMEnv(fake_env=False, save_video=False, config=cfg)
        env.max_episode_length = int(1e9)

        if args.do_reset:
            print("[INIT] env.reset() ...", flush=True)
            env.reset()
        else:
            env._update_currpos()
            env.cmd_pose = env.currpos.copy()
            env.nextpos = env.currpos.copy()
            env._post_reset_realign_pending = False

        spacemouse = SpaceMouseExpert()
        print(
            "[RUN] SpaceMouse teleop started. Ctrl+C to stop. "
            f"(linear_scale={linear_scale}, angular_scale={angular_scale}, deadband={deadband})",
            flush=True,
        )

        loop_period = (1.0 / args.control_hz) if args.control_hz > 1e-9 else 0.0
        next_tick = time.perf_counter()
        t0 = time.perf_counter()
        last_print_t = t0

        env._update_currpos()
        prev_pose = env.currpos.copy()
        moving_trigger_armed = False
        motion_onset_t = 0.0

        loop_count = 0
        step_dt_sum = 0.0
        step_dt_max = 0.0
        event_latencies_ms = []

        while True:
            now = time.perf_counter()
            if args.max_sec > 0 and (now - t0) >= args.max_sec:
                break

            raw_action, _buttons = spacemouse.get_action()
            raw_action = np.array(raw_action, dtype=np.float64).reshape(-1)
            if raw_action.shape[0] < 6:
                raw_action = np.pad(raw_action, (0, max(0, 6 - raw_action.shape[0])))

            expert_a = _apply_deadband(raw_action[:6], deadband=deadband)
            expert_a[:3] *= linear_scale
            expert_a[3:6] *= angular_scale

            motion_mag = float(np.max(np.abs(expert_a[:6])))
            if (not moving_trigger_armed) and motion_mag > float(args.intent_eps):
                moving_trigger_armed = True
                motion_onset_t = time.perf_counter()

            cmd = np.zeros((7,), dtype=np.float32)
            cmd[:6] = expert_a[:6].astype(np.float32)
            cmd[6] = 0.0

            t_step0 = time.perf_counter()
            obs, _rew, done, _truncated, _info = env.step(cmd)
            t_step1 = time.perf_counter()

            step_dt = t_step1 - t_step0
            loop_count += 1
            step_dt_sum += step_dt
            step_dt_max = max(step_dt_max, step_dt)

            pose = np.array(obs["state"]["tcp_pose"], dtype=np.float64).reshape(-1)
            dxyz_mm = float(np.linalg.norm(pose[:3] - prev_pose[:3]) * 1000.0)
            prev_pose = pose.copy()

            if moving_trigger_armed and dxyz_mm >= float(args.pose_move_eps_mm):
                latency_ms = (time.perf_counter() - motion_onset_t) * 1000.0
                event_latencies_ms.append(latency_ms)
                print(
                    f"[LATENCY] trigger->{args.pose_move_eps_mm:.2f}mm: {latency_ms:.1f} ms "
                    f"(step_dt={step_dt*1000:.1f} ms, dxyz={dxyz_mm:.2f} mm)",
                    flush=True,
                )
                moving_trigger_armed = False

            if done:
                env.curr_path_length = 0

            if (time.perf_counter() - last_print_t) >= float(args.print_every_sec):
                wall_elapsed = time.perf_counter() - t0
                loop_hz = loop_count / max(1e-6, wall_elapsed)
                avg_step_ms = (step_dt_sum / max(1, loop_count)) * 1000.0
                msg = (
                    f"[STAT] loop_hz={loop_hz:.1f}, avg_step={avg_step_ms:.1f}ms, "
                    f"max_step={step_dt_max*1000:.1f}ms, motion_mag={motion_mag:.4f}"
                )
                if event_latencies_ms:
                    msg += (
                        f", latency_ms(avg/min/max)="
                        f"{np.mean(event_latencies_ms):.1f}/"
                        f"{np.min(event_latencies_ms):.1f}/"
                        f"{np.max(event_latencies_ms):.1f}"
                    )
                print(msg, flush=True)
                last_print_t = time.perf_counter()

            if loop_period > 0:
                next_tick += loop_period
                sleep_dt = max(0.0, next_tick - time.perf_counter())
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("\n[EXIT] interrupted by user.", flush=True)
    finally:
        try:
            if spacemouse is not None:
                spacemouse.close()
        except Exception:
            pass
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        print("[DONE] shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
