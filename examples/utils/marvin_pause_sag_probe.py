#!/usr/bin/env python3
"""
Probe whether the robot sags when command streaming is paused.

Flow:
1) Send right-arm joint command for a while (with a small delta).
2) Stop sending any command and only monitor drift.
3) Send command again and observe recovery.

This script uses the same low-level controller path as TianjiEnv:
`test_teleop/demo/marvin_arm_controller.py`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _inject_paths() -> Path:
    here = Path(__file__).resolve()
    workspace_root = here.parents[3]  # .../teleop_tianji
    demo_dir = workspace_root / "test_teleop" / "demo"
    python_dir = workspace_root / "test_teleop" / "python"

    for p in (demo_dir, python_dir):
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)
    return workspace_root


_WORKSPACE_ROOT = _inject_paths()
from marvin_arm_controller import MarvinArmController  # noqa: E402


def _read_state(controller: MarvinArmController):
    ql_rad = np.array(controller.get_joint_pos_rad(arm_id=1), dtype=np.float64)
    qr_rad = np.array(controller.get_joint_pos_rad(arm_id=2), dtype=np.float64)
    ql_deg = ql_rad / controller.DEG_TO_RAD
    qr_deg = qr_rad / controller.DEG_TO_RAD
    fk_r = controller.compute_fk(qr_rad)
    pos_r = None if fk_r is None else np.array(fk_r[:3, 3], dtype=np.float64)
    return ql_deg, qr_deg, pos_r


def _fmt_mm(v_m: float | None) -> str:
    if v_m is None:
        return "nan"
    return f"{v_m * 1000.0:+7.2f}"


def _print_drift(
    tag: str,
    now_qr_deg: np.ndarray,
    ref_qr_deg: np.ndarray,
    now_pos_r: np.ndarray | None,
    ref_pos_r: np.ndarray | None,
    joint_index: int,
):
    dq = now_qr_deg - ref_qr_deg
    max_abs_deg = float(np.max(np.abs(dq)))
    sel_deg = float(dq[joint_index])
    dxyz_m = None if (now_pos_r is None or ref_pos_r is None) else (now_pos_r - ref_pos_r)
    dz_m = None if dxyz_m is None else float(dxyz_m[2])
    dnorm_m = None if dxyz_m is None else float(np.linalg.norm(dxyz_m))
    dnorm_mm = None if dnorm_m is None else dnorm_m * 1000.0
    dnorm_text = "nan" if dnorm_mm is None else f"{dnorm_mm:7.2f}"

    print(
        f"[{tag}] "
        f"dq_max={max_abs_deg:6.3f} deg, "
        f"dq_j{joint_index + 1}={sel_deg:+6.3f} deg, "
        f"dz(mm)={_fmt_mm(dz_m)}, "
        f"|dxyz|(mm)={dnorm_text}"
    )


def _send_stream(
    controller: MarvinArmController,
    target_qr_deg: np.ndarray,
    duration_sec: float,
    send_hz: float,
    monitor_hz: float,
    phase_name: str,
    ref_qr_deg: np.ndarray,
    ref_pos_r: np.ndarray | None,
    joint_index: int,
):
    if duration_sec <= 0:
        return
    send_period = 1.0 / max(1.0, float(send_hz))
    monitor_period = 1.0 / max(1.0, float(monitor_hz))
    end_t = time.perf_counter() + duration_sec
    next_send = time.perf_counter()
    next_monitor = time.perf_counter()

    while time.perf_counter() < end_t:
        now = time.perf_counter()
        if now >= next_send:
            # Keep left arm at current measured position (same style as TianjiEnv right-arm control).
            ql_cmd_deg = (
                np.array(controller.get_joint_pos_rad(arm_id=1), dtype=np.float64)
                / controller.DEG_TO_RAD
            )
            controller.step(
                np.array(ql_cmd_deg, dtype=np.float64),
                np.array(target_qr_deg, dtype=np.float64),
                verbose=False,
            )
            next_send += send_period

        if now >= next_monitor:
            _, qr_now_deg, pos_now_r = _read_state(controller)
            _print_drift(
                phase_name,
                qr_now_deg,
                ref_qr_deg,
                pos_now_r,
                ref_pos_r,
                joint_index,
            )
            next_monitor += monitor_period

        time.sleep(0.001)


def _pause_only_monitor(
    controller: MarvinArmController,
    duration_sec: float,
    monitor_hz: float,
    phase_name: str,
    ref_qr_deg: np.ndarray,
    ref_pos_r: np.ndarray | None,
    joint_index: int,
):
    if duration_sec <= 0:
        return
    monitor_period = 1.0 / max(1.0, float(monitor_hz))
    end_t = time.perf_counter() + duration_sec
    next_monitor = time.perf_counter()

    while time.perf_counter() < end_t:
        now = time.perf_counter()
        if now >= next_monitor:
            _, qr_now_deg, pos_now_r = _read_state(controller)
            _print_drift(
                phase_name,
                qr_now_deg,
                ref_qr_deg,
                pos_now_r,
                ref_pos_r,
                joint_index,
            )
            next_monitor += monitor_period
        time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser(
        description="Send -> pause(no command) -> send again, and monitor sag."
    )
    parser.add_argument("--robot_ip", default="10.10.13.10")
    parser.add_argument("--mode", choices=["impedance", "position"], default="impedance")
    parser.add_argument("--vel_ratio", type=int, default=90)
    parser.add_argument("--acc_ratio", type=int, default=90)
    parser.add_argument("--joint_index", type=int, default=2, help="0..6 for right arm")
    parser.add_argument("--delta_deg", type=float, default=-2.0, help="small right-arm delta")
    parser.add_argument("--send_hz", type=float, default=50.0)
    parser.add_argument("--monitor_hz", type=float, default=10.0)
    parser.add_argument("--send_sec", type=float, default=3.0)
    parser.add_argument("--pause_sec", type=float, default=5.0)
    parser.add_argument("--resume_sec", type=float, default=3.0)
    parser.add_argument(
        "--keep_enabled",
        action="store_true",
        help="Do not call cleanup() on exit (for quick repeated tests).",
    )
    args = parser.parse_args()

    if args.joint_index < 0 or args.joint_index > 6:
        raise ValueError("--joint_index must be in [0, 6]")

    controller = MarvinArmController(robot_ip=args.robot_ip, test_mode=False)

    try:
        print(f"[INIT] workspace={_WORKSPACE_ROOT}")
        print(f"[INIT] connect robot {args.robot_ip} ...")
        controller.init_kinematics()
        ok = controller.init_robot_connection()
        # if not ok:
        #     raise RuntimeError("robot connection failed")

        # if args.mode == "position":
        #     ok = controller.enter_position_mode(
        #         vel_ratio=args.vel_ratio,
        #         acc_ratio=args.acc_ratio,
        #     )
        #     if not ok:
        #         raise RuntimeError("enter_position_mode failed")
        # else:
        #     ok = controller.enter_cartesian_impedance_mode(
        #         vel_ratio=args.vel_ratio,
        #         acc_ratio=args.acc_ratio,
        #     )
        #     if not ok:
        #         raise RuntimeError("enter_cartesian_impedance_mode failed")

        # time.sleep(0.2)
        # ql0_deg, qr0_deg, pos0_r = _read_state(controller)
        # target_qr_deg = np.array(qr0_deg, dtype=np.float64)
        # target_qr_deg[args.joint_index] += float(args.delta_deg)

        # print(
        #     f"[TARGET] mode={args.mode}, right_j{args.joint_index + 1} "
        #     f"{qr0_deg[args.joint_index]:.3f} -> {target_qr_deg[args.joint_index]:.3f} deg"
        # )
        # print(
        #     f"[PLAN] send {args.send_sec:.1f}s @ {args.send_hz:.1f}Hz -> "
        #     f"pause {args.pause_sec:.1f}s (no command) -> "
        #     f"send {args.resume_sec:.1f}s"
        # )

        # print("[PHASE1] send command stream ...")
        # _send_stream(
        #     controller=controller,
        #     target_qr_deg=target_qr_deg,
        #     duration_sec=args.send_sec,
        #     send_hz=args.send_hz,
        #     monitor_hz=args.monitor_hz,
        #     phase_name="send1",
        #     ref_qr_deg=qr0_deg,
        #     ref_pos_r=pos0_r,
        #     joint_index=args.joint_index,
        # )

        # # Use phase-1 end state as the pause drift reference.
        # _, qr_anchor_deg, pos_anchor_r = _read_state(controller)
        # print("[PHASE2] stop all commands, monitor drift ...")
        # _pause_only_monitor(
        #     controller=controller,
        #     duration_sec=args.pause_sec,
        #     monitor_hz=args.monitor_hz,
        #     phase_name="pause",
        #     ref_qr_deg=qr_anchor_deg,
        #     ref_pos_r=pos_anchor_r,
        #     joint_index=args.joint_index,
        # )

        # print("[PHASE3] resume command stream ...")
        # _send_stream(
        #     controller=controller,
        #     target_qr_deg=target_qr_deg,
        #     duration_sec=args.resume_sec,
        #     send_hz=args.send_hz,
        #     monitor_hz=args.monitor_hz,
        #     phase_name="send2",
        #     ref_qr_deg=qr_anchor_deg,
        #     ref_pos_r=pos_anchor_r,
        #     joint_index=args.joint_index,
        # )

        # print("[DONE] probe finished.")
        # print(
        #     "[TIP] If pause phase shows growing |dxyz|/dz, it supports "
        #     "the 'no command -> sag' hypothesis in current mode."
        # )
    finally:
        if args.keep_enabled:
            print("[EXIT] keep_enabled=True, skip cleanup().")
        else:
            controller.cleanup()


if __name__ == "__main__":
    main()

