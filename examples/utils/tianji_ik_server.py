#!/usr/bin/env python3
"""Standalone HTTP IK server for Tianji/Marvin full-body IK."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


def _inject_paths() -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    teleop_root = workspace_root / "test_teleop"
    paths = [
        teleop_root / "python",
        teleop_root / "demo",
        workspace_root / "hilserl_spider" / "serl_robot_infra",
    ]
    for path in paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return workspace_root


_WORKSPACE_ROOT = _inject_paths()


def _as_array(payload: dict[str, Any], key: str, shape=None) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"missing field: {key}")
    arr = np.asarray(payload[key], dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{key} expected shape {shape}, got {arr.shape}")
    return arr


def _matrix_to_xyzwxyz(transform_to_wxyzxyz, matrix: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    pose_wxyzxyz = transform_to_wxyzxyz(matrix)
    pose_xyzwxyz = np.concatenate([pose_wxyzxyz[4:7], pose_wxyzxyz[0:4]])
    pose_xyzwxyz[2] += float(z_offset)
    return pose_xyzwxyz


class TianjiIKService:
    def __init__(self, urdf_path: str, scale_t: float = 1.0, scale_a: float = 1.0):
        from utils.marvin_dual_arm_waist_teleop import DualArmWaistIK
        from utils.kine_model import transform_to_wxyzxyz

        self.transform_to_wxyzxyz = transform_to_wxyzxyz
        self.solver = DualArmWaistIK(urdf_path=urdf_path, enable_viewer=True)
        self.solver.set_scale(scale_t, scale_a)
        self.solver.start()
        self.lock = threading.Lock()

    def solve(self, payload: dict[str, Any]) -> dict[str, Any]:
        current_ql = _as_array(payload, "current_ql_rad", (7,))
        current_qr = _as_array(payload, "current_qr_rad", (7,))
        left_tf = _as_array(payload, "left_target_pose", (4, 4))
        right_tf = _as_array(payload, "right_target_pose", (4, 4))
        head_tf = _as_array(payload, "head_target_pose", (4, 4))
        body_waist_q = _as_array(payload, "body_waist_q", (2,))
        head_z_offset = float(payload.get("head_z_offset", 0.0))
        force_seed_from_current = bool(payload.get("force_seed_from_current", False))

        current_cfg = np.concatenate([body_waist_q, current_qr, current_ql])
        left_pose = _matrix_to_xyzwxyz(self.transform_to_wxyzxyz, left_tf)
        right_pose = _matrix_to_xyzwxyz(self.transform_to_wxyzxyz, right_tf)
        head_pose = _matrix_to_xyzwxyz(
            self.transform_to_wxyzxyz, head_tf, z_offset=head_z_offset
        )

        start = time.perf_counter()
        with self.lock:
            if force_seed_from_current and hasattr(self.solver, "sync_with_current_cfg"):
                self.solver.sync_with_current_cfg(
                    current_cfg[2:], reset_delta_reference=True
                )
                ik_seed_cfg = None
            else:
                ik_seed_cfg = current_cfg[2:]
            solver_cfg, solve_time = self.solver.compute_ik(
                ik_seed_cfg,
                head_pose,
                right_pose,
                left_pose,
                force_seed_from_current=force_seed_from_current,
            )

        elapsed = time.perf_counter() - start
        if solver_cfg is None:
            return {
                "success": False,
                "error": "IK solver returned None",
                "solve_time": float(solve_time),
                "elapsed": elapsed,
            }

        full_cfg = np.concatenate([current_cfg[:2], np.asarray(solver_cfg)])
        return {
            "success": True,
            "solver_cfg_rad": np.asarray(solver_cfg, dtype=np.float64).tolist(),
            "full_cfg_rad": full_cfg.tolist(),
            "qr_target_rad": full_cfg[2:9].tolist(),
            "ql_target_rad": full_cfg[9:].tolist(),
            "solve_time": float(solve_time),
            "elapsed": elapsed,
        }


def _make_handler(service: TianjiIKService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"success": False, "error": "not found"})

        def do_POST(self):
            if self.path != "/solve_ik":
                self._send_json(404, {"success": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send_json(200, service.solve(payload))
            except Exception as exc:
                self._send_json(400, {"success": False, "error": str(exc)})

    return Handler


def main() -> None:
    default_urdf = (
        _WORKSPACE_ROOT / "test_teleop" / "models" / "spiderrobot" / "robot_marvin_arms.urdf"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--urdf_path", default=str(default_urdf))
    parser.add_argument("--scale_t", type=float, default=1.0)
    parser.add_argument("--scale_a", type=float, default=1.0)
    args = parser.parse_args()

    service = TianjiIKService(
        urdf_path=args.urdf_path,
        scale_t=args.scale_t,
        scale_a=args.scale_a,
    )
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(service))
    print(
        f"Tianji IK server listening on http://{args.host}:{args.port} "
        f"urdf={args.urdf_path}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
