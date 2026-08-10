"""Tianji tlop_drv joint and gripper IO."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


def _find_teleop_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "test_teleop").exists() and (parent / "tlop_drv").exists():
            return parent
    return Path("/home/ubuntu/teleop_tianji")


def inject_tianji_paths() -> Path:
    """Make tlop_drv and Tianji kinematics helper modules importable."""
    teleop_root = _find_teleop_root()
    test_teleop = teleop_root / "test_teleop"
    tlop_root = teleop_root / "tlop_drv"
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    candidates = (
        tlop_root,
        test_teleop / "python",
        test_teleop / "demo",
        Path("/opt/ros/humble/lib") / python_version / "site-packages",
        Path("/opt/ros/humble/local/lib") / python_version / "dist-packages",
    )

    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    return teleop_root


_TELEOP_ROOT = inject_tianji_paths()
_TLOP_IMPORT_ERROR = None

try:
    from tlop_client import ClientConfig, TlopClient, PayloadParameters
except Exception as exc:  # pragma: no cover - runtime dependency
    _TLOP_IMPORT_ERROR = exc
    ClientConfig = None
    TlopClient = None


@dataclass
class TianjiBackendConfig:
    transport: str = "udp"
    udp_host: str = "10.10.12.2"
    udp_port: int = 17030
    wait_ready_timeout: float = 3.0
    service_timeout: float = 0.05
    pose_timeout: float = 1.0
    gripper_timeout: float = 15.0
    trace_enabled: bool = False
    udp_step_ack: bool = False
    background_feedback: bool = True
    subscription_refresh_s: float = 1.0


class TianjiTlopBackend:
    """Thin wrapper around tlop_drv.

    All joint arrays exposed by this class are radians, matching tlop test.py.
    """

    def __init__(self, config: TianjiBackendConfig | None = None):
        self.config = config or TianjiBackendConfig()
        self.api = None
        self.last_error = ""

    def connect(self) -> None:
        if _TLOP_IMPORT_ERROR is not None:
            raise ImportError(
                "Failed to import tlop_client from /home/ubuntu/teleop_tianji/tlop_drv. "
                "Make sure that directory is on PYTHONPATH or backend.py can find it."
            ) from _TLOP_IMPORT_ERROR

        if str(self.config.transport).lower() != "udp":
            raise ValueError("tlop_drv only supports the UDP client API.")

        client_config = ClientConfig(
            host=str(self.config.udp_host),
            port=int(self.config.udp_port),
            timeout_s=float(self.config.service_timeout),
            ready_check_timeout_s=float(self.config.service_timeout),
            pose_timeout_s=float(self.config.pose_timeout),
            gripper_timeout_s=float(self.config.gripper_timeout),
            background_feedback=bool(self.config.background_feedback),
            subscription_refresh_s=float(self.config.subscription_refresh_s),
            heartbeat_enabled=False,
        )
        self.api = TlopClient(client_config)
        if not self.api.wait_ready(float(self.config.wait_ready_timeout), idx="LR"):
            self.last_error = getattr(self.api, "last_error", "")
            raise RuntimeError(f"tlop_drv is not ready: {self.last_error}")

    def _require_api(self):
        if self.api is None:
            self.connect()
        return self.api

    def get_joints_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        api = self._require_api()
        left, right = api.get_psn("LR", timeout=float(self.config.pose_timeout))
        return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)

    def get_joints_deg(self) -> Tuple[np.ndarray, np.ndarray]:
        left, right = self.get_joints_rad()
        return np.rad2deg(left), np.rad2deg(right)

    def send_joints_rad(
        self,
        left_rad: np.ndarray | None,
        right_rad: np.ndarray | None,
    ) -> bool:
        api = self._require_api()
        left_cmd = (
            None
            if left_rad is None
            else np.asarray(left_rad, dtype=np.float64).reshape(7).tolist()
        )
        right_cmd = (
            None
            if right_rad is None
            else np.asarray(right_rad, dtype=np.float64).reshape(7).tolist()
        )
        # print("right cmd: ", right_cmd, flush=True)
        ok = api.step(
            left_cmd,
            right_cmd,
            timeout=float(self.config.service_timeout),
            check_ready=False,
            wait_ack=False,
            validate=True,
        )
        ok = True
        self.last_error = "" if ok else getattr(api, "last_error", "")
        return bool(ok)

    def send_joints_deg(
        self,
        left_deg: np.ndarray | None,
        right_deg: np.ndarray | None,
    ) -> bool:
        return self.send_joints_rad(
            None
            if left_deg is None
            else np.deg2rad(np.asarray(left_deg, dtype=np.float64).reshape(7)),
            None
            if right_deg is None
            else np.deg2rad(np.asarray(right_deg, dtype=np.float64).reshape(7)),
        )

    def set_gripper_closed(self, closed: bool, arm: str = "R") -> bool:
        api = self._require_api()
        # tlop_drv uses open=True for opening. Keep env semantics: closed=True closes.
        ok = api.set_gripper(
            arm,
            open=not bool(closed),
            timeout=float(self.config.gripper_timeout),
            wait=False,
        )
        self.last_error = "" if ok else getattr(api, "last_error", "")
        return bool(ok)

    def set_payload_empty(self, arm="LR"):
        payload_empty = PayloadParameters(
            mass=0.2,
            com=(0.0, 0.0, 0.12),
            inertia=(1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0),
        )
        if not self.api.set_pldprm(payload_empty, arm):
            raise RuntimeError(self.api.last_error)

    def set_payload_full(self, arm="LR"):
        payload_full = PayloadParameters(
            mass=1.05,
            com=(0.0, 0.0, 0.15),
            inertia=(0.01, 0.01, 0.01, 0.0, 0.0, 0.0),
        )
        if not self.api.set_pldprm(payload_full, arm):
            raise RuntimeError(self.api.last_error)

    def close(self) -> None:
        if self.api is not None and hasattr(self.api, "close"):
            self.api.close()
            
    def get_eef_force(self, arm: str = "R") -> np.ndarray:
        api = self._require_api()
        eef = api.eeforce(arm)
        return np.asarray(eef.external_wrench_tcp, dtype=np.float64).reshape(6)
