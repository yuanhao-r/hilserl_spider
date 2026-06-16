"""HTTP client for the standalone Tianji IK service."""

from __future__ import annotations

from typing import Any

import numpy as np
import requests


class TianjiIKClient:
    def __init__(self, url: str, timeout: float = 1.0):
        self.url = url.rstrip("/")
        self.timeout = float(timeout)

    @staticmethod
    def _array(value: Any) -> list:
        return np.asarray(value, dtype=np.float64).tolist()

    def solve(
        self,
        *,
        current_ql_rad,
        current_qr_rad,
        left_target_pose,
        right_target_pose,
        head_target_pose,
        body_waist_q,
        head_z_offset: float,
        force_seed_from_current: bool,
    ) -> dict:
        payload = {
            "current_ql_rad": self._array(current_ql_rad),
            "current_qr_rad": self._array(current_qr_rad),
            "left_target_pose": self._array(left_target_pose),
            "right_target_pose": self._array(right_target_pose),
            "head_target_pose": self._array(head_target_pose),
            "body_waist_q": self._array(body_waist_q),
            "head_z_offset": float(head_z_offset),
            "force_seed_from_current": bool(force_seed_from_current),
        }
        response = requests.post(
            f"{self.url}/solve_ik",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(result.get("error", "remote IK solve failed"))
        return result
