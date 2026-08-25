#!/usr/bin/env python3
"""Print current left/right arm joint angles in config-friendly list form.

The Tlop client returns joint angles in radians; this script prints degrees.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


TLOP_DRV_DIR = Path("/home/ubuntu/teleop_tianji/tlop_drv")
if TLOP_DRV_DIR.exists():
    sys.path.insert(0, str(TLOP_DRV_DIR))

from tlop_client import ClientConfig, TlopClient  # noqa: E402


def format_list(values, precision: int) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def rad_to_deg(values):
    return [math.degrees(float(v)) for v in values]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read current left/right joint angles and print Python lists."
    )
    parser.add_argument("--host", default="10.10.12.2", help="Tlop server host")
    parser.add_argument("--port", type=int, default=17030, help="Tlop server port")
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for the Tlop client to become ready",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Decimal places for printed joint angles",
    )
    args = parser.parse_args()

    config = ClientConfig(host=args.host, port=args.port)
    with TlopClient(config) as client:
        if not client.wait_ready(args.timeout):
            raise RuntimeError(client.last_error)

        left_position_rad, right_position_rad = client.get_psn("LR")

    left = format_list(rad_to_deg(left_position_rad), args.precision)
    right = format_list(rad_to_deg(right_position_rad), args.precision)

    print("# Unit: degrees")
    print(f"LEFT_JOINTS = {left}")
    print(f"RIGHT_JOINTS = {right}")
    print(f"LR_JOINTS = [{left}, {right}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
