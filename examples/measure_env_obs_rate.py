#!/usr/bin/env python3
"""Measure how fast an experiment environment can produce observations."""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _inject_paths() -> None:
    examples_dir = Path(__file__).resolve().parent
    repo_root = examples_dir.parents[1]
    paths = [
        examples_dir,
        repo_root / "hilserl_spider" / "serl_robot_infra",
        repo_root / "hilserl_spider" / "serl_launcher",
        repo_root / "test_teleop" / "python",
        repo_root / "test_teleop" / "demo",
    ]
    for path in paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_inject_paths()

from experiments.mappings import CONFIG_MAPPING  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _print_stats(name: str, durations: list[float]) -> None:
    if not durations:
        print(f"{name}: no samples")
        return

    total = sum(durations)
    hz = len(durations) / total if total > 0 else float("inf")
    mean_ms = statistics.mean(durations) * 1000.0
    print(
        f"{name}: samples={len(durations)} total={total:.3f}s hz={hz:.2f} "
        f"mean={mean_ms:.2f}ms p50={_percentile(durations, 50) * 1000.0:.2f}ms "
        f"p90={_percentile(durations, 90) * 1000.0:.2f}ms "
        f"p99={_percentile(durations, 99) * 1000.0:.2f}ms "
        f"min={min(durations) * 1000.0:.2f}ms max={max(durations) * 1000.0:.2f}ms"
    )


def _make_call(env, mode: str):
    base_env = env.unwrapped
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    if mode == "get_obs":
        if not hasattr(base_env, "_get_obs"):
            raise AttributeError("env.unwrapped does not expose _get_obs()")
        return base_env._get_obs

    if mode == "obs_update":
        if not hasattr(base_env, "_update_currpos") or not hasattr(base_env, "_get_obs"):
            raise AttributeError("env.unwrapped does not expose _update_currpos()/_get_obs()")

        def call():
            base_env._update_currpos()
            return base_env._get_obs()

        return call

    if mode == "step_zero":
        def call():
            obs, reward, done, truncated, info = env.step(zero_action)
            if done or truncated:
                env.reset()
            return obs

        return call

    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure observation acquisition frequency for a configured env."
    )
    parser.add_argument("--exp_name", default="ram_insertion", choices=sorted(CONFIG_MAPPING))
    parser.add_argument(
        "--mode",
        default="obs_update",
        choices=("get_obs", "obs_update", "step_zero"),
        help=(
            "get_obs: only call base _get_obs(); "
            "obs_update: call base _update_currpos()+_get_obs(); "
            "step_zero: run full wrapped env.step(zero_action)."
        ),
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--fake_env", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--classifier", action="store_true")
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional HILSERL_ARM_BACKEND override, for example tianji or xarm.",
    )
    args = parser.parse_args()

    if args.backend:
        os.environ["HILSERL_ARM_BACKEND"] = args.backend

    config = CONFIG_MAPPING[args.exp_name]()
    env = config.get_environment(
        fake_env=args.fake_env,
        save_video=args.save_video,
        classifier=args.classifier,
    )

    try:
        print(
            f"env={args.exp_name} mode={args.mode} fake_env={args.fake_env} "
            f"action_shape={env.action_space.shape}"
        )
        print("resetting env...")
        env.reset()
        call = _make_call(env, args.mode)

        for _ in range(max(0, args.warmup)):
            call()

        durations: list[float] = []
        deadline = time.perf_counter() + max(0.0, args.duration)
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            call()
            durations.append(time.perf_counter() - start)

        _print_stats(args.mode, durations)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
