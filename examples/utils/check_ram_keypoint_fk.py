#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线检查 RAM 任务关键关节点的 FK 位姿范围。

用途：
1) 读取 examples/experiments/ram_insertion/config.py 中的关节关键点
2) 使用 Marvin_Kine 计算关节对应 FK（不连接机器人）
3) 按 TianjiEnv 的方式叠加 base 和 tool 偏置，输出 TCP pose
4) 打印关键点之间的位移差，便于核对“是否真的只差很小范围”
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation


def _find_envconfig_class(tree: ast.AST, source: str) -> ast.ClassDef | None:
    """递归找到包含 TARGET_JOINTS/RESET_JOINTS 的 EnvConfig。"""
    candidates: list[ast.ClassDef] = []

    def _walk(node: ast.AST):
        if isinstance(node, ast.ClassDef) and node.name == "EnvConfig":
            candidates.append(node)
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)
    for cls in candidates:
        names = set()
        for stmt in cls.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
        if "TARGET_JOINTS" in names or "RESET_JOINTS" in names:
            return cls
    return candidates[0] if candidates else None


def _eval_assignments_from_class(
    cls: ast.ClassDef,
    source: str,
    names: list[str],
) -> dict[str, object]:
    out: dict[str, object] = {}
    safe_globals = {"np": np, "__builtins__": {}}
    for stmt in cls.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        key = stmt.targets[0].id
        if key not in names:
            continue
        expr = ast.get_source_segment(source, stmt.value)
        if expr is None:
            continue
        try:
            out[key] = eval(expr, safe_globals, {})
        except Exception:
            # 某些表达式可能依赖外部符号，这里跳过
            pass
    return out


def _pose6_from_transform(tf: np.ndarray) -> np.ndarray:
    euler = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz")
    return np.concatenate([tf[:3, 3], euler]).astype(np.float64)


def _format_xyz_m(xyz_m: np.ndarray) -> str:
    return f"[{xyz_m[0]: .4f}, {xyz_m[1]: .4f}, {xyz_m[2]: .4f}] m"


def _format_xyz_mm(xyz_m: np.ndarray) -> str:
    xyz = xyz_m * 1000.0
    return f"[{xyz[0]: .1f}, {xyz[1]: .1f}, {xyz[2]: .1f}] mm"


def main():
    this_file = Path(__file__).resolve()
    spider_root = this_file.parents[2]  # .../hilserl_spider
    workspace_root = spider_root.parent  # .../teleop_tianji
    config_path = spider_root / "examples/experiments/ram_insertion/config.py"
    demo_path = workspace_root / "test_teleop/demo"
    kine_cfg = demo_path / "ccs_m6_40.MvKDCfg"

    print("=" * 88)
    print("RAM 关键点 FK 检查")
    print("=" * 88)
    print(f"config: {config_path}")
    print(f"kine cfg: {kine_cfg}")

    if not config_path.exists():
        raise FileNotFoundError(f"config.py 不存在: {config_path}")
    if not kine_cfg.exists():
        raise FileNotFoundError(f"运动学配置不存在: {kine_cfg}")

    source = config_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = _find_envconfig_class(tree, source)
    if cls is None:
        raise RuntimeError("未在 config.py 中找到 EnvConfig。")

    keys = [
        "TARGET_JOINTS",
        "TOP_JOINTS",
        "RESET_JOINTS",
        "GRASP_JOINTS",
        "ABS_POSE_LIMIT_LOW",
        "ABS_POSE_LIMIT_HIGH",
        "AUTO_EXPAND_ABS_POSE_LIMIT",
        "ABS_POSE_LIMIT_EXPAND_MARGIN_XYZ",
    ]
    cfg = _eval_assignments_from_class(cls, source, keys)

    # 注入 SDK 路径，仅用于 Marvin_Kine FK
    demo_str = str(demo_path)
    if demo_str not in sys.path:
        sys.path.insert(0, demo_str)
    try:
        from SDK_PYTHON.fx_kine import Marvin_Kine  # noqa: WPS433
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "导入 Marvin_Kine 失败。请确认在可用环境运行（通常是 Python 3.10 的 hilserl_tianji 环境）。"
        ) from exc

    # 初始化运动学
    try:
        kine = Marvin_Kine()
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            "加载 libKine.so 失败。若报 _PyThreadState_UncheckedGet 等符号错误，"
            "请改用 Python 3.10 环境运行该脚本。"
        ) from exc
    kine.log_switch(0)
    ini = kine.load_config(arm_type=0, config_path=str(kine_cfg))
    if not ini:
        raise RuntimeError("Marvin_Kine.load_config 失败。")
    kine.initial_kine(
        robot_type=ini["TYPE"][0],
        dh=ini["DH"][0],
        pnva=ini["PNVA"][0],
        j67=ini["BD"][0],
    )
    # 与 marvin_arm_controller / tianji_env 一致：无额外 tool（先设再移除）
    tool = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    kine.set_tool_kine(tool_mat=tool)
    kine.remove_tool_kine()

    # 与 tianji_env 默认一致的 base 和 tool z 偏置
    base_right_wxyzxyz = np.array(
        [0.707105, 0.707108, -0.000005, 0.000005, 0.319484, -0.012501, 1.127505],
        dtype=np.float64,
    )
    tool_z_offset_m = 0.2

    w, x, y, z, tx, ty, tz = base_right_wxyzxyz
    base_tf = np.eye(4, dtype=np.float64)
    base_tf[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    base_tf[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    tool_tf = np.eye(4, dtype=np.float64)
    tool_tf[2, 3] = tool_z_offset_m

    points: dict[str, np.ndarray] = {}
    for name in ("TARGET_JOINTS", "TOP_JOINTS", "RESET_JOINTS", "GRASP_JOINTS"):
        if name not in cfg:
            continue
        joints_deg = np.array(cfg[name], dtype=np.float64).reshape(-1)
        if joints_deg.shape[0] != 7:
            print(f"[跳过] {name} 形状不是7维: {joints_deg.shape}")
            continue
        fk_mm = kine.fk(joints=joints_deg.tolist())
        fk = np.array(fk_mm, dtype=np.float64)
        fk[:3, 3] *= 0.001  # mm -> m
        tcp_tf = base_tf @ fk @ tool_tf
        points[name] = _pose6_from_transform(tcp_tf)

    if not points:
        raise RuntimeError("未解析到任何关键关节点（TARGET/TOP/RESET/GRASP）。")

    print("\n关键点 TCP 位姿（与 TianjiEnv 奖励/限幅同坐标系）")
    for name, pose in points.items():
        xyz = pose[:3]
        rpy_deg = np.rad2deg(pose[3:])
        print(
            f"- {name:<13} xyz={_format_xyz_m(xyz)}  "
            f"rpy(deg)=[{rpy_deg[0]: .2f}, {rpy_deg[1]: .2f}, {rpy_deg[2]: .2f}]"
        )

    names = list(points.keys())
    print("\n关键点两两位移差（仅 xyz）")
    for a, b in combinations(names, 2):
        d = points[b][:3] - points[a][:3]
        dn = np.linalg.norm(d) * 1000.0
        print(f"- {a:>13} -> {b:<13}  Δxyz={_format_xyz_mm(d)}  |Δ|={dn: .1f} mm")

    stacked = np.vstack([p[:3] for p in points.values()])
    key_low = np.min(stacked, axis=0)
    key_high = np.max(stacked, axis=0)
    print("\n关键点 xyz 包围盒")
    print(f"- low : {_format_xyz_m(key_low)}")
    print(f"- high: {_format_xyz_m(key_high)}")

    low = cfg.get("ABS_POSE_LIMIT_LOW")
    high = cfg.get("ABS_POSE_LIMIT_HIGH")
    if low is not None and high is not None:
        low = np.array(low, dtype=np.float64).reshape(-1)
        high = np.array(high, dtype=np.float64).reshape(-1)
        if low.shape[0] >= 3 and high.shape[0] >= 3:
            print("\n配置中的 ABS_POSE_LIMIT（xyz）")
            print(f"- low : {_format_xyz_m(low[:3])}")
            print(f"- high: {_format_xyz_m(high[:3])}")

    auto_expand = bool(cfg.get("AUTO_EXPAND_ABS_POSE_LIMIT", True))
    margin = np.array(cfg.get("ABS_POSE_LIMIT_EXPAND_MARGIN_XYZ", [0.02, 0.02, 0.02]), dtype=np.float64).reshape(-1)
    if margin.size == 1:
        margin = np.full(3, float(margin.item()), dtype=np.float64)
    if margin.size != 3:
        margin = np.array([0.02, 0.02, 0.02], dtype=np.float64)
    print("\n自动扩框设置")
    print(f"- AUTO_EXPAND_ABS_POSE_LIMIT = {auto_expand}")
    print(f"- ABS_POSE_LIMIT_EXPAND_MARGIN_XYZ = {margin.tolist()} m")
    if auto_expand:
        print(
            f"- 仅按关键点推算的最小扩框: "
            f"low={_format_xyz_m(key_low - margin)}, "
            f"high={_format_xyz_m(key_high + margin)}"
        )

    print("\n提示")
    print("- 上面结果只用关节关键点 + FK 推算，不包含你运行过程中的 SpaceMouse 漂移。")
    print("- 如果运行时边界仍比这里大，通常是因为 currpos 在某次 reset/intervention 到过更远位置后被自动扩框。")


if __name__ == "__main__":
    main()
