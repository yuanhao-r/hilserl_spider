import copy
import os
import time
import numpy as np
from pynput import keyboard

from franka_env.envs.xarm_env import XArmEnv
from franka_env.envs.tianji_env import TianjiEnv
from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.fisheye_capture import FisheyeCapture
from collections import OrderedDict




ROBOT_BACKEND = os.environ.get("HILSERL_ARM_BACKEND", "tianji").lower()

if ROBOT_BACKEND in {"tianji", "marvin"}:
    BaseRAMRobotEnv = TianjiEnv
else:
    BaseRAMRobotEnv = XArmEnv


# Orbbec 仅用于 wrist_1 RGB 图像源（不依赖 Gemini）
_ram_dir = os.path.dirname(os.path.abspath(__file__))

def _import_orbbec():
    """按需导入 OrbbecCapture，仅用于 wrist_1 RGB 采集。"""
    import sys
    if _ram_dir not in sys.path:
        sys.path.insert(0, _ram_dir)
    try:
        from orbbec_view import OrbbecCapture
        return OrbbecCapture
    except Exception:
        return None

# 模块加载时尝试导入一次
OrbbecCapture = _import_orbbec()

class RAMEnv(BaseRAMRobotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False
        self.auto_quick_regrasp = bool(getattr(self.config, "AUTO_QUICK_REGRASP", False))
        self._is_tianji_backend = ROBOT_BACKEND in {"tianji", "marvin"}
        self._quick_regrasp_count = 0
        self._gripper_control(True)

        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()

    def _wait_stable(self, sec: float, reason: str = ""):
        if hasattr(self, "_pause_with_hold"):
            self._pause_with_hold(sec, reason=reason)
        else:
            time.sleep(sec)

    def init_cameras(self, name_serial_dict=None):
        """wrist_1 用奥比中光（OrbbecCapture），wrist_2 用鱼眼（FisheyeCapture），供录 Demo 与 RL 采集图像。"""
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        self._orbbec_capture = None
        for cam_name, kwargs in name_serial_dict.items():
            if kwargs.get("camera_type") == "orbbec":
                orbbec_cap_cls = OrbbecCapture
                if orbbec_cap_cls is None:
                    # 按需再试一次导入（例如从 examples/ 跑 record_demos 时 path 可能不同）
                    import sys
                    for p in (_ram_dir, os.getcwd()):
                        if p and p not in sys.path:
                            sys.path.insert(0, p)
                    try:
                        from orbbec_view import OrbbecCapture as _O
                        orbbec_cap_cls = _O
                        globals()["OrbbecCapture"] = _O
                    except Exception as e:
                        raise RuntimeError(
                            "wrist_1 配置为 orbbec，但 OrbbecCapture 导入失败。请确保已安装 pyorbbecsdk： pip install pyorbbecsdk2"
                        ) from e
                if orbbec_cap_cls is None:
                    raise RuntimeError(
                        "wrist_1 配置为 orbbec，但 OrbbecCapture 导入失败。请确保已安装 pyorbbecsdk： pip install pyorbbecsdk2"
                    )
                cap_inner = orbbec_cap_cls(name=cam_name, dim=kwargs.get("dim", (1280, 720)))
                self.cap[cam_name] = VideoCapture(cap_inner)
                self._orbbec_capture = cap_inner
            elif "camera_index" in kwargs:
                self.cap[cam_name] = VideoCapture(FisheyeCapture(name=cam_name, **kwargs))
            else:
                print(f"RAMEnv: 未识别的相机配置: {cam_name}")

    def go_to_reset(
        self,
        joint_reset=False,
        replay_start_pose=None,
        from_quick_regrasp=False,
        from_target_after_grasp=False,
    ):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """     
        if self._is_tianji_backend and hasattr(self, "interpolate_joint_move"):
            replay_dwell = float(
                getattr(self.config, "RESET_REPLAY_ALIGN_DWELL_SEC", 0.0)
            )
            after_reset_dwell = float(
                getattr(self.config, "RESET_AFTER_RESET_DWELL_SEC", 0.0)
            )
            after_random_dwell = float(
                getattr(self.config, "RESET_AFTER_RANDOM_DWELL_SEC", 0.0)
            )
            if replay_start_pose is not None:
                print("[自动复位] 对齐到回放起点...")
                self.interpolate_move(replay_start_pose, timeout=1.0, is_reset=True)
                self._wait_stable(
                    replay_dwell, reason="RAM go_to_reset replay_start_pose"
                )
                return

            if joint_reset:
                print("[自动复位] 执行关节复位...")
                self._send_joint_command(np.array(self._BASIC_JOINT_RESET, dtype=np.float64))
                self._wait_stable(0.5, reason="RAM go_to_reset joint_reset")
                return

            print("[自动复位] 移动到初始待命点...")
            reset_pose = None
            top_pose = None
            if hasattr(self.config, "TOP_JOINTS") and hasattr(self, "_joints_deg_to_pose6"):
                top_pose = self.clip_safety_box(
                    np.array(
                        self._joints_deg_to_pose6(
                            np.array(self.config.TOP_JOINTS, dtype=np.float64)
                        ),
                        dtype=np.float64,
                    )
                )

            if hasattr(self.config, "RESET_JOINTS") and hasattr(self, "_joints_deg_to_pose6"):
                reset_pose = self.clip_safety_box(
                    np.array(
                        self._joints_deg_to_pose6(
                            np.array(self.config.RESET_JOINTS, dtype=np.float64)
                        ),
                        dtype=np.float64,
                    )
                )
            else:
                reset_pose = np.array(self.resetpos.copy(), dtype=np.float64)

            # 天机分支随机化：在 RESET_JOINTS 周围做笛卡尔微扰，便于 RL 数据增强
            random_pose = None
            random_timeout = float(getattr(self.config, "RANDOM_RESET_TIMEOUT", 1.0))
            if self.randomreset:
                base_pose = reset_pose.copy() if reset_pose is not None else (
                    self._RESET_POSE.copy() if hasattr(self, "_RESET_POSE") else self.currpos.copy()
                )
                random_pose = base_pose.copy()

                xy_range = float(getattr(self.config, "RANDOM_XY_RANGE", self.random_xy_range))
                x_range = float(getattr(self.config, "RANDOM_X_RANGE", xy_range))
                y_range = float(getattr(self.config, "RANDOM_Y_RANGE", xy_range))
                z_range = float(getattr(self.config, "RANDOM_Z_RANGE", 0.0))
                rz_range = float(getattr(self.config, "RANDOM_RZ_RANGE", self.random_rz_range))
                keep_ori = bool(getattr(self.config, "RANDOM_KEEP_TOOL_ORIENTATION", True))
                x_bias = float(getattr(self.config, "RANDOM_X_BIAS", 0.0))
                y_bias = float(getattr(self.config, "RANDOM_Y_BIAS", 0.0))
                z_bias = float(getattr(self.config, "RANDOM_Z_BIAS", 0.0))

                dx_min_cfg = getattr(self.config, "RANDOM_DX_MIN", None)
                dx_max_cfg = getattr(self.config, "RANDOM_DX_MAX", None)
                dy_min_cfg = getattr(self.config, "RANDOM_DY_MIN", None)
                dy_max_cfg = getattr(self.config, "RANDOM_DY_MAX", None)
                dz_min_cfg = getattr(self.config, "RANDOM_DZ_MIN", None)
                dz_max_cfg = getattr(self.config, "RANDOM_DZ_MAX", None)

                dx_min = -x_range if dx_min_cfg is None else float(dx_min_cfg)
                dx_max = x_range if dx_max_cfg is None else float(dx_max_cfg)
                dy_min = -y_range if dy_min_cfg is None else float(dy_min_cfg)
                dy_max = y_range if dy_max_cfg is None else float(dy_max_cfg)
                dz_min = -z_range if dz_min_cfg is None else float(dz_min_cfg)
                dz_max = z_range if dz_max_cfg is None else float(dz_max_cfg)

                if dx_min > dx_max:
                    dx_min, dx_max = dx_max, dx_min
                if dy_min > dy_max:
                    dy_min, dy_max = dy_max, dy_min
                if dz_min > dz_max:
                    dz_min, dz_max = dz_max, dz_min

                random_pose[0] += np.random.uniform(dx_min, dx_max) + x_bias
                random_pose[1] += np.random.uniform(dy_min, dy_max) + y_bias
                random_pose[2] += np.random.uniform(dz_min, dz_max) + z_bias
                if not keep_ori:
                    random_pose[5] += np.random.uniform(-rz_range, rz_range)

                random_pose = self.clip_safety_box(random_pose)
                delta_mm = (random_pose[:3] - base_pose[:3]) * 1000.0
                print(
                    "[自动复位] 随机化初始化位姿 "
                    f"Δx={delta_mm[0]:.1f}mm Δy={delta_mm[1]:.1f}mm Δz={delta_mm[2]:.1f}mm "
                    f"keep_ori={keep_ori}"
                )

            chained_waypoint_motion = bool(
                getattr(self.config, "RESET_CHAINED_WAYPOINT_MOTION", True)
            )
            chained_timeout = float(getattr(self.config, "RESET_CHAINED_TIMEOUT", 3.0))
            if from_target_after_grasp:
                chained_timeout *= float(
                    getattr(self.config, "RESET_CHAINED_FROM_TARGET_TIMEOUT_SCALE", 1.35)
                )
            top_revisit_threshold = float(
                getattr(self.config, "TOP_REVISIT_THRESHOLD_M", 0.004)
            )
            include_top_waypoint = top_pose is not None
            if include_top_waypoint and from_target_after_grasp:
                include_top_waypoint = True
            elif include_top_waypoint and from_quick_regrasp:
                self._update_currpos()
                include_top_waypoint = (
                    float(np.linalg.norm(self.currpos[:3] - top_pose[:3])) > top_revisit_threshold
                )
            if chained_waypoint_motion and hasattr(self, "interpolate_move_waypoints"):
                waypoints = []
                if include_top_waypoint:
                    waypoints.append(top_pose)
                if reset_pose is not None:
                    waypoints.append(reset_pose)
                if random_pose is not None:
                    waypoints.append(random_pose)
                if len(waypoints) > 0:
                    self.interpolate_move_waypoints(
                        waypoints,
                        timeout=max(0.2, chained_timeout),
                        is_reset=True,
                    )
                    # if self.randomreset:
                    #     self._wait_stable(
                    #         after_random_dwell, reason="RAM go_to_reset after random"
                    #     )
                    # else:
                    #     self._wait_stable(
                    #         after_reset_dwell, reason="RAM go_to_reset after reset"
                    #     )
                    return

            # 回退逻辑：分段执行（保留旧逻辑兜底）
            if hasattr(self.config, "RESET_JOINTS"):
                reset_continuous_mode = bool(
                    getattr(self.config, "RESET_CONTINUOUS_MODE", True)
                )
                reset_intermediate_settle = bool(
                    getattr(self.config, "RESET_INTERMEDIATE_SETTLE", False)
                )
                intermediate_settle = (
                    reset_intermediate_settle if reset_continuous_mode else True
                )
                reset_final_settle_timeout = float(
                    getattr(self.config, "RESET_FINAL_SETTLE_TIMEOUT", 0.12)
                )
                linear_top_to_reset = bool(
                    getattr(self.config, "RESET_TOP_TO_RESET_LINEAR", False)
                )
                if linear_top_to_reset and hasattr(self, "_joints_deg_to_pose6"):
                    self.interpolate_move(
                        reset_pose,
                        timeout=max(0.2, float(getattr(self.config, "RESET_TOP_TO_RESET_TIMEOUT", 2.0))),
                        is_reset=True,
                        ease=False,
                    )
                else:
                    final_settle = (not self.randomreset)
                    if hasattr(self, "interpolate_joint_waypoints"):
                        joint_waypoints = []
                        if hasattr(self.config, "TOP_JOINTS") and include_top_waypoint:
                            joint_waypoints.append(
                                np.array(self.config.TOP_JOINTS, dtype=np.float64)
                            )
                        joint_waypoints.append(
                            np.array(self.config.RESET_JOINTS, dtype=np.float64)
                        )
                        self.interpolate_joint_waypoints(
                            joint_waypoints,
                            timeout=2.6,
                            settle_final=final_settle,
                            settle_timeout=(reset_final_settle_timeout if final_settle else 0.0),
                        )
                    else:
                        self.interpolate_joint_move(
                            np.array(self.config.RESET_JOINTS, dtype=np.float64),
                            timeout=2.0,
                            settle=intermediate_settle if not final_settle else True,
                            settle_timeout=(
                                reset_final_settle_timeout
                                if final_settle
                                else 0.0
                            ),
                        )
            elif reset_pose is not None:
                self._send_pos_command(reset_pose, is_reset=True)

            if random_pose is not None:
                self.interpolate_move(
                    random_pose,
                    timeout=max(0.2, random_timeout),
                    is_reset=True,
                    ease=False,
                )
                self._wait_stable(
                    after_random_dwell, reason="RAM go_to_reset after random"
                )
            else:
                self._wait_stable(
                    after_reset_dwell, reason="RAM go_to_reset after reset"
                )
            return

        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos, is_reset=True)
        time.sleep(0.3)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = reset_pose[2] + 0.07
        self._send_pos_command(reset_pose, is_reset=True)
        # self.interpolate_move(reset_pose, timeout=0.2, is_reset=True)
        self._update_currpos()
        
        if replay_start_pose is not None:
            ret = self._send_pos_command(replay_start_pose, is_reset=True)
            time.sleep(0.5)
            return

        # perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            self._send_joint_command(self._BASIC_JOINT_RESET)
            time.sleep(0.5)
            return

        # perform Cartesian reset
        reset_pose = self.resetpos.copy()
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_random
        ret = self._send_pos_command(reset_pose, is_reset=True)
        if ret != 0:
            self._send_joint_command(self._BASIC_JOINT_RESET)
        
        time.sleep(0.5)


    def regrasp(self):
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = reset_pose[2] + 0.07
        self._send_pos_command(reset_pose, is_reset=True)
        # self.interpolate_move(reset_pose, timeout=1, is_reset=True)

        input("Press enter to release gripper...")
        self._send_gripper_command(1.0)
        input("Place RAM in holder and press enter to grasp...")
        top_pose = self._GRASP_POSE.copy()
        top_pose[2] += 0.1
        self._send_pos_command(top_pose, is_reset=True)
        time.sleep(0.5)

        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.1
        self._send_pos_command(grasp_pose, is_reset=True)

        self._send_gripper_command(-1.0)
        print("Regrasp Done!")
        self.last_gripper_act = time.time()
        time.sleep(1.0)
        
    def quick_regrasp(self, lift_after_grasp=True):
        if self._is_tianji_backend and hasattr(self, "interpolate_joint_move"):
            default_open_wait = float(getattr(self.config, "GRIPPER_OPEN_WAIT_SEC", 1.0))
            first_round_open_wait = float(
                getattr(self.config, "FIRST_ROUND_GRIPPER_OPEN_WAIT_SEC", default_open_wait)
            )
            open_wait = first_round_open_wait if self._quick_regrasp_count == 0 else default_open_wait
            first_round_timeout_scale = float(
                getattr(self.config, "FIRST_ROUND_MOTION_TIMEOUT_SCALE", 1.8)
            )
            motion_timeout_scale = (
                first_round_timeout_scale if self._quick_regrasp_count == 0 else 1.0
            )
            dwell_top = float(getattr(self.config, "QUICK_REGRASP_DWELL_TOP_SEC", 0.0))
            dwell_target = float(
                getattr(self.config, "QUICK_REGRASP_DWELL_TARGET_SEC", 0.0)
            )
            dwell_final_top = float(
                getattr(self.config, "QUICK_REGRASP_DWELL_FINAL_TOP_SEC", 0.0)
            )
            reset_continuous_mode = bool(
                getattr(self.config, "RESET_CONTINUOUS_MODE", True)
            )
            intermediate_settle = (
                bool(getattr(self.config, "RESET_INTERMEDIATE_SETTLE", False))
                if reset_continuous_mode
                else True
            )

            print("[自动复位] 张开夹爪...")
            self._gripper_control(False)
            self._wait_stable(open_wait, reason="RAM quick_regrasp open")
            # 第一轮常出现夹爪仍在完成初始化动作，补发一次开爪并短暂等待，防止未张开就下探
            if self._quick_regrasp_count == 0:
                self._gripper_control(False)
                self._wait_stable(0.3, reason="RAM quick_regrasp open reinforce")

            print("[自动复位] 向上拔出到安全点...")
            if hasattr(self.config, "TOP_JOINTS"):
                self.interpolate_joint_move(
                    np.array(self.config.TOP_JOINTS, dtype=np.float64),
                    timeout=1.5 * motion_timeout_scale,
                    settle=intermediate_settle,
                    settle_timeout=0.0,
                )
            else:
                top_pose = self._GRASP_POSE.copy()
                top_pose[2] += 0.1
                self._send_pos_command(top_pose, is_reset=True)
            self._wait_stable(dwell_top, reason="RAM quick_regrasp at TOP")

            print("[自动复位] 下降到抓取点...")
            if hasattr(self.config, "TARGET_JOINTS"):
                linear_drop = bool(getattr(self.config, "LINEAR_DROP_TOP_TO_TARGET", True))
                linear_drop_timeout = float(
                    getattr(self.config, "LINEAR_DROP_TIMEOUT", 1.5)
                )
                if linear_drop and hasattr(self, "_joints_deg_to_pose6"):
                    target_pose = self._joints_deg_to_pose6(
                        np.array(self.config.TARGET_JOINTS, dtype=np.float64)
                    )
                    self.interpolate_move(
                        np.array(target_pose, dtype=np.float64),
                        timeout=max(0.2, linear_drop_timeout * motion_timeout_scale),
                        is_reset=True,
                        ease=False,
                    )
                else:
                    self.interpolate_joint_move(
                        np.array(self.config.TARGET_JOINTS, dtype=np.float64),
                        timeout=1.5 * motion_timeout_scale,
                        settle=intermediate_settle,
                        settle_timeout=0.0,
                    )
            else:
                self._send_pos_command(self._GRASP_POSE.copy(), is_reset=True)
            self._wait_stable(dwell_target, reason="RAM quick_regrasp at TARGET")

            print("[自动复位] 闭合夹爪...")
            self._gripper_control(True)
            self.last_gripper_act = time.time()
            self._wait_stable(1.0, reason="RAM quick_regrasp close")

            print("[自动复位] 抓取完毕，提起到安全点...")
            if lift_after_grasp:
                if hasattr(self.config, "TOP_JOINTS"):
                    linear_lift = bool(getattr(self.config, "LINEAR_LIFT_TARGET_TO_TOP", True))
                    linear_timeout = float(getattr(self.config, "LINEAR_LIFT_TIMEOUT", 1.5))
                    if linear_lift and hasattr(self, "_joints_deg_to_pose6"):
                        # 改为笛卡尔直线插值：使 TARGET_JOINTS -> TOP_JOINTS 的末端路径更接近直线
                        top_pose = self._joints_deg_to_pose6(
                            np.array(self.config.TOP_JOINTS, dtype=np.float64)
                        )
                        self.interpolate_move(
                            np.array(top_pose, dtype=np.float64),
                            timeout=max(0.2, linear_timeout * motion_timeout_scale),
                            is_reset=True,
                            ease=False,
                        )
                    else:
                        self.interpolate_joint_move(
                            np.array(self.config.TOP_JOINTS, dtype=np.float64),
                            timeout=1.5 * motion_timeout_scale,
                            settle=intermediate_settle,
                            settle_timeout=0.0,
                        )
                else:
                    top_pose = self._GRASP_POSE.copy()
                    top_pose[2] += 0.1
                    self._send_pos_command(top_pose, is_reset=True)
                self._wait_stable(dwell_final_top, reason="RAM quick_regrasp final TOP")
            self._quick_regrasp_count += 1
            return

        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        
        self._gripper_control(False)
        time.sleep(1.5)

        top_pose = self._GRASP_POSE.copy()
        top_pose[2] += 0.1
        self._send_pos_command(top_pose, is_reset=True)
        time.sleep(2.0)

        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.1
        self._send_pos_command(grasp_pose, is_reset=True)

        self._gripper_control(True)
        self.last_gripper_act = time.time()
        time.sleep(1.5)

    def reset(self, joint_reset=False, replay_start_pose=None, **kwargs):
        if hasattr(self, "_enter_reset_motion_mode"):
            # self._enter_reset_motion_mode(reason="RAMEnv.reset")
        # try:
            if hasattr(self, "_stop_async_handoff_hold"):
                self._stop_async_handoff_hold()
            if hasattr(self, "_debug_log"):
                self._debug_log("RAMEnv.reset start")
            if hasattr(self, "debug_dump_recent_joint_samples"):
                self.debug_dump_recent_joint_samples(reason="ram_reset_enter", window=3)

            self.last_gripper_act = time.time()
            if self.save_video:
                self.save_video_recording()

            # if True:
            if self.should_regrasp:
                self.regrasp()
                self.should_regrasp = False
            
            did_quick_regrasp = False
            continue_from_target_after_grasp = False
            if self.auto_quick_regrasp:
                chain_from_target = bool(
                    getattr(self.config, "CHAIN_FROM_TARGET_AFTER_GRASP", True)
                ) and bool(getattr(self.config, "RESET_CHAINED_WAYPOINT_MOTION", True))
                self.quick_regrasp(lift_after_grasp=(not chain_from_target))
                did_quick_regrasp = True
                continue_from_target_after_grasp = chain_from_target

            self.go_to_reset(
                joint_reset=joint_reset,
                replay_start_pose=replay_start_pose,
                from_quick_regrasp=did_quick_regrasp,
                from_target_after_grasp=continue_from_target_after_grasp,
            )
            self.curr_path_length = 0

            if self.force_sensor is not None:
                self.force_sensor.reset_baseline()
            self._update_currpos()
            # if hasattr(self, "_ensure_safety_box_contains_key_poses"):
            #     self._ensure_safety_box_contains_key_poses(reason="ram_reset")
            # # 复位后把控制目标与当前位置强制对齐，避免下一拍沿旧目标跳变
            # self.cmd_pose = self.currpos.copy()
            # self.nextpos = self.currpos.copy()
            # self._wait_stable(
            #     float(getattr(self.config, "RESET_HANDOFF_HOLD_SEC", 0.10)),
            #     reason="ram reset handoff to step",
            # )
            # if hasattr(self, "_start_async_handoff_hold"):
            #     self._start_async_handoff_hold(
            #         max_sec=float(getattr(self.config, "RESET_HANDOFF_MAX_SEC", 1.0))
            #     )
            # if hasattr(self, "_mark_ik_seed_refresh"):
            #     self._mark_ik_seed_refresh("RAMEnv.reset exit")
            # if hasattr(self, "debug_dump_recent_joint_samples"):
            #     self.debug_dump_recent_joint_samples(reason="ram_reset_exit", window=3)
            obs = self._get_obs()
            self.terminate = False
            self.max_distance = None
            return obs, {}
        # finally:
        #     if hasattr(self, "_restore_control_mode_after_reset"):
                # self._restore_control_mode_after_reset(reason="RAMEnv.reset")
