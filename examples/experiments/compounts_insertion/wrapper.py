import time

import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation

from franka_env.envs.tianji.env import TianjiEnv


class COMPONENTEnv(TianjiEnv):
    """Task wrapper for compounts insertion on the minimal Tianji backend."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False
        self._quick_regrasp_count = 0
        self._gripper_control(True)

        self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self._keyboard_listener.start()

    def _on_key_press(self, key):
        if str(key) == "Key.f1":
            self.should_regrasp = True

    def _gripper_control(self, closed: bool) -> None:
        if self.fake_env:
            self.curr_gripper_pos = 0.0 if closed else 1.0
            return

        ok = self.backend.set_gripper_closed(bool(closed))
        if ok:
            self.curr_gripper_pos = 0.0 if closed else 1.0
            self.last_gripper_act = time.time()
        else:
            print(f"[CompountsEnv] gripper command failed: {self.backend.last_error}")

    def _wait_stable(self, sec: float, reason: str = "") -> None:
        del reason
        if sec <= 0:
            return
        self._hold_current_target(float(sec))

    def _joints_deg_to_pose6(self, joints_deg: np.ndarray) -> np.ndarray:
        return self.kinematics.pose_from_right_joints_deg(
            np.asarray(joints_deg, dtype=np.float64).reshape(7)
        )

    def _pose6_to_joints_deg(self, pose: np.ndarray) -> np.ndarray | None:
        return self._solve_ik(np.asarray(pose, dtype=np.float64).reshape(6))

    # def _apply_rpy_to_pose6(self, pose: np.ndarray, rpy_delta: np.ndarray) -> np.ndarray:
    #     pose = np.asarray(pose, dtype=np.float64).copy()
    #     pose[3:] = (
    #         Rotation.from_euler("xyz", np.asarray(rpy_delta, dtype=np.float64).reshape(3))
    #         * Rotation.from_euler("xyz", pose[3:])
    #     ).as_euler("xyz")
    #     return pose

    def interpolate_move(
        self,
        target_pose: np.ndarray,
        timeout: float = 1.0,
        is_reset: bool = False,
        ease: bool = True,
    ) -> None:
        del is_reset
        target = self.clip_safety_box(np.asarray(target_pose, dtype=np.float64).reshape(6))
        self._update_currpos()
        start = self.currpos.copy()

        rate_hz = max(1.0, float(getattr(self.config, "INTERPOLATE_HZ", self.hz)))
        pos_step = max(1e-5, float(getattr(self.config, "INTERPOLATE_MAX_POS_STEP_M", 0.003)))
        rot_step = max(1e-4, float(getattr(self.config, "INTERPOLATE_MAX_ROT_STEP_RAD", 0.03)))
        max_pos_delta = float(np.max(np.abs(target[:3] - start[:3])))
        max_rot_delta = float(np.max(np.abs(target[3:] - start[3:])))
        steps = max(
            2,
            int(np.ceil(max(0.0, float(timeout)) * rate_hz)),
            int(np.ceil(max_pos_delta / pos_step)),
            int(np.ceil(max_rot_delta / rot_step)),
        )

        u = np.linspace(0.0, 1.0, steps)
        if ease:
            u = 0.5 - 0.5 * np.cos(np.pi * u)

        period = 1.0 / rate_hz
        next_tick = time.perf_counter()
        for i, ratio in enumerate(u):
            pose = start + (target - start) * ratio
            ret, _ = self._send_pos_command(
                pose,
                force_seed_from_current=(i == 0),
                reset_stats=False,
            )
            if ret != 0:
                print(f"[CompountsEnv] interpolate_move IK failed at step {i + 1}/{steps}")
                break
            next_tick += period
            sleep_dt = next_tick - time.perf_counter()
            if sleep_dt > 0:
                time.sleep(sleep_dt)

        # self._update_currpos()
        # self.cmd_pose = self.currpos.copy()
        # self.nextpos = self.currpos.copy()
        # self._ik_need_seed_refresh = True

    def interpolate_joint_move(
        self,
        right_joints_deg: np.ndarray,
        timeout: float = 2.0,
        settle: bool = True,
        settle_timeout: float | None = None,
    ) -> None:
        if self.fake_env:
            return
        target = np.asarray(right_joints_deg, dtype=np.float64).reshape(7)
        _, current_right_rad = self.backend.get_joints_rad()
        current = np.rad2deg(np.asarray(current_right_rad, dtype=np.float64).reshape(7))

        if np.max(np.abs(target - current)) < 1e-3:
            # self._update_currpos()
            # self.cmd_pose = self.currpos.copy()
            # self.nextpos = self.currpos.copy()
            # self._ik_need_seed_refresh = True
            return

        rate_hz = max(1.0, float(getattr(self.config, "INTERPOLATE_HZ", self.hz)))
        step_deg = max(1e-3, float(getattr(self.config, "INTERPOLATE_MAX_STEP_DEG", 1.2)))
        steps = max(
            2,
            int(np.ceil(max(0.0, float(timeout)) * rate_hz)),
            int(np.ceil(float(np.max(np.abs(target - current))) / step_deg)),
        )

        u = np.linspace(0.0, 1.0, steps)
        if bool(getattr(self.config, "INTERPOLATE_EASE", True)):
            u = 0.5 - 0.5 * np.cos(np.pi * u)

        period = 1.0 / rate_hz
        next_tick = time.perf_counter()
        for ratio in u:
            joints = current + (target - current) * ratio
            self._move_to_right_joints_deg(joints)
            next_tick += period
            sleep_dt = next_tick - time.perf_counter()
            if sleep_dt > 0:
                time.sleep(sleep_dt)

        if settle:
            hold_sec = (
                float(settle_timeout)
                if settle_timeout is not None
                else float(getattr(self.config, "INTERPOLATE_SETTLE_TIMEOUT", 0.2))
            )
            self._move_to_right_joints_deg(target, hold_sec=max(0.0, hold_sec))

        # self._update_currpos()
        # self.cmd_pose = self.currpos.copy()
        # self.nextpos = self.currpos.copy()
        # self._ik_need_seed_refresh = True

    def interpolate_joint_waypoints(
        self,
        waypoints_deg: list[np.ndarray],
        timeout: float | None = None,
        settle_final: bool = True,
        settle_timeout: float | None = None,
    ) -> None:
        waypoints = [
            np.asarray(wp, dtype=np.float64).reshape(7)
            for wp in (waypoints_deg or [])
        ]
        if not waypoints:
            return

        total_timeout = float(timeout) if timeout is not None else 2.0 * len(waypoints)
        per_segment_timeout = max(0.2, total_timeout / len(waypoints))
        for i, waypoint in enumerate(waypoints):
            self.interpolate_joint_move(
                waypoint,
                timeout=per_segment_timeout,
                settle=settle_final and i == len(waypoints) - 1,
                settle_timeout=settle_timeout,
            )

    def _randomized_reset_pose(self, base_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(base_pose, dtype=np.float64).copy()
        pose[0] += np.random.uniform(
            float(getattr(self.config, "RANDOM_DX_MIN", -self.random_xy_range)),
            float(getattr(self.config, "RANDOM_DX_MAX", self.random_xy_range)),
        )
        pose[1] += np.random.uniform(
            float(getattr(self.config, "RANDOM_DY_MIN", -self.random_xy_range)),
            float(getattr(self.config, "RANDOM_DY_MAX", self.random_xy_range)),
        )
        pose[2] += np.random.uniform(
            float(getattr(self.config, "RANDOM_DZ_MIN", 0.0)),
            float(getattr(self.config, "RANDOM_DZ_MAX", 0.0)),
        )
        pose[5] += np.random.uniform(-self.random_rz_range, self.random_rz_range)
        return self.clip_safety_box(pose)

    def go_to_reset(self, joint_reset=False, replay_start_pose=None, **kwargs) -> None:
        
        
        # 张开夹爪
        self._gripper_control(False)
        # 去TOP->TARGET
        waypoints = []
        if hasattr(self.config, "TOP_JOINTS"):
            waypoints.append(np.asarray(self.config.TOP_JOINTS, dtype=np.float64))
        if hasattr(self.config, "TARGET_JOINTS"):
            waypoints.append(np.asarray(self.config.TARGET_JOINTS, dtype=np.float64))
      
        if waypoints:
            print("[自动复位] 关节插值移动到初始待命点...")
            self.interpolate_joint_waypoints(
                waypoints,
                timeout=float(getattr(self.config, "RESET_CHAINED_TIMEOUT", 3.0)),
                settle_final=not self.randomreset,
                settle_timeout=float(getattr(self.config, "RESET_FINAL_SETTLE_TIMEOUT", 0.2)),
            )
            # time.sleep(2.0)
        else:
            self.interpolate_move(self.resetpos, timeout=2.0, is_reset=True)
        # 闭合夹爪
        self._gripper_control(True)
        # 抬起TOP->RESET
        waypoints = []
        if hasattr(self.config, "TOP_JOINTS"):
            waypoints.append(np.asarray(self.config.TOP_JOINTS, dtype=np.float64))
        if hasattr(self.config, "RESET_JOINTS"):
            waypoints.append(np.asarray(self.config.RESET_JOINTS, dtype=np.float64))

        if waypoints:
            print("[自动复位] 关节插值移动到初始待命点...")
            self.interpolate_joint_waypoints(
                waypoints,
                timeout=float(getattr(self.config, "RESET_CHAINED_TIMEOUT", 3.0)),
                settle_final=not self.randomreset,
                settle_timeout=float(getattr(self.config, "RESET_FINAL_SETTLE_TIMEOUT", 0.2)),
            )
            # time.sleep(2.0)
        else:
            self.interpolate_move(self.resetpos, timeout=2.0, is_reset=True)

        base_pose = self.resetpos.copy()
        if self.randomreset:
            target_pose = base_pose.copy()
            target_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            target_pose[3:] = base_pose[3:].copy()
            target_pose[5] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
        else:
            target_pose = base_pose.copy()

        print(f"[TianjiEnv] reset target pose {np.round(target_pose, 4).tolist()}")
        ret, ik_error = self._send_pos_command(
            target_pose,
            force_seed_from_current=True,
            reset_stats=True,
        )
        while ret != 0 and ik_error > 0.001:
            ret, ik_error = self._send_pos_command(
                target_pose,
                force_seed_from_current=False,
                reset_stats=True,
            )
        time.sleep(float(self.config.reset_wait_sec))
        # time.sleep(5.0)
        self.cmd_pose = target_pose.copy()
        self.nextpos = target_pose.copy()        
    
    def regrasp(self) -> None:
        self._update_currpos()
        lift_pose = self.currpos.copy()
        lift_pose[2] += float(getattr(self.config, "REGRASP_LIFT_M", 0.07))
        self.interpolate_move(lift_pose, timeout=1.0, is_reset=True)

        input("Press enter to release gripper...")
        self._gripper_control(False)
        input("Place object in holder and press enter to grasp...")

        if hasattr(self.config, "TARGET_JOINTS"):
            self.interpolate_joint_move(np.asarray(self.config.TARGET_JOINTS, dtype=np.float64))
        self._gripper_control(True)
        self._wait_stable(1.0)

    def quick_regrasp(self, lift_after_grasp: bool = True) -> None:
        motion_scale = (
            float(getattr(self.config, "FIRST_ROUND_MOTION_TIMEOUT_SCALE", 1.0))
            if self._quick_regrasp_count == 0
            else 1.0
        )

        print("[自动复位] 张开夹爪...")
        self._gripper_control(False)
        self._wait_stable(float(getattr(self.config, "GRIPPER_OPEN_WAIT_SEC", 1.0)))

        if hasattr(self.config, "TOP_JOINTS"):
            print("[自动复位] 到安全点...")
            self.interpolate_joint_move(
                np.asarray(self.config.TOP_JOINTS, dtype=np.float64),
                timeout=1.5 * motion_scale,
                settle=False,
            )

        if hasattr(self.config, "TARGET_JOINTS"):
            print("[自动复位] 到抓取点...")
            if bool(getattr(self.config, "LINEAR_DROP_TOP_TO_TARGET", True)):
                target_pose = self._joints_deg_to_pose6(
                    np.asarray(self.config.TARGET_JOINTS, dtype=np.float64)
                )
                self.interpolate_move(
                    target_pose,
                    timeout=float(getattr(self.config, "LINEAR_DROP_TIMEOUT", 1.0)) * motion_scale,
                    is_reset=True,
                    ease=False,
                )
            else:
                self.interpolate_joint_move(
                    np.asarray(self.config.TARGET_JOINTS, dtype=np.float64),
                    timeout=1.5 * motion_scale,
                    settle=False,
                )

        print("[自动复位] 闭合夹爪...")
        self._gripper_control(True)
        self._wait_stable(1.0)

        # if lift_after_grasp and hasattr(self.config, "TOP_JOINTS"):
        #     print("[自动复位] 抓取后提起...")
        #     self.interpolate_joint_move(
        #         np.asarray(self.config.TOP_JOINTS, dtype=np.float64),
        #         timeout=1.5 * motion_scale,
        #         settle=False,
        #     )
        # self._quick_regrasp_count += 1

    def reset(self, joint_reset=False, replay_start_pose=None, **kwargs):
        options = kwargs.get("options") or {}
        skip_regrasp = bool(options.get("skip_regrasp", False))

        if self.save_video:
            self.save_video_recording()
        self.last_gripper_act = time.time()

        # if self.should_regrasp:
        #     self.regrasp()
        #     self.should_regrasp = False
        # elif bool(getattr(self.config, "AUTO_QUICK_REGRASP", True)) and not skip_regrasp:
        # self.quick_regrasp(lift_after_grasp=True)

        self.go_to_reset(joint_reset=joint_reset, replay_start_pose=replay_start_pose)
        self._update_currpos()
        self.curr_path_length = 0
        # self.cmd_pose = self.currpos.copy()
        # self.nextpos = self.currpos.copy()
        # self._ik_need_seed_refresh = True
        self.terminate = False
        self.max_distance = None
        return self._get_obs(), {"succeed": False}
