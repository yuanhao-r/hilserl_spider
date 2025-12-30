import argparse
import time
import json
import threading
import sys
import random
import numpy as np
import cv2
import queue
from pathlib import Path

# ---------------------------------------------------------
# 硬件导入
# ---------------------------------------------------------
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    print("Error: xarm-python-sdk not found.")
    XArmAPI = None

class AutoDataRecorder:
    def __init__(self, ip, output_dir, camera_indices, crop_configs=None):
        self.ip = ip
        self.output_dir = Path(output_dir)
        self.camera_indices = camera_indices
        self.arm = None
        self.caps = {}
        self.crop_configs = crop_configs if crop_configs else {}
        
        self.is_recording = False
        self.stop_event = threading.Event()
        self.current_gripper_state = 0.0 
        self.pause_requested = False
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- 关键位姿定义 (单位: 毫米/弧度) ---
        # 1. Home 点 (高处，空闲等待位)can not be changed
        self.pos_home = [539.120605, 17.047951, 100-59.568863, 3.12897, 0.012689, -1.01436]
        
        # 2. A 点 (初始取物点，桌面上)
        self.pos_A = [539.120605, 17.047951, -59.568863, 3.12897, 0.012689, -1.01436]
        
        # 3. 随机采样配置
        # Z轴高度：应该与 A 点保持一致 (假设桌面是平的)
        self.fixed_z = self.pos_A[2] 
        # 固定roll和pitch，只随机化yaw（最后一个值）
        self.fixed_roll = self.pos_A[3]
        self.fixed_pitch = self.pos_A[4]
        
        # Yaw角度随机范围 (弧度) - 可根据实际需求调整
        # 示例：在基础yaw角度 ±π/4 (45度) 范围内随机
        self.base_yaw = self.pos_A[5]
        self.yaw_random_range = (-np.pi/4, np.pi/4)  # ±45度，可根据需要调整
        
        # X, Y 的采样范围
        # self.offset_x_range_positive = 150.0  
        # self.offset_y_range_positive = 250.0  
        # self.offset_x_range_negative = -150.0  
        # self.offset_y_range_negative = -50.0 
        self.offset_x_range_positive =  150.0
        self.offset_y_range_positive =  150.0
        self.offset_x_range_negative =   -150.0
        self.offset_y_range_negative =  -150.0

        self.offset_x_random_range =(self.offset_x_range_negative, self.offset_x_range_positive)
        self.offset_y_random_range =(self.offset_y_range_negative, self.offset_y_range_positive)
        
        # 存储当前轮次的随机yaw角度
        self.current_yaw_angle = self.base_yaw

        # 运动速度
        self.speed_fast = 300 # 复位用
        self.speed_record = 300 # 录制用
        self.joint_speed_fast = 0.5 # 复位用
        self.joint_speed_record = 0.15# 录制用
        
        # === 数据队列 (maxsize=50) ===
        self.data_queue = queue.Queue(maxsize=50) 

    def connect_robot(self):
        if XArmAPI is None: raise ImportError("xArm SDK Missing")
        print(f"Connecting to xArm at {self.ip}...")
        self.arm = XArmAPI(self.ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        print("xArm connected.")

    def close_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 16, 1, 2, 0, 2, 4, 0, 0, 46, 224])
        self.arm.getset_tgpio_modbus_data([0x01, 6, 1, 8, 0, 1])
        self.current_gripper_state = 1.0 

    def open_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 16, 1, 2, 0, 2, 4, 0, 0, 0, 0])
        self.arm.getset_tgpio_modbus_data([0x01, 6, 1, 8, 0, 1])
        self.current_gripper_state = 0.0 

    def connect_cameras(self):
        for name, idx in self.camera_indices.items():
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened(): self.caps[name] = cap
        print(f"Connected to {len(self.caps)} cameras.")
        
    def clear_robot_error(self):
        """当机械臂报错时，尝试恢复状态"""
        print("!!! 检测到机械臂错误，正在尝试自动恢复...")
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        time.sleep(1.0)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        self.move_to(self.pos_home, speed=self.joint_speed_fast)
        print("回到home点")
        
    def move_to(self, pos, speed=None, wait=True):
        if speed is None: speed = self.speed_fast
        ret = self.arm.set_position(x=pos[0], y=pos[1], z=pos[2], 
                                    roll=pos[3], pitch=pos[4], yaw=pos[5], 
                                    speed=speed, wait=wait, is_radian=True)
        code = ret 
        if code != 0:
            print(f"[Error] set_position failed with code={code}. Target: {pos}")
            self.clear_robot_error()
            return False
        return True
    
    def move_to_joint_by_pose(self, target_pose, speed=None, wait=True):
        """关节运动 (Joint Move) - 适用于长距离移动"""
        # 1. 计算逆运动学 (IK)
        code, joint_angles = self.arm.get_inverse_kinematics(target_pose, input_is_radian=True, return_is_radian=True)
        
        if code != 0:
            print(f"[Error] IK 计算失败，无法执行关节运动。Code={code}")
            return False
            
        # 2. 执行关节运动
        joint_speed = speed
        ret = self.arm.set_servo_angle(angle=joint_angles, speed=joint_speed, 
                                       wait=wait, is_radian=True)
        print("关节运动速度：",joint_speed)
        if ret != 0:
            print(f"[Error] set_servo_angle failed with code={ret}")
            self.clear_robot_error()
            return False
        return True

    def check_pose_reachable(self, pose):
        """检查位姿是否可达"""
        code, _ = self.arm.get_inverse_kinematics(pose, input_is_radian=True, return_is_radian=True)
        return code == 0
    
    def sample_random_target(self):
        """生成随机点（含随机yaw角度），并确保可达"""
        base_x = self.pos_home[0]
        base_y = self.pos_home[1]
        
        # 1. 生成原始随机偏移
        # offset_x = random.uniform(self.offset_x_range_negative, self.offset_x_range_positive)
        # offset_y = random.uniform(self.offset_y_range_negative, self.offset_y_range_positive)
        offset_x = random.uniform(*self.offset_x_random_range)
        offset_y = random.uniform(*self.offset_y_random_range)
        target_x = base_x + offset_x
        target_y = base_y + offset_y
        
        # 2. 随机生成yaw角度（最后一个值）并保存
        self.current_yaw_angle = self.base_yaw + random.uniform(*self.yaw_random_range)
        print(f"[Random Yaw] 生成随机Yaw角度: {np.degrees(self.current_yaw_angle):.1f}° "
              f"(基础值: {np.degrees(self.base_yaw):.1f}°，偏移: {np.degrees(self.current_yaw_angle - self.base_yaw):.1f}°)")
        
        # 构造目标位姿 (包含随机yaw角度，roll和pitch固定)
        candidate_pose = [
            target_x, 
            target_y, 
            self.fixed_z, 
            self.fixed_roll,          # 固定roll
            self.fixed_pitch,         # 固定pitch
            self.current_yaw_angle    # 随机yaw（最后一个值）
        ]

        # 3. 检查可达性并回缩
        max_retries = 20
        for i in range(max_retries):
            if self.check_pose_reachable(candidate_pose):
                if i > 0:
                    print(f"  [Warn] 原始点不可达，已回缩 {i*10}% 到安全位置。")
                
                print(f"[Sample] Valid Target: ({candidate_pose[0]:.1f}, {candidate_pose[1]:.1f}), "
                      f"Yaw: {np.degrees(candidate_pose[5]):.1f}°")
                return candidate_pose
            
            # 回缩逻辑
            vec_x = candidate_pose[0] - self.pos_home[0]
            vec_y = candidate_pose[1] - self.pos_home[1]
            
            new_target_x = self.pos_home[0] + vec_x * 0.9
            new_target_y = self.pos_home[1] + vec_y * 0.9
            
            candidate_pose[0] = new_target_x
            candidate_pose[1] = new_target_y
        
        # 兜底方案
        print("[Error] 采样点完全不可达，使用 Home 点作为 fallback。")
        return self.pos_home

    # ---------------------------------------------------------
    # 后台写入线程
    # ---------------------------------------------------------
    def _writer_worker(self, episode_dir):
        state_file = episode_dir / "data.jsonl"
        with open(state_file, "a") as f:
            while True:
                try:
                    item = self.data_queue.get(timeout=2)
                    if item is None:
                        self.data_queue.task_done()
                        break
                    
                    data_dict, images_dict = item
                    
                    # 1. 写 JSON
                    f.write(json.dumps(data_dict) + "\n")
                    
                    # 2. 写 图片
                    frame_idx = data_dict["frame_idx"]
                    for cam_name, img_data in images_dict.items():
                        save_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                        cv2.imwrite(str(save_path), img_data)
                        
                    self.data_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Writer Error] {e}")

    # ---------------------------------------------------------
    # 录制线程
    # ---------------------------------------------------------
    def _recording_thread(self, episode_dir, instruction):
        frame_idx = 0
        target_interval = 0.2 # 5Hz
        
        print(f">>> [REC] 启动录制: {episode_dir.name}")
        
        writer = threading.Thread(target=self._writer_worker, args=(episode_dir,))
        writer.start()

        # 清空相机缓冲区
        print(">>> Flushing camera buffers...")
        for _ in range(5):
            for cap in self.caps.values():
                cap.grab() 
        
        code, last_joint_pos = self.arm.get_servo_angle(is_radian=True)
        ret, last_cart_pos = self.arm.get_position(is_radian=True)
        if code != 0: last_joint_pos = [0.0]*7

        while not self.stop_event.is_set():
            loop_start = time.time()
            try:
                # A. 机器人状态
                code, curr_joint_pos = self.arm.get_servo_angle(is_radian=True)
                ret, curr_cart_pos = self.arm.get_position(is_radian=True)
                
                if ret != 0 or code != 0:
                    time.sleep(0.005); continue
                
                delta_joints = np.array(curr_joint_pos) - np.array(last_joint_pos)
                delta_joints = np.round(delta_joints, 6).tolist()
                
                # B. 图像读取与裁剪
                current_images = {}
                for cam_name, cap in self.caps.items():
                    ret_cap, frame = cap.read()
                    if ret_cap:
                        if cam_name in self.crop_configs and self.crop_configs[cam_name] is not None:
                            x, y, w, h = self.crop_configs[cam_name]
                            frame = frame[y:y+h, x:x+w]
                            if frame_idx == 0:
                                print(f"  - [SUCCESS] Cropped Shape: {frame.shape}")
                        current_images[cam_name] = frame

                # C. 构造数据（yaw角度已体现在joint_abs和cartesian_pos中）
                data_dict = {
                    "timestamp": time.time(),
                    "frame_idx": frame_idx,
                    "instruction": instruction,
                    "joint_pos": delta_joints,       # 关节角度变化量
                    "joint_abs": curr_joint_pos,     # 关节绝对角度（已包含yaw对应的关节角度）
                    "cartesian_pos": curr_cart_pos,  # 笛卡尔坐标（包含yaw角度）
                    "gripper_state": self.current_gripper_state
                }
                
                # D. 入队
                self.data_queue.put((data_dict, current_images))
                
                last_joint_pos = curr_joint_pos
                
                if frame_idx % 10 == 0:
                    q_size = self.data_queue.qsize()
                    sys.stdout.write(f"\r[REC] Frame: {frame_idx} | Q_Size: {q_size}")
                    sys.stdout.flush()
                    
                frame_idx += 1
                
                # E. 频率控制
                elapsed = time.time() - loop_start
                sleep_time = max(0, target_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Record error: {e}")
                break
        
        self.data_queue.put(None)
        writer.join()
        print(f"\n>>> [REC] 录制结束。总帧数: {frame_idx}")

    def start_recording(self, episode_idx, instruction):
        episode_dir = self.output_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(exist_ok=True)
        for cam_name in self.caps.keys():
            (episode_dir / "images" / cam_name).mkdir(parents=True, exist_ok=True)
            
        self.stop_event.clear()
        while not self.data_queue.empty(): self.data_queue.get()
            
        self.rec_thread = threading.Thread(
            target=self._recording_thread, 
            args=(episode_dir, instruction)
        )
        self.rec_thread.start()

    def stop_recording(self):
        if hasattr(self, 'rec_thread') and self.rec_thread.is_alive():
            self.stop_event.set()
            self.rec_thread.join()

    def input_listener(self):
        while True:
            cmd = input()
            if cmd.strip().lower() == 'p':
                self.pause_requested = True
                print("\n>>> [指令收到] 将在当前循环结束后暂停...")

    def run_auto_collection(self):
        self.connect_robot()
        self.connect_cameras()
        threading.Thread(target=self.input_listener, daemon=True).start()

        print("\n" + "="*60)
        print("自动数据采集程序 (5Hz, Fixed Buffer Lag)")
        print("流程: 初始抓取 -> [放物体 -> 回Home -> 录制抓取 -> 回Home] -> 循环")
        print("特性: 随机化Yaw角度（最后一个值）±45度，放置和抓取使用相同角度")
        print("说明: Yaw角度已体现在joint_abs和cartesian_pos中，无需额外记录")
        print("操作: 在终端输入 'p' 并回车，可在本轮结束后暂停")
        print("="*60 + "\n")

        # === 阶段 1: 初始化 (Initial Pickup) ===
        print(">>> 正在进行初始抓取 (Initial Pickup from A)...")
        self.open_gripper()
        self.move_to(self.pos_home, speed=self.speed_fast)
        self.move_to(self.pos_A, speed=self.speed_fast)
        
        self.close_gripper()
        time.sleep(2.0)
        self.move_to(self.pos_home, speed=self.speed_fast)
        print(">>> 初始抓取完成，物体在 Home 点。\n")

        # === 自动检测下一集的 Index ===
        existing_episodes = [
            d.name for d in self.output_dir.iterdir() 
            if d.is_dir() and d.name.startswith("episode_")
        ]
        if existing_episodes:
            indices = []
            for name in existing_episodes:
                try:
                    idx = int(name.split("_")[-1])
                    indices.append(idx)
                except ValueError:
                    continue
            episode_idx = max(indices) + 1 if indices else 0
        else:
            episode_idx = 0
            
        print(f">>> 自动检测完成，将从 Episode {episode_idx} 开始录制")
        
        # === 阶段 2: 循环采集 ===
        try:
            while True:
                # 检查暂停
                if self.pause_requested:
                    print("\n" + "!"*40)
                    print(">>> 程序已暂停 (物体在 Home 点，夹持状态)")
                    input(">>> 按回车键继续采集...")
                    print("!"*40 + "\n")
                    self.pause_requested = False

                print(f"\n--- Starting Loop {episode_idx} ---")
                # 1. 生成随机目标点 (包含随机yaw角度)
                target_pos = self.sample_random_target()
                # 目标点上方的安全点
                safe_pos_up = list(target_pos)
                safe_pos_up[2] += 100 # Z轴抬高 10cm
                safe_pos_up[5] = self.current_yaw_angle  # 保持相同yaw角度
                
                # -------------------------------------------------
                # 步骤 A: 放置物体 (Reset Environment) - [不录制]
                # -------------------------------------------------
                print(f"[Reset] 放置物体到随机位置: X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}, "
                      f"Yaw={np.degrees(self.current_yaw_angle):.1f}°")
                # success = self.move_to_joint_by_pose(safe_pos_up, speed=self.joint_speed_fast)
                success = self.move_to(safe_pos_up, speed=self.speed_fast)
                
                if not success: 
                    print("跳过本轮")
                    continue
                
                self.move_to(target_pos, speed=self.speed_fast)
                self.open_gripper()
                time.sleep(2.0)
                # 返回Home
                self.move_to(safe_pos_up, speed=self.speed_fast)
                # self.move_to_joint_by_pose(self.pos_home, speed=self.joint_speed_fast)
                self.move_to(self.pos_home, speed=self.speed_fast)
                
                # -------------------------------------------------
                # 步骤 B: 执行任务 (Execute Task) - [录制]
                # -------------------------------------------------
                print(f"[Record] 开始录制 Episode {episode_idx} (Yaw={np.degrees(self.current_yaw_angle):.1f}°)")
                time.sleep(1.0)
                self.start_recording(episode_idx, "pick up the industrial components")
                # 抓取动作
                # self.move_to_joint_by_pose(safe_pos_up, speed=self.joint_speed_record) 
                self.move_to(safe_pos_up, speed=self.speed_record) 
                self.move_to(target_pos, speed=self.speed_record)
                self.close_gripper()
                time.sleep(1.0)
                # 停止录制
                time.sleep(2.0)
                self.stop_recording()
                print(f"[Record] 录制完成 (使用相同Yaw角度: {np.degrees(self.current_yaw_angle):.1f}°)")

                # -------------------------------------------------
                # 步骤 C: 复位 (Return Home) - [不录制]
                # -------------------------------------------------
                print("[Reset] 抓着物体返回 Home")
                self.move_to(safe_pos_up, speed=self.speed_fast)
                self.move_to_joint_by_pose(self.pos_home, speed=self.speed_fast)
                
                # 循环结束
                episode_idx += 1
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n强制停止...")
            self.stop_recording()
            self.arm.set_state(4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.232", help="xArm IP")
    parser.add_argument("--output", type=str, default="data_auto_queue_PutAndRecord_1212/raw", help="Output directory")
    args = parser.parse_args()
    
    cameras = {
        "cam_high": 4,
        "cam_left_wrist": 0,
        "cam_right_wrist": 2
    }
    
    my_crop_configs = {
        'cam_high': [61, 1, 434, 479], 
        'cam_left_wrist': [118, 60, 357, 420],
        'cam_right_wrist': [136, 57, 349, 412],
    }
    
    recorder = AutoDataRecorder(args.ip, args.output, cameras, crop_configs=my_crop_configs)
    recorder.run_auto_collection()
