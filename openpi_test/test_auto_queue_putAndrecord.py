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
        self.pos_home = [700.662476, -161.464981, 100-42.313469, -3.113828, 0.019474, 0.620307]
        
        # 2. A 点 (初始取物点，桌面上)
        self.pos_A = [700.662476, -161.464981, -42.313469, -3.113828, 0.019474, 0.620307]
        
        # 3. 随机采样配置
        # Z轴高度：应该与 A 点保持一致 (假设桌面是平的)
        self.fixed_z = self.pos_A[2] 
        # 姿态 (Roll/Pitch/Yaw)：保持与 A 点一致，抓取时垂直向下
        self.fixed_rpy = self.pos_A[3:]
        
        # X, Y 的采样范围 (请根据实际桌面范围修改)
        # 示例：在 A 点周围 +/- 100mm 范围内随机
        self.x_range = (70, 0) 
        self.y_range = (50, 0)
        
        self.offset_range = 50.0  
        # self.offset_range = 150.0  
        self.offset_x_range_positive = 150.0  
        self.offset_y_range_positive = 250.0  
        self.offset_x_range_negative = -150.0  
        self.offset_y_range_negative = -50.0  

         # 运动速度
        self.speed_fast = 200 # 复位用
        self.speed_record = 100 # 录制用
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
            # 即使设置了BufferSize=1，系统底层可能还是有缓存，所以需要Flush
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
        print("!!! 恢复完成")
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
         # xArm Python SDK 的 set_position 返回值通常是 code
        code = ret 
        if code != 0:
            print(f"[Error] set_position failed with code={code}. Target: {pos}")
            self.clear_robot_error() # 自动清除错误
            return False
        return True
    
    def move_to_joint_by_pose(self, target_pose, speed=None, wait=True):
        """
        关节运动 (Joint Move) - 适用于长距离移动 (Home -> Target上方)
        先计算目标位姿的 IK 关节角，然后驱动关节移动。
        """
        # 1. 计算逆运动学 (IK)
        code, joint_angles = self.arm.get_inverse_kinematics(target_pose, input_is_radian=True, return_is_radian=True)
        
        if code != 0:
            print(f"[Error] IK 计算失败，无法执行关节运动。Code={code}")
            return False
            
        # 2. 执行关节运动
        # 注意：关节速度单位不同，这里简单做个转换或者给个固定值
        # set_servo_angle 的 speed 单位是 rad/s 或 deg/s (取决于 is_radian)
        # 为了安全，这里给个适中的速度，比如 0.35 rad/s (约20度/秒) 或沿用 speed 参数(如果是同量级)
        # joint_speed = 0.5 if speed is None else (speed / 1000.0) * 2.0 # 粗略转换，或者直接写死
        joint_speed = speed
        
        ret = self.arm.set_servo_angle(angle=joint_angles, speed=joint_speed, 
                                       wait=wait, is_radian=True)
        print("关节运动速度：",joint_speed)
        if ret != 0:
            print(f"[Error] set_servo_angle failed with code={ret}")
            self.clear_robot_error()
            return False
        return True

    # 新增辅助方法：检查位姿是否可达
    # ---------------------------------------------------------
    def check_pose_reachable(self, pose):
        """
        利用逆运动学 (IK) 检查位姿是否在机械臂工作空间内
        pose: [x, y, z, roll, pitch, yaw]
        return: True (可达), False (不可达)
        """
        # get_inverse_kinematics 只计算不移动
        # input_is_radian=True, return_is_radian=True 确保单位统一
        code, _ = self.arm.get_inverse_kinematics(pose, input_is_radian=True, return_is_radian=True)
        return code == 0
    
    # def sample_random_target(self):
    #     base_x = self.pos_home[0]
    #     base_y = self.pos_home[1]
    #     # offset_x = random.uniform(0, self.offset_range)
    #     # offset_y = random.uniform(0, self.offset_range)
    #     offset_x = random.uniform(self.offset_x_range_negative, self.offset_x_range_positive)
    #     offset_y = random.uniform(self.offset_y_range_negative, self.offset_y_range_positive)

    #     target_x = base_x + offset_x
    #     target_y = base_y + offset_y
        
    #     print(f"[Sample] Home:({base_x:.1f}, {base_y:.1f}) -> "
    #           f"Offset:({offset_x:.1f}, {offset_y:.1f}) -> "
    #           f"Target:({target_x:.1f}, {target_y:.1f})")

    #     return [target_x, target_y, self.fixed_z] + self.fixed_rpy
    def sample_random_target(self):
        """
        生成随机点，并确保可达。如果不可达，则向 Home 点回缩直到可达。
        """
        base_x = self.pos_home[0]
        base_y = self.pos_home[1]
        
        # 1. 生成原始随机偏移
        offset_x = random.uniform(self.offset_x_range_negative, self.offset_x_range_positive)
        offset_y = random.uniform(self.offset_y_range_negative, self.offset_y_range_positive)

        target_x = base_x + offset_x
        target_y = base_y + offset_y
        
        # 构造目标位姿
        candidate_pose = [target_x, target_y, self.fixed_z] + self.fixed_rpy

        # 2. 检查可达性并回缩 (Fallback Logic)
        # 如果不可达，就将目标点向 Home 点移动，每次移动 10% 的距离，最多尝试 20 次
        max_retries = 20
        for i in range(max_retries):
            if self.check_pose_reachable(candidate_pose):
                if i > 0:
                    print(f"  [Warn] 原始点不可达，已回缩 {i*10}% 到安全位置。")
                
                print(f"[Sample] Valid Target: ({candidate_pose[0]:.1f}, {candidate_pose[1]:.1f})")
                return candidate_pose
            
            # --- 回缩逻辑 ---
            # 计算向量: Home -> Candidate
            vec_x = candidate_pose[0] - self.pos_home[0]
            vec_y = candidate_pose[1] - self.pos_home[1]
            
            # 缩短向量长度 (乘以 0.9)
            new_target_x = self.pos_home[0] + vec_x * 0.9
            new_target_y = self.pos_home[1] + vec_y * 0.9
            
            # 更新 candidate_pose
            candidate_pose[0] = new_target_x
            candidate_pose[1] = new_target_y
        
        # 3. 兜底方案
        # 如果缩了20次还是不行（极小概率），直接返回 Home 点，保证程序不崩
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
    # 录制线程 (包含 Buffer Flush 修复)
    # ---------------------------------------------------------
    def _recording_thread(self, episode_dir, instruction):
        frame_idx = 0
        target_interval = 0.2 # 5Hz
        
        print(f">>> [REC] 启动录制: {episode_dir.name}")
        
        writer = threading.Thread(target=self._writer_worker, args=(episode_dir,))
        writer.start()

        # >>>>>> 关键修复开始 <<<<<<
        # 在开始采集数据前，先清空相机缓冲区
        # 否则第0帧会读到几秒前（抓着物体时）的旧图
        print(">>> Flushing camera buffers...")
        for _ in range(5): # 空读5次
            for cap in self.caps.values():
                cap.grab() 
        # >>>>>> 关键修复结束 <<<<<<
        
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

                # C. 构造数据
                data_dict = {
                    "timestamp": time.time(),
                    "frame_idx": frame_idx,
                    "instruction": instruction,
                    "joint_pos": delta_joints,
                    "joint_abs": curr_joint_pos,
                    "cartesian_pos": curr_cart_pos,
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
                
                # E. 频率控制 (5Hz)
                elapsed = time.time() - loop_start
                sleep_time = max(0, target_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Record error: {e}")
                break
        
        self.data_queue.put(None)
        writer.join()
        print(f"\n>>> [REC] 录制结束。总帧数: {frame_idx}")

    # ... (其余方法保持不变) ...
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
        print("操作: 在终端输入 'p' 并回车，可在本轮结束后暂停")
        print("="*60 + "\n")

        # === 阶段 1: 初始化 (Initial Pickup) ===
        # 逻辑：机械臂去 A 点，把物体抓起来，然后回到 Home
        print(">>> 正在进行初始抓取 (Initial Pickup from A)...")
        self.open_gripper()
        self.move_to(self.pos_home, speed=self.speed_fast)
        # print("wait 2s to wait for xarm move to home point")
        # time.sleep(2.0)
        self.move_to(self.pos_A, speed=self.speed_fast)
        # print("wait 2s to wait for xarm move to A")
        # time.sleep(2.0)
        
        self.close_gripper()
        time.sleep(2.0)
        self.move_to(self.pos_home, speed=self.speed_fast)
        print(">>> 初始抓取完成，物体在 Home 点。\n")

        # episode_idx = 0
        # === 自动检测下一集的 Index ===
        existing_episodes = [
            d.name for d in self.output_dir.iterdir() 
            if d.is_dir() and d.name.startswith("episode_")
        ]
        if existing_episodes:
            # 提取所有文件夹的编号 (episode_123 -> 123)
            indices = []
            for name in existing_episodes:
                try:
                    idx = int(name.split("_")[-1])
                    indices.append(idx)
                except ValueError:
                    continue
            if indices:
                episode_idx = max(indices) + 1
            else:
                episode_idx = 0
        else:
            episode_idx = 0
            
        print(f">>> 自动检测完成，将从 Episode {episode_idx} 开始录制")
        # ============================
        # === 阶段 2: 循环采集 ===
        try:
            while True:
                # 检查暂停
                if self.pause_requested:
                    print("\n" + "!"*40)
                    print(">>> 程序已暂停 (物体在 Home 点，夹持状态)")
                    input(">>> 按回车键继续采集...")
                    print("!"*40 + "\n")
                    self.pause_requested = False # 重置标志

                print(f"\n--- Starting Loop {episode_idx} ---")
                # 1. 生成随机目标点 (C/E)
                target_pos = self.sample_random_target()
                # 目标点上方的一个安全点 D (防止移动时打翻物体)
                safe_pos_up = list(target_pos)
                safe_pos_up[2] += 100 # Z轴抬高 10cm
                
                # -------------------------------------------------
                # 步骤 A: 放置物体 (Reset Environment) - [不录制]
                # -------------------------------------------------
                print(f"[Reset] 放置物体到随机位置: X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}")
                # Home -> C
                # self.move_to(target_pos, speed=self.speed_fast)
                success = self.move_to_joint_by_pose(safe_pos_up, speed=self.joint_speed_fast) #使用关节运动！防止直线奇异点
                if not success: 
                    print("跳过本轮")
                    continue # 如果移动失败，跳过本轮
                
                self.move_to(target_pos, speed=self.speed_fast) # D -> C (Down) 使用直线运动，保持垂直
                
                self.open_gripper()
                time.sleep(2.0)
                # C -> D (上抬) -> Home
                # self.move_to(safe_pos_up, speed=self.speed_fast)
                # self.move_to(self.pos_home, speed=self.speed_fast)
                self.move_to(safe_pos_up, speed=self.speed_fast)
                self.move_to_joint_by_pose(self.pos_home, speed=self.joint_speed_fast)
                # 此时：物体在 C 点，机械臂在 Home 点，夹爪是开的
                
                # -------------------------------------------------
                # 步骤 B: 执行任务 (Execute Task) - [录制]
                # -------------------------------------------------
                # instruction = "pick up the industrial components D"
                print(f"[Record] 开始录制 Episode {episode_idx}")
                # <--- 开始录制 --->
                time.sleep(1.0)
                self.start_recording(episode_idx, "pick up the industrial components A")
                # Home - > D (上抬) -> C
                # self.move_to(safe_pos_up, speed=self.speed_record)
                self.move_to_joint_by_pose(safe_pos_up, speed=self.joint_speed_record) 
                # Home -> C
                # self.move_to(target_pos, speed=self.speed_record)
                self.move_to(target_pos, speed=self.speed_record)
                
                self.close_gripper()
                time.sleep(1.0) # 稍微等一下确保夹紧
                # <--- 停止录制 --->
                time.sleep(2.0) # 稍微等一下确保夹紧
                self.stop_recording()
                print(f"[Record] 录制完成")

                # -------------------------------------------------
                # 步骤 C: 复位 (Return Home) - [不录制]
                # -------------------------------------------------
                print("[Reset] 抓着物体返回 Home")
                # self.move_to(safe_pos_up, speed=self.speed_fast)
                # self.move_to(self.pos_home, speed=self.speed_fast)
                self.move_to(safe_pos_up, speed=self.speed_fast)
                self.move_to_joint_by_pose(self.pos_home, speed=self.joint_speed_fast)
                
                # 循环结束，准备下一轮
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