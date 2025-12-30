import argparse
import time
import json
import threading
import sys
import random
import numpy as np
import cv2
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
        # 线程控制
        self.is_recording = False
        self.stop_event = threading.Event()
        self.current_gripper_state = 0.0 
        
        # 暂停控制
        self.pause_requested = False
        self.input_thread = None
        self.up_z_fixed_value = 100

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- 关键位姿定义 (单位: 毫米/弧度) ---
        # 1. Home 点 (高处，空闲等待位)
        self.pos_home = [704.702393, -141.064682, 100-51.845684, 3.112087, -0.010487, 1.071136]
        
        # 2. A 点 (初始取物点，桌面上)
        self.pos_A = [704.702393, -141.064682, -51.845684, 3.112087, -0.010487, 1.071136]
        
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
        self.offset_y_range_positive = 200.0  
        self.offset_x_range_negative = -50.0  
        self.offset_y_range_negative = -50.0  

        # 运动速度
        self.speed_fast = 200 # 复位用
        self.speed_record = 100 # 录制用

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
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 1.0 
        print("Gripper Closed.")

    def open_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 0.0 
        print("Gripper Opened.")

    def connect_cameras(self):
        for name, idx in self.camera_indices.items():
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened(): self.caps[name] = cap
        print(f"Connected to {len(self.caps)} cameras.")

    def move_to(self, pos, speed=None, wait=True):
        """封装运动指令，方便调用"""
        if speed is None: speed = self.speed_fast
        self.arm.set_position(x=pos[0], y=pos[1], z=pos[2], 
                              roll=pos[3], pitch=pos[4], yaw=pos[5], 
                              speed=speed, wait=wait, is_radian=True)

    def sample_random_target(self):
        """
        生成随机的 C/E 点
        逻辑：基于 Home 点的 X,Y 坐标，增加 0~50mm 的随机正向偏移
        """
        # 1. 获取基准点 (Home 点的 X 和 Y)
        base_x = self.pos_home[0]
        base_y = self.pos_home[1]
        
        # 2. 生成随机偏移量 (0 到 50)
        # offset_x = random.uniform(0, self.offset_range)
        # offset_y = random.uniform(0, self.offset_range)
        offset_x = random.uniform(self.offset_x_range_negative, self.offset_x_range_positive)
        offset_y = random.uniform(self.offset_y_range_negative, self.offset_y_range_positive)
        
        # 3. 计算目标坐标
        # 只能比当前大 -> base + offset
        target_x = base_x + offset_x
        target_y = base_y + offset_y
        
        print(f"[Sample] Home:({base_x:.1f}, {base_y:.1f}) -> "
              f"Offset:({offset_x:.1f}, {offset_y:.1f}) -> "
              f"Target:({target_x:.1f}, {target_y:.1f})")

        # 4. 组合最终位姿: [随机X, 随机Y, 固定Z, 固定RPY]
        return [target_x, target_y, self.fixed_z] + self.fixed_rpy

    # ---------------------------------------------------------
    # 录制线程 (你的原始逻辑，保持不变)
    # ---------------------------------------------------------
    def _recording_thread(self, episode_dir, instruction):
        state_file = episode_dir / "data.jsonl"
        frame_idx = 0
        # target_interval = 0.1 # 10Hz
        target_interval = 0.2 # 5Hz
        
        print(f">>> [REC] 开始录制: {episode_dir.name}")
        
        # 获取初始状态用于计算相对值
        code, last_joint_pos = self.arm.get_servo_angle(is_radian=True)
        ret, last_cart_pos = self.arm.get_position(is_radian=True)

        if code != 0: last_joint_pos = [0.0]*7

        with open(state_file, "w") as f:
            while not self.stop_event.is_set():
                loop_start = time.time()
                try:
                    code, curr_joint_pos = self.arm.get_servo_angle(is_radian=True)
                    ret, curr_cart_pos = self.arm.get_position(is_radian=True)
                    
                    if ret != 0 or code != 0:
                        time.sleep(0.005); continue
                    
                    # 计算相对值
                    delta_joints = np.array(curr_joint_pos) - np.array(last_joint_pos)
                    delta_joints = np.round(delta_joints, 6).tolist()
                    
                    for cam_name, cap in self.caps.items():
                        ret_cap, frame = cap.read()
                        if ret_cap:
                            # 检查是否有该相机的裁剪配置
                            if cam_name in self.crop_configs and self.crop_configs[cam_name] is not None:
                                x, y, w, h = self.crop_configs[cam_name]
                                # numpy slice: [y:y+h, x:x+w]
                                frame = frame[y:y+h, x:x+w]
                                if frame_idx == 0:
                                    print(f"  - [SUCCESS] Cropped Shape: {frame.shape}")
                            
                            img_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                            cv2.imwrite(str(img_path), frame)
                    
                    # 存图
                    # for cam_name, cap in self.caps.items():
                    #     ret_cap, frame = cap.read()
                    #     if ret_cap:
                    #         img_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                    #         cv2.imwrite(str(img_path), frame)
                    #     else:
                    #         print(f"[Warn] Camera {cam_name} read failed.")


                    # 存数据
                    data = {
                        "timestamp": time.time(),
                        "frame_idx": frame_idx,
                        "instruction": instruction,
                        "joint_pos": delta_joints,      # 相对值
                        "joint_abs": curr_joint_pos,    # 绝对值
                        "cartesian_pos": curr_cart_pos, # 笛卡尔这里还是存绝对比较好，如果需要相对可以后期算
                        "gripper_state": self.current_gripper_state
                    }
                    f.write(json.dumps(data) + "\n")
                    
                    last_joint_pos = curr_joint_pos
                    
                    if frame_idx % 10 == 0:
                        sys.stdout.write(f"\r[Recording] Frame: {frame_idx}")
                        sys.stdout.flush()
                        
                    frame_idx += 1
                    
                    elapsed = time.time() - loop_start
                    time.sleep(max(0, target_interval - elapsed))
                    
                except Exception as e:
                    print(f"Record error: {e}")
                    break
        print(f">>> [REC] 录制结束: {episode_dir.name} (Frames: {frame_idx})")

    # ---------------------------------------------------------
    # 核心业务逻辑
    # ---------------------------------------------------------
    
    def start_recording(self, episode_idx, instruction):
        """启动录制线程"""
        episode_dir = self.output_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(exist_ok=True)
        for cam_name in self.caps.keys():
            (episode_dir / "images" / cam_name).mkdir(parents=True, exist_ok=True)
            
        self.stop_event.clear()
        self.rec_thread = threading.Thread(
            target=self._recording_thread, 
            args=(episode_dir, instruction)
        )
        self.rec_thread.start()

    def stop_recording(self):
        """停止录制线程"""
        if hasattr(self, 'rec_thread') and self.rec_thread.is_alive():
            self.stop_event.set()
            self.rec_thread.join()

    def input_listener(self):
        """后台监听键盘输入，用于暂停"""
        while True:
            cmd = input() # 阻塞等待输入
            if cmd.strip().lower() == 'p':
                self.pause_requested = True
                print("\n>>> [指令收到] 将在当前循环结束后暂停...")

    def run_auto_collection(self):
        self.connect_robot()
        self.connect_cameras()
        
        # 启动键盘监听线程
        threading.Thread(target=self.input_listener, daemon=True).start()

        print("\n" + "="*60)
        print("自动数据采集程序启动")
        print("流程: 初始抓取 -> [放物体 -> 回Home -> 录制抓取 -> 回Home] -> 循环")
        print("操作: 在终端输入 'p' 并回车，可在本轮结束后暂停")
        print("="*60 + "\n")

        # === 阶段 1: 初始化 (Initial Pickup) ===
        # 逻辑：机械臂去 A 点，把物体抓起来，然后回到 Home
        print(">>> 正在进行初始抓取 (Initial Pickup from A)...")
        self.open_gripper()
        self.move_to(self.pos_home, speed=self.speed_fast)
        print("wait 2s to wait for xarm move to home point")
        time.sleep(2.0)
        self.move_to(self.pos_A, speed=self.speed_fast)
        print("wait 2s to wait for xarm move to A")
        time.sleep(2.0)
        
        self.close_gripper()
        time.sleep(2.0)
        self.move_to(self.pos_home, speed=self.speed_fast)
        print(">>> 初始抓取完成，物体在 Home 点。\n")

        episode_idx = 0
        
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
                self.move_to(target_pos, speed=self.speed_fast)
                # 松开
                self.open_gripper()
                time.sleep(1.0)
                # C -> D (上抬) -> Home
                self.move_to(safe_pos_up, speed=self.speed_fast)
                self.move_to(self.pos_home, speed=self.speed_fast)
                
                # 此时：物体在 C 点，机械臂在 Home 点，夹爪是开的
                
                # -------------------------------------------------
                # 步骤 B: 执行任务 (Execute Task) - [录制]
                # -------------------------------------------------
                instruction = "pickup"
                print(f"[Record] 开始录制 Episode {episode_idx}")
                
                # <--- 开始录制 --->
                time.sleep(1.0)
                self.start_recording(episode_idx, instruction)
                # Home - > D (上抬) -> C
                self.move_to(safe_pos_up, speed=self.speed_record)
                # Home -> C
                self.move_to(target_pos, speed=self.speed_record)
                
                # 闭合夹爪 (这个动作也要录进去)
                self.close_gripper()
                time.sleep(1.0) # 稍微等一下确保夹紧
                
                # <--- 停止录制 --->
                time.sleep(1.0) # 稍微等一下确保夹紧
                self.stop_recording()
                print(f"[Record] 录制完成")

                # -------------------------------------------------
                # 步骤 C: 复位 (Return Home) - [不录制]
                # -------------------------------------------------
                print("[Reset] 抓着物体返回 Home")
                # 抬高 -> Home
                self.move_to(safe_pos_up, speed=self.speed_fast)
                self.move_to(self.pos_home, speed=self.speed_fast)

                # 循环结束，准备下一轮
                episode_idx += 1
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n强制停止...")
            self.stop_recording()
            self.arm.set_state(4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.232", help="xArm IP") # 记得改默认IP
    parser.add_argument("--output", type=str, default="data_autoPutAndRecord/raw", help="Output directory")
    args = parser.parse_args()
    
    cameras = {
        "cam_high": 4,         # 根据实际情况修改
        "cam_left_wrist": 0,
        "cam_right_wrist": 2
    }
    # my_crop_configs = {
    #     'cam_high': [100, 50, 400, 300], # [x, y, w, h]
    #     'cam_left_wrist': [20, 0, 300, 300],
    #     'cam_right_wrist': None, # None 表示不裁剪
    # }
    my_crop_configs = {
            'cam_high': [61, 1, 434, 479],  # [x, y, w, h]
            'cam_left_wrist': [118, 60, 357, 420],  # [x, y, w, h]
            'cam_right_wrist': [136, 57, 349, 412],  # [x, y, w, h]
        }
    
    recorder = AutoDataRecorder(args.ip, args.output, cameras, crop_configs=my_crop_configs)
    recorder.run_auto_collection()