import argparse
import time
import json
import os
import cv2
import numpy as np
import sys
import threading
from pathlib import Path
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------
# 硬件导入
# ---------------------------------------------------------
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    print("Error: xarm-python-sdk not found.")
    XArmAPI = None

class AutoDataRecorder:
    def __init__(self, ip, output_dir, camera_indices):
        self.ip = ip
        self.output_dir = Path(output_dir)
        self.camera_indices = camera_indices
        self.arm = None
        self.caps = {}
        
        self.is_recording = False
        self.stop_event = threading.Event()
        self.current_gripper_state = 0.0 
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_name = "Data Collection Monitor"
        
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
        print("xArm connected (Mode 0).")

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
        print("Connecting to cameras...")
        for name, idx in self.camera_indices.items():
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened(): self.caps[name] = cap
        print(f"Connected to {len(self.caps)} cameras.")

    def reset_robot(self):
        print("\n>>> 正在复位机械臂到初始位姿...")
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        
        # 笛卡尔初始点
        target_pos = [703.073364, -137.630966, 3.215134, 3.125138, 0.021918, 1.124914]
        
        self.arm.set_position(x=target_pos[0], y=target_pos[1], z=target_pos[2], 
                              roll=target_pos[3], pitch=target_pos[4], yaw=target_pos[5], 
                              speed=60, wait=True, is_radian=True)
        
        self.open_gripper()
        time.sleep(0.5)
        print(">>> 复位完成")

    # ---------------------------------------------------------
    # 【核心修改】计算相对值 (Delta)
    # ---------------------------------------------------------
    def _recording_thread(self, episode_dir, instruction):
        state_file = episode_dir / "data.jsonl"
        frame_idx = 0
        target_interval = 0.1 # 10Hz
        
        print(">>> 后台录制线程启动 (Mode: Radians, Type: RELATIVE, Freq: 10Hz)...")
        
        # 1. 初始化上一帧状态 (用于计算差值)
        # 强制使用弧度
        code, last_joint_pos = self.arm.get_servo_angle(is_radian=True)
        # 获取笛卡尔坐标也建议存一下，备用
        ret, last_cart_pos = self.arm.get_position(is_radian=True)
        
        if code != 0:
            print("[Error] 初始状态获取失败，可能会导致第一帧数据异常")
            last_joint_pos = [0.0] * 7
            last_cart_pos = [0.0] * 6

        with open(state_file, "w") as f:
            while not self.stop_event.is_set():
                loop_start = time.time()
                
                try:
                    # 2. 获取当前绝对状态 (弧度)
                    code, curr_joint_pos = self.arm.get_servo_angle(is_radian=True)
                    ret, curr_cart_pos = self.arm.get_position(is_radian=True)
                    
                    if ret != 0 or code != 0:
                        time.sleep(0.005)
                        continue
                    
                    # 3. 计算相对值 (Delta = Current - Last)
                    # 使用 numpy 方便计算，保留6位小数防止浮点噪点
                    # 关节相对值
                    delta_joints = np.array(curr_joint_pos) - np.array(last_joint_pos)
                    delta_joints = np.round(delta_joints, 6).tolist()
                    
                    # 笛卡尔相对值 (可选，如果以后需要)
                    # delta_cart = np.array(curr_cart_pos) - np.array(last_cart_pos)
                    
                    # 4. 获取图像
                    images = {}
                    for cam_name, cap in self.caps.items():
                        ret_cap, frame = cap.read()
                        if ret_cap:
                            img_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                            cv2.imwrite(str(img_path), frame)
                        else:
                            print(f"[Warn] Camera {cam_name} read failed.")


                    # 5. 构造数据
                    data = {
                        "timestamp": time.time(),
                        "frame_idx": frame_idx,
                        "instruction": instruction,
                        
                        # === 修改点: joint_pos 现在存的是相对值 ===
                        "joint_pos": delta_joints, 
                        
                        # === 安全起见: 额外存一个绝对值字段，防止以后想用绝对值没得救 ===
                        "joint_abs": curr_joint_pos, 
                        
                        "cartesian_pos": curr_cart_pos, # 笛卡尔这里还是存绝对比较好，如果需要相对可以后期算
                        
                        "spacemouse_action": [0.0] * 6,
                        "buttons": [0, 0],
                        "gripper_state": self.current_gripper_state
                    }
                    f.write(json.dumps(data) + "\n")
                    
                    # 6. 更新上一帧
                    last_joint_pos = curr_joint_pos
                    last_cart_pos = curr_cart_pos
                    
                    if frame_idx % 10 == 0:
                        sys.stdout.write(f"\r[Recording] Frame: {frame_idx}")
                        sys.stdout.flush()

                    frame_idx += 1
                    
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, target_interval - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"\n[Error in recording thread]: {e}")
                    break
        
        print("\n>>> 后台录制线程结束。")

    def execute_sequence(self):
        # 确保这些动作坐标中的 RPY 是弧度！
        target_pos_A = [703.073364, -137.630966, -50+3.215134, 3.125138, 0.021918, 1.124914]
        target_pos_B = [703.073364, -137.630966, 3.215134, 3.125138, 0.021918, 1.124914]
        target_pos_C = [716.396301, 16.379723, 3.061427, 3.125071, 0.022338, 1.340843]
        target_pos_D = [716.396301, 16.379723, -50+3.061427, 3.125071, 0.022338, 1.340843]
        target_pos_E = [716.396301, 16.379723, 3.061427, 3.125071, 0.022338, 1.340843]

        speed = 100 
        
        time.sleep(1.0)
        print(">>> 移动到位置 A...")
        self.arm.set_position(*target_pos_A, speed=speed, wait=True, is_radian=True)
        time.sleep(1.0) 

        print(">>> 闭合夹爪...")
        self.close_gripper()
        time.sleep(2.0) 

        print(">>> 移动到位置 B...")
        self.arm.set_position(*target_pos_B, speed=speed, wait=True, is_radian=True)
        
        print(">>> 移动到位置 C...")
        self.arm.set_position(*target_pos_C, speed=speed, wait=True, is_radian=True)
        
        print(">>> 移动到位置 D...")
        self.arm.set_position(*target_pos_D, speed=speed, wait=True, is_radian=True)
        
        print(">>> 松开夹爪...")
        self.open_gripper()
        time.sleep(2.0)

        print(">>> 移动到位置 E...")
        self.arm.set_position(*target_pos_E, speed=speed, wait=True, is_radian=True)
        
    def record_episode(self, episode_idx, instruction):
        episode_dir = self.output_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(exist_ok=True)
        for cam_name in self.caps.keys():
            (episode_dir / "images" / cam_name).mkdir(parents=True, exist_ok=True)
        # 1. 先复位
        self.reset_robot()
        print(f"\n=== 开始录制 Episode {episode_idx} ===")
        # 2. 启动录制线程
        self.stop_event.clear()
        rec_thread = threading.Thread(
            target=self._recording_thread, 
            args=(episode_dir, instruction)
        )
        rec_thread.start()

        try:
            self.execute_sequence() # 3. 执行阻塞式运动序列 (主线程)
            time.sleep(2.0) # 4. 序列执行完，稍微等一下让数据记录完整
        except Exception as e:
            print(f"运动序列执行出错: {e}")
        finally:
            self.stop_event.set() # 5. 停止录制
            rec_thread.join() # 等待线程结束
            self.reset_robot() # 6. 最后再次复位 (可选，看需求)
            print(f"Episode {episode_idx} 完成并保存。\n")

    def run(self):
        self.connect_robot()
        self.connect_cameras()
        episode_idx = 0
        while True:
            print("-" * 50)
            user_input = input(f"[Episode {episode_idx}] 按回车开始录制 (输入 'q' 退出): ")
            if user_input.strip().lower() == 'q':
                break
            instruction = user_input.strip() if user_input.strip() else "pickup"
            try:
                self.record_episode(episode_idx, instruction)
                episode_idx += 1
            except KeyboardInterrupt:
                print("\n强制停止。")
                self.stop_event.set() # 确保线程停止
                self.arm.set_state(4) # 停止机械臂
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.198", help="xArm IP address")
    parser.add_argument("--output", type=str, default="data_auto_relative/raw", help="Output directory")
    args = parser.parse_args()
    
    cameras = {
        "cam_high": 4,
        "cam_left_wrist": 0,
        "cam_right_wrist": 2
    }
    
    recorder = AutoDataRecorder(args.ip, args.output, cameras)
    recorder.run()