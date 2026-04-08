import argparse
import time
import json
import os
import cv2
import numpy as np
import sys
import threading  # 【新增】引入多线程
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
        
        # 录制控制标志位
        self.is_recording = False
        self.stop_event = threading.Event()
        
        # 当前夹爪状态 (用于多线程共享)
        self.current_gripper_state = 0.0 
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 【配置】初始位姿 (关节角)
        self.home_joints = [0.0, -0.26, 0.0, 0.52, 0.0, 1.57, 0.0] # 请修改为你的实际Home点

        self.window_name = "Data Collection Monitor"
        
    def connect_robot(self):
        if XArmAPI is None: raise ImportError("xArm SDK Missing")
        print(f"Connecting to xArm at {self.ip}...")
        self.arm = XArmAPI(self.ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        # 自动脚本通常使用 Mode 0 (位置规划模式)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        print("xArm connected (Mode 0).")

    def close_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 1.0 # 更新状态供录制线程使用
        print("Gripper Closed.")

    def open_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 0.0 # 更新状态供录制线程使用
        print("Gripper Opened.")

    def connect_cameras(self):
        print("Connecting to cameras...")
        for name, idx in self.camera_indices.items():
            cap = cv2.VideoCapture(idx)
            # 设置缓冲区为1，减少延迟
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened(): self.caps[name] = cap
        print(f"Connected to {len(self.caps)} cameras.")
        
        # print("Initializing Monitor Window...")
        # cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)


    def reset_robot(self):
        print("\n>>> 正在复位机械臂到初始位姿...")
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        
        # 1. 切换到 Mode 0 (位置规划模式) 才能进行关节运动
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        target_pos = [703.073364, -137.630966, 3.215134, 3.125138, 0.021918, 1.124914]
        self.arm.set_position(x=target_pos[0], y=target_pos[1], z=target_pos[2], 
                              roll=target_pos[3], pitch=target_pos[4], yaw=target_pos[5], 
                              speed=60, wait=True, is_radian=True)
        
        # 2. 移动到 home_joints (wait=True 表示阻塞直到运动完成)
        # speed 单位是 rad/s，0.35 比较安全
        # self.arm.set_servo_angle(angle=self.home_joints, speed=0.35, is_radian=True, wait=True)
        
        # 3. 复位夹爪 (张开)
        self.open_gripper()
        time.sleep(0.5)
        print(">>> 复位完成")

    # ---------------------------------------------------------
    # 【核心逻辑1】后台录制线程
    # 这个函数会在单独的线程中运行，负责以固定频率保存数据
    # ---------------------------------------------------------
    def _recording_thread(self, episode_dir, instruction):
        state_file = episode_dir / "data.jsonl"
        frame_idx = 0
        
        print(">>> 后台录制线程启动...")
        
        with open(state_file, "w") as f:
            while not self.stop_event.is_set():
                loop_start = time.time()
                
                try:
                    # 1. 获取机械臂状态
                    ret, pos = self.arm.get_position(is_radian=True)
                    code, joint_pos = self.arm.get_servo_angle()
                    joint_pos = joint_pos if code == 0 else []
                    
                    if ret != 0 or code != 0:
                        continue

                    # 2. 获取图像
                    images = {}
                    for cam_name, cap in self.caps.items():
                        ret_cap, frame = cap.read()
                        if ret_cap:
                            # 只有成功读取才保存
                            img_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                            cv2.imwrite(str(img_path), frame)
                        else:
                            print(f"[Warn] Camera {cam_name} read failed.")

                    # 3. 构造数据
                    # 注意：自动化脚本中，spacemouse_action 设为全0，或者你可以计算当前位置和上一帧位置的差值
                    data = {
                        "timestamp": time.time(),
                        "frame_idx": frame_idx,
                        "instruction": instruction,
                        "joint_pos": joint_pos,
                        "cartesian_pos": pos,
                        "spacemouse_action": [0.0] * 6, # 自动模式下无手柄输入
                        "buttons": [0, 0],
                        "gripper_state": self.current_gripper_state
                    }
                    f.write(json.dumps(data) + "\n")
                    
                    if frame_idx % 10 == 0:
                        sys.stdout.write(f"\r[Recording] Frame: {frame_idx}")
                        sys.stdout.flush()

                    frame_idx += 1
                    
                    # 4. 控制帧率 (例如 10Hz)
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, 0.1 - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"\n[Error in recording thread]: {e}")
                    break
        
        print("\n>>> 后台录制线程结束。")

    def _recording_thread_framecontrol(self, episode_dir, instruction):
        state_file = episode_dir / "data.jsonl"
        frame_idx = 0
        
        TARGET_FPS = 10
        target_interval = 1.0 / TARGET_FPS
        next_frame_time = time.time()
        
        print(">>> 后台录制线程启动 (10Hz)...")
        
        # 【修改2】在子线程开始时创建窗口
        # 这样窗口的生命周期就完全属于这个线程，避免死锁
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        with open(state_file, "w") as f:
            while not self.stop_event.is_set():
                # 1. 采集数据
                # 【优化】为了防止 get_position 和主线程 set_position 冲突导致卡顿，
                # 这里可以加一个简单的异常捕获或超时重试，但通常 xArm SDK 能处理
                try:
                    ret, pos = self.arm.get_position(is_radian=True)
                    code, joint_pos = self.arm.get_servo_angle()
                    joint_pos = joint_pos if code == 0 else []
                except Exception as e:
                    print(f"[Warn] SDK Error: {e}")
                    continue
                
                if ret != 0 or code != 0: 
                    # 短暂休眠防止死循环刷屏
                    time.sleep(0.01)
                    continue

                images_to_show = []
                display_order = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                
                for cam_name in display_order:
                    if cam_name in self.caps:
                        cap = self.caps[cam_name]
                        ret_cap, frame = cap.read()
                        if ret_cap:
                            # 保存
                            img_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                            cv2.imwrite(str(img_path), frame)
                            
                            # 显示预览 (缩放)
                            preview_img = cv2.resize(frame, (320, 240))
                            cv2.putText(preview_img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            images_to_show.append(preview_img)
                        else:
                            images_to_show.append(np.zeros((240, 320, 3), dtype=np.uint8))
                    else:
                        images_to_show.append(np.zeros((240, 320, 3), dtype=np.uint8))

                # 2. 写入 JSONL
                data = {
                    "timestamp": time.time(),
                    "frame_idx": frame_idx,
                    "instruction": instruction,
                    "joint_pos": joint_pos,
                    "cartesian_pos": pos,
                    "spacemouse_action": [0.0] * 6,
                    "buttons": [0, 0],
                    "gripper_state": self.current_gripper_state
                }
                f.write(json.dumps(data) + "\n")
                
                # 3. 更新显示
                if images_to_show:
                    combined_img = np.hstack(images_to_show)
                    cv2.imshow(self.window_name, combined_img)
                    # waitKey 在此线程中调用，处理此线程创建的窗口事件
                    cv2.waitKey(1)

                if frame_idx % 10 == 0:
                    sys.stdout.write(f"\r[Recording] Frame: {frame_idx}")
                    sys.stdout.flush()

                frame_idx += 1
                
                # 4. 频率控制
                next_frame_time += target_interval
                sleep_time = next_frame_time - time.time()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.time()
        
        # 【修改3】线程结束时销毁窗口
        # 这会导致窗口在每个 Episode 之间闪烁一次（关闭再打开），但这能保证稳定性
        cv2.destroyAllWindows()
        print("\n>>> 后台录制线程结束。")
    # ---------------------------------------------------------
    # 【核心逻辑2】执行脚本化运动序列
    # 这里定义机械臂具体的动作流程
    # ---------------------------------------------------------
    def execute_sequence(self):
        # 示例坐标，请替换为你实际记录的坐标点 [x, y, z, roll, pitch, yaw]
        # 单位：mm 和 rad
        
        # 1. 目标位置 A
        target_pos_A = [703.073364, -137.630966, -50+3.215134, 3.125138, 0.021918, 1.124914]
        # 2. 目标位置 B (闭合后去的点)
        target_pos_B = [703.073364, -137.630966, 3.215134, 3.125138, 0.021918, 1.124914]
        # 3. 目标位置 C
        target_pos_C = [716.396301, 16.379723, 3.061427, 3.125071, 0.022338, 1.340843]
        # 4. 目标位置 D (松开前去的点)
        target_pos_D = [716.396301, 16.379723, -50+3.061427, 3.125071, 0.022338, 1.340843]
        # 5. 目标位置 E
        target_pos_E = [716.396301, 16.379723, 3.061427, 3.125071, 0.022338, 1.340843]

        speed = 100 # mm/s
        
        time.sleep(1.0)
        print(">>> 移动到位置 A...")
        self.arm.set_position(*target_pos_A, speed=speed, wait=True, is_radian=True)
        time.sleep(1.0) # 停顿一下

        print(">>> 闭合夹爪...")
        self.close_gripper()
        time.sleep(2.0) # 等待夹紧

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
        # time.sleep(2.0)

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
            # target=self._recording_thread_framecontrol,
            args=(episode_dir, instruction)
        )
        rec_thread.start()

        try:
            # 3. 执行阻塞式运动序列 (主线程)
            self.execute_sequence()
            
            # 4. 序列执行完，稍微等一下让数据记录完整
            time.sleep(2.0)
            
        except Exception as e:
            print(f"运动序列执行出错: {e}")
        finally:
            # 5. 停止录制
            self.stop_event.set()
            rec_thread.join() # 等待线程结束
            
            # 6. 最后再次复位 (可选，看你需求)
            self.reset_robot()
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
            
            # 这里指令可以写死，或者每次输入
            instruction = user_input.strip() if user_input.strip() else "pick and place object"
            
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
    parser.add_argument("--output", type=str, default="data_auto/raw", help="Output directory")
    args = parser.parse_args()
    
    cameras = {
        "cam_high": 4,
        "cam_left_wrist": 0,
        "cam_right_wrist": 2
    }
    
    recorder = AutoDataRecorder(args.ip, args.output, cameras)
    recorder.run()