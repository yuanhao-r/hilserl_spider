import argparse
import time
import json
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import queue

# Placeholder for xArm and Spacemouse imports
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    print("Error: xarm-python-sdk not found. Please install it: pip install xarm-python-sdk")
    XArmAPI = None

try:
    import pyspacemouse
except ImportError:
    print("Error: pyspacemouse not found. Please install it: pip install pyspacemouse")
    pyspacemouse = None

class DataRecorder:
    def __init__(self, ip, output_dir, camera_indices):
        self.ip = ip
        self.output_dir = Path(output_dir)
        self.camera_indices = camera_indices
        self.running = False
        self.arm = None
        self.caps = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def connect_robot(self):
        if XArmAPI is None:
            raise ImportError("xArm SDK not available")
        print(f"Connecting to xArm at {self.ip}...")
        self.arm = XArmAPI(self.ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        print("xArm connected.")

    def connect_cameras(self):
        print("Connecting to cameras...")
        for name, idx in self.camera_indices.items():
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"Warning: Could not open camera {name} at index {idx}")
            else:
                self.caps[name] = cap
        print(f"Connected to {len(self.caps)} cameras.")

    def connect_spacemouse(self):
        if pyspacemouse is None:
            raise ImportError("pyspacemouse not available")
        print("Opening Spacemouse...")
        success = pyspacemouse.open()
        if not success:
            print("Warning: Could not open Spacemouse.")
        else:
            print("Spacemouse connected.")

    def get_spacemouse_action(self):
        # Read Spacemouse state
        # This is a simplified example. You might need to adjust sensitivity and mapping.
        state = pyspacemouse.read()
        if state:
            # Map state to x, y, z, roll, pitch, yaw
            # xArm expects [x, y, z, roll, pitch, yaw] in mm and rad (or degrees depending on API)
            # Adjust scaling factors as needed
            x = state.x * 10
            y = state.y * 10
            z = state.z * 10
            roll = state.roll * 0.1
            pitch = state.pitch * 0.1
            yaw = state.yaw * 0.1
            
            # Button 0 for gripper close, Button 1 for gripper open (example)
            gripper = 0
            if state.buttons[0]:
                gripper = 1 # Close
            elif state.buttons[1]:
                gripper = -1 # Open
                
            return [x, y, z, roll, pitch, yaw], gripper
        return [0, 0, 0, 0, 0, 0], 0

    def record_episode(self, episode_idx):
        episode_dir = self.output_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(exist_ok=True)
        
        # Create directories for images
        for cam_name in self.caps.keys():
            (episode_dir / "images" / cam_name).mkdir(parents=True, exist_ok=True)
            
        state_file = episode_dir / "data.jsonl"
        
        print(f"Recording episode {episode_idx}. Press Ctrl+C to stop episode.")
        
        frame_idx = 0
        with open(state_file, "w") as f:
            try:
                while True:
                    start_time = time.time()
                    
                    # 1. Get Action from Spacemouse
                    cartesian_vel, gripper_cmd = self.get_spacemouse_action()
                    
                    # 2. Send Command to Robot
                    # Use velocity control for smooth teleop
                    # self.arm.vc_set_cartesian_velocity(cartesian_vel)
                    # Handle gripper...
                    
                    # 3. Get Robot State
                    code, output = self.arm.get_servo_angle()
                    joint_pos = output if code == 0 else []
                    
                    code, output = self.arm.get_position()
                    cartesian_pos = output if code == 0 else []
                    
                    # 4. Capture Images
                    images = {}
                    for cam_name, cap in self.caps.items():
                        ret, frame = cap.read()
                        if ret:
                            images[cam_name] = frame
                            # Save image
                            img_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                            cv2.imwrite(str(img_path), frame)
                    
                    # 5. Save State
                    data = {
                        "timestamp": time.time(),
                        "frame_idx": frame_idx,
                        "joint_pos": joint_pos,
                        "cartesian_pos": cartesian_pos,
                        "cartesian_vel_cmd": cartesian_vel,
                        "gripper_cmd": gripper_cmd
                    }
                    f.write(json.dumps(data) + "\n")
                    
                    frame_idx += 1
                    
                    # Maintain 50Hz
                    elapsed = time.time() - start_time
                    sleep_time = max(0, 0.02 - elapsed)
                    time.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                print("Episode stopped.")
                # Stop robot
                self.arm.vc_set_cartesian_velocity([0]*6)

    def run(self):
        self.connect_robot()
        self.connect_cameras()
        self.connect_spacemouse()
        
        episode_idx = 0
        while True:
            input(f"Press Enter to start recording episode {episode_idx} (or 'q' to quit)...")
            self.record_episode(episode_idx)
            episode_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.198", help="xArm IP address")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    # Define camera mapping: name -> index
    # Adjust these indices based on your system
    cameras = {
        "cam_high": 4,
        "cam_left_wrist": 0,
        "cam_right_wrist": 2
    }
    
    recorder = DataRecorder(args.ip, args.output, cameras)
    recorder.run()
