import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def create_dataset(repo_id, root_dir, robot_type="xarm"):
    root_dir = Path(root_dir)
    
    # === 【修改1】计算完整的最终路径 ===
    # 比如: /home/hil-serl/my_custom_data/xarm_convert
    output_path = root_dir / repo_id
    
    # 清理旧数据
    if output_path.exists():
        print(f"Cleaning up existing dataset at {output_path}")
        shutil.rmtree(output_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,), 
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
        },
        "language_instruction": {
            "dtype": "string",
            "shape": (1,),
            "names": ["instruction"],
        },
    }
    
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, 480, 640), 
            "names": ["channels", "height", "width"],
        }
        
    # === 【修改2】将完整路径传给 root ===
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,  # <--- 关键修改：直接传完整路径
        fps=10,
        robot_type=robot_type,
        features=features,
        use_videos=True, 
    )
    return dataset

def load_episode_data(episode_dir):
    data_file = episode_dir / "data.jsonl"
    with open(data_file, "r") as f:
        lines = [json.loads(line) for line in f]
        
    states = []
    instructions = []
    for line in lines:
        joint_pos = line["joint_pos"]
        gripper_state = line.get("gripper_state", 0.0)
      
        if len(joint_pos) != 7:
            pass
        if len(joint_pos) == 7:
            joint_pos[6] = gripper_state
        elif len(joint_pos) == 6:
            joint_pos.append(gripper_state)  
        states.append(joint_pos)
        instructions.append(line.get("instruction", ""))
        
    states = np.array(states, dtype=np.float32)
    
    images = {}
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for cam in cameras:
        cam_dir = episode_dir / "images" / cam
        if not cam_dir.exists():
            continue
            
        img_files = sorted(cam_dir.glob("*.jpg"))
        
        # 降采样逻辑
        if len(img_files) > len(states) * 1.5:
            print(f"Downsampling {cam}: {len(img_files)} -> {len(states)} (approx)")
            step = 3 
            img_files = img_files[::step]
        
        cam_imgs = []
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cam_imgs.append(img)
        
        if len(cam_imgs) != len(states):
            print(f"Warning: Mismatch in {cam} images ({len(cam_imgs)}) and states ({len(states)}) for {episode_dir.name}")
            min_len = min(len(cam_imgs), len(states))
            cam_imgs = cam_imgs[:min_len]
            
        images[cam] = np.array(cam_imgs)
        
    return states, images, instructions

def convert_data(raw_dir, repo_id, output_dir):
    raw_dir = Path(raw_dir)
    # output_dir 是你的 my_custom_data
    dataset = create_dataset(repo_id, output_dir)
    
    episode_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    
    for ep_idx, ep_dir in enumerate(tqdm.tqdm(episode_dirs)):
        states, images, instructions = load_episode_data(ep_dir)
        
        min_len = len(states)
        for cam, imgs in images.items():
            min_len = min(min_len, len(imgs))
            
        if min_len == 0:
            print(f"Skipping empty episode {ep_dir.name}")
            continue
            
        states = states[:min_len]
        instructions = instructions[:min_len]
        for cam in images:
            images[cam] = images[cam][:min_len]
            
        num_frames = min_len
        actions = np.zeros_like(states)
        actions[:-1] = states[1:]
        actions[-1] = states[-1]
        
        for i in range(num_frames):
            frame = {
                "observation.state": torch.from_numpy(states[i]),
                "action": torch.from_numpy(actions[i]),
                "language_instruction": instructions[i],
            }
            for cam, imgs in images.items():
                frame[f"observation.images.{cam}"] = imgs[i]
            
            dataset.add_frame(frame, task=instructions[i])
        dataset.save_episode()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw data directory")
    parser.add_argument("--repo-id", type=str, required=True, help="dataset name (e.g. xarm_convert)")
    parser.add_argument("--output-dir", type=str, default="./my_dataset_output", help="Root directory for output")
    args = parser.parse_args()
    
    convert_data(args.raw_dir, args.repo_id, args.output_dir)