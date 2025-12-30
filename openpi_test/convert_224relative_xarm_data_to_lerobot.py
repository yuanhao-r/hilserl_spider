import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 相对值放大倍数
SCALE_FACTOR = 1.0

# 采集代码中定义的裁剪尺寸（与采集代码保持一致）
CROP_CONFIGS = {
    "cam_high": (479, 434),       # H, W
    "cam_left_wrist": (420, 357), # H, W
    "cam_right_wrist": (412, 349) # H, W
}

def create_dataset(repo_id, root_dir, robot_type="xarm"):
    root_dir = Path(root_dir)
    
    # === 计算完整的最终路径 ===
    output_path = root_dir / repo_id
    
    # 清理旧数据
    if output_path.exists():
        print(f"Cleaning up existing dataset at {output_path}")
        shutil.rmtree(output_path)

    features = {
        # --- 1. 绝对值 ---
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
        
        # --- 2. 相对值 ---
        "observation.state_delta": {
            "dtype": "float32",
            "shape": (7,), 
            "names": ["j1_d", "j2_d", "j3_d", "j4_d", "j5_d", "j6_d", "gripper"],
        },
        "action_delta": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["j1_d", "j2_d", "j3_d", "j4_d", "j5_d", "j6_d", "gripper"],
        },

        "language_instruction": {
            "dtype": "string",
            "shape": (1,),
            "names": ["instruction"],
        },
    }
    
    # 关键修复：按实际裁剪尺寸定义每个摄像头的特征形状
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for cam in cameras:
        H, W = CROP_CONFIGS[cam]  # 读取对应摄像头的裁剪尺寸
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, H, W),  # (C, H, W) 格式，与实际采集的图像匹配
            "names": ["channels", "height", "width"],
        }
        
    # 创建数据集（使用正确的特征定义）
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path, 
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
        
    states_abs = []   # 存储绝对值
    states_delta = [] # 存储相对值(放大后)
    instructions = []
    
    for line in lines:
        gripper_state = line.get("gripper_state", 0.0)
        
        # --- 1. 获取绝对值 ---
        if "joint_abs" in line:
            raw_abs = line["joint_abs"]
        else:
            raw_abs = line["joint_pos"] # 兼容旧数据
            
        # 截取前6维 + 夹爪
        abs_6d = np.array(raw_abs[:6], dtype=np.float32)
        final_abs = np.append(abs_6d, gripper_state)
        states_abs.append(final_abs)

        # --- 2. 获取相对值并放大 ---
        if "joint_abs" in line:
            raw_delta = line["joint_pos"]
        else:
            raw_delta = [0.0] * 7 
            
        # 截取前6维 * 放大倍数
        delta_6d = np.array(raw_delta[:6], dtype=np.float32) * SCALE_FACTOR
        final_delta = np.append(delta_6d, gripper_state)
        states_delta.append(final_delta)

        instructions.append(line.get("instruction", ""))
        
    # 转为 numpy
    states_abs = np.array(states_abs, dtype=np.float32)
    states_delta = np.array(states_delta, dtype=np.float32)
    
    images = {}
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for cam in cameras:
        cam_dir = episode_dir / "images" / cam
        if not cam_dir.exists():
            continue
            
        img_files = sorted(cam_dir.glob("*.jpg"))
        
        cam_imgs = []
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 验证图像尺寸是否符合裁剪配置（可选，用于调试）
            if img.shape[:2] != CROP_CONFIGS[cam]:
                print(f"Warning: {cam} image {img_file.name} shape {img.shape[:2]} != expected {CROP_CONFIGS[cam]}")
            cam_imgs.append(img)
        
        # 对齐逻辑 (以 states_abs 长度为准)
        if len(cam_imgs) != len(states_abs):
            print(f"Warning: Mismatch in {cam} images ({len(cam_imgs)}) and states ({len(states_abs)}) for {episode_dir.name}")
            min_len = min(len(cam_imgs), len(states_abs))
            cam_imgs = cam_imgs[:min_len]
            
        images[cam] = np.array(cam_imgs)
        
    return states_abs, states_delta, images, instructions

def convert_data(raw_dir, repo_id, output_dir):
    raw_dir = Path(raw_dir)
    dataset = create_dataset(repo_id, output_dir)
    
    episode_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    
    for ep_idx, ep_dir in enumerate(tqdm.tqdm(episode_dirs)):
        # 加载数据
        st_abs, st_delta, images, instructions = load_episode_data(ep_dir)
        
        # 计算最小长度对齐
        min_len = len(st_abs)
        for cam, imgs in images.items():
            min_len = min(min_len, len(imgs))
            
        if min_len == 0:
            print(f"Skipping empty episode {ep_dir.name}")
            continue
            
        # 统一裁切
        st_abs = st_abs[:min_len]
        st_delta = st_delta[:min_len]
        instructions = instructions[:min_len]
        for cam in images:
            images[cam] = images[cam][:min_len]
            
        num_frames = min_len
        
        # --- 构建 Action (绝对值) ---
        act_abs = np.zeros_like(st_abs)
        act_abs[:-1] = st_abs[1:]
        act_abs[-1] = st_abs[-1] # 最后一帧重复

        # --- 构建 Action (相对值) ---
        act_delta = np.zeros_like(st_delta)
        act_delta[:-1] = st_delta[1:]
        act_delta[-1] = st_delta[-1]
        
        # 关键修复：之前错误地将 state_delta/action_delta 赋值为绝对值，这里修正
        for i in range(num_frames):
            frame = {
                # 绝对值
                "observation.state": torch.from_numpy(st_abs[i]),
                "action": torch.from_numpy(act_abs[i]),
                
                # 相对值（修正：使用 st_delta/act_delta 而非 st_abs/act_abs）
                "observation.state_delta": torch.from_numpy(st_delta[i]),
                "action_delta": torch.from_numpy(act_delta[i]),
                
                "language_instruction": instructions[i],
            }
            # 写入每个摄像头的图像
            for cam, imgs in images.items():
                frame[f"observation.images.{cam}"] = imgs[i]
            
            dataset.add_frame(frame, task=instructions[i])
        dataset.save_episode()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="/home/hil-serl/openpi_test/data_auto_queue_PutAndRecord_1212/raw", help="Path to raw data directory")
    parser.add_argument("--repo-id", type=str, default="xarm_relative_pi05_dataset", help="dataset name (e.g. xarm_convert)")
    parser.add_argument("--output-dir", type=str, default="/home/hil-serl/openpi_test/lerobot_4object_autoPut_data_1212night_originSize", help="Root directory for output")
    args = parser.parse_args()
    
    convert_data(args.raw_dir, args.repo_id, args.output_dir)
