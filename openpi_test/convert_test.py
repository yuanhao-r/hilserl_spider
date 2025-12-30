import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openpi.shared import image_tools




# 相对值放大倍数
SCALE_FACTOR = 1.0
image_resolution=(224, 224)

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
    
    # TARGET_H_HIGH, TARGET_W_HIGH = 479, 434
    # TARGET_H_LEFT, TARGET_W_LEFT = 420, 357 
    # TARGET_H_RIGHT, TARGET_W_RIGHT = 412, 349
    # cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    # for cam in cameras:
    #     if cam == "cam_high":
    #         H, W = TARGET_H_HIGH, TARGET_W_HIGH
    #     elif cam == "cam_left_wrist":
    #         H, W = TARGET_H_LEFT, TARGET_W_LEFT
    #     elif cam == "cam_right_wrist":
    #         H, W = TARGET_H_RIGHT, TARGET_W_RIGHT
    #     else:
    #         # 默认值
    #         H, W = 480, 640
    
    # cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    # for cam in cameras:
    #     # features[f"observation.images.{cam}"] = {
    #     #     "dtype": "image",
    #     #     "shape": (3, 480, 640), 
    #     #     "names": ["channels", "height", "width"],
    #     # }
    #     features[f"observation.images.{cam}"] = {
    #         "dtype": "image",
    #         # 直接使用实际的尺寸 H 和 W
    #         "shape": (3, H, W),  # 或者 (H, W, 3) 都可以被 lerobot 接受
    #         "names": ["channels", "height", "width"],
    #     }
    # TARGET_H_HIGH, TARGET_W_HIGH = 479, 434
    # TARGET_H_LEFT, TARGET_W_LEFT = 420, 357 
    # TARGET_H_RIGHT, TARGET_W_RIGHT = 412, 349
    
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    
    for cam in cameras:
        # if cam == "cam_high":
        #     H, W = TARGET_H_HIGH, TARGET_W_HIGH
        # elif cam == "cam_left_wrist":
        #     H, W = TARGET_H_LEFT, TARGET_W_LEFT
        # elif cam == "cam_right_wrist":
        #     H, W = TARGET_H_RIGHT, TARGET_W_RIGHT
        # else:
            # 默认值
        H, W = image_resolution

        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            # 直接使用实际的尺寸 H 和 W
            "shape": (3, H, W),  # 或者 (H, W, 3) 都可以被 lerobot 接受
            "names": ["channels", "height", "width"],
        }
        
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
    states_delta = [] # 存储相对值
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

        # --- 2. 获取相对值 ---
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
    
    # ========== 新增：打印当前episode的基础数据 ==========
    print(f"\n=== Episode: {episode_dir.name} 数据概览 ===")
    print(f"总帧数: {len(states_abs)}")
    print(f"--- observation.state (绝对值) 样例 (前3帧) ---")
    print(f"维度: {states_abs.shape}")
    print(np.round(states_abs[:3], 4))  # 保留4位小数，打印前3帧
    print(f"--- observation.state_delta (相对值) 样例 (前3帧) ---")
    print(f"维度: {states_delta.shape}")
    print(np.round(states_delta[:3], 6))  # 相对值精度更高，保留6位
    print(f"--------------------------------------------------")
    
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
            cam_imgs.append(img)
        
        # 对齐逻辑
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
    
    # 用于保存所有episode的示例数据
    all_sample_data = []
    
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
        # print("SSSS",st_delta.shape)
        # --- 构建 Action (绝对值) ---
        act_abs = np.zeros_like(st_abs)
        act_abs[:-1] = st_abs[1:]
        act_abs[-1] = st_abs[-1] # 最后一帧重复

        # --- 构建 Action (相对值) ---
        act_delta = np.zeros_like(st_delta)
        act_delta[:-1] = st_delta[1:]
        act_delta[-1] = st_delta[-1]
        
        # ========== 新增：打印Action相关数据 ==========
        # print(f"\n=== Episode: {ep_dir.name} Action 数据 ===")
        # print(f"--- action (绝对值) 样例 (前20帧) ---")
        # print(np.round(act_abs[:20], 4))
        if len(act_delta.shape) != 2:
            print(f"Warning: action (绝对值) 维度异常: {act_delta.shape}")
        # print(f"--- action_delta (相对值) 样例 (前20帧) ---")
        # print(np.round(act_delta[:20], 6))
        # print(f"------------------------------------------")
        
        # 保存当前episode的第0帧作为示例
        sample_frame = {
            "episode": ep_dir.name,
            "frame_idx": 0,
            "observation.state": st_abs[0].tolist(),
            "observation.state_delta": st_delta[0].tolist(),
            "action": act_abs[0].tolist(),
            "action_delta": act_delta[0].tolist(),
            "instruction": instructions[0]
        }
        all_sample_data.append(sample_frame)
        
        # 写入数据集
        for i in range(num_frames):
            assert st_abs[i].shape == st_delta[i].shape
            assert act_abs[i].shape == act_delta[i].shape
            frame = {
                # 绝对值
                "observation.state": torch.from_numpy(st_abs[i]),
                "action": torch.from_numpy(act_abs[i]),
                # 相对值
                "observation.state_delta": torch.from_numpy(st_delta[i]),
                "action_delta": torch.from_numpy(act_delta[i]),
                # 指令
                "language_instruction": instructions[i],
            }
            # image = imgs[i]
            # if image.shape[:2] != image_resolution:
            #     image = image_tools.resize_with_pad(image, *image_resolution)
            #     # print("after resize image",image.shape, flush=True)
            #     image = np.array(image)
            # for cam, imgs in images.items():
            #     frame[f"observation.images.{cam}"] = image
            
            # 为每个摄像头单独处理图像尺寸
            for cam, cam_imgs in images.items():  # 重命名变量避免冲突
                img = cam_imgs[i]  # 取当前摄像头、当前帧的图像
                # 尺寸转换：如果不一致则缩放+补边到 224×224
                if img.shape[:2] != image_resolution:
                    img = image_tools.resize_with_pad(img, *image_resolution)
                    img = np.array(img)
                # 赋值到对应摄像头的字段
                frame[f"observation.images.{cam}"] = img
        
            dataset.add_frame(frame, task=instructions[i])
        dataset.save_episode()
    
    # ========== 新增：保存所有示例数据到JSON文件 ==========
    sample_file = Path(output_dir) / repo_id / "sample_data.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(all_sample_data, f, indent=4, ensure_ascii=False)
    print(f"\n所有episode的第0帧示例数据已保存到: {sample_file}")

# ========== 新增：读取转换后的数据集并验证 ==========
def verify_converted_data(output_dir, repo_id):
    print("\n=== 验证转换后的数据集 ===")
    dataset_path = Path(output_dir) / repo_id
    if not dataset_path.exists():
        print(f"数据集不存在: {dataset_path}")
        return
    
    # 加载数据集
    dataset = LeRobotDataset(str(dataset_path))
    print(f"数据集总长度: {len(dataset)}")
    
    # 取第0个数据点验证
    sample_data = dataset[0]
    print(f"\n--- 转换后数据集第0个数据点 ---")
    print(f"observation.state (绝对值): {np.round(sample_data['observation.state'], 4)}")
    print(f"observation.state_delta (相对值): {np.round(sample_data['observation.state_delta'], 6)}")
    print(f"action (绝对值): {np.round(sample_data['action'], 4)}")
    print(f"action_delta (相对值): {np.round(sample_data['action_delta'], 6)}")
    print(f"language_instruction: {sample_data['language_instruction']}")
    print(f"----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="/home/hil-serl/openpi_test/data_auto_queue_PutAndRecord_1212/raw", help="Path to raw data directory")
    parser.add_argument("--repo-id", type=str, default="xarm_autoPut_pi05_dataset", help="dataset name (e.g. xarm_convert)")
    parser.add_argument("--output-dir", type=str, default="/home/hil-serl/openpi_test/lerobot_4object_autoPut_data_1212night_224_224", help="Root directory for output")
    args = parser.parse_args()
    
    # 转换数据（带实时打印）
    convert_data(args.raw_dir, args.repo_id, args.output_dir)
    
    # 验证转换后的数据
    verify_converted_data(args.output_dir, args.repo_id)