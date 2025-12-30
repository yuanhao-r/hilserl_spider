import cv2
import time

def find_camera_crops(camera_indices):
    """
    依次打开相机，允许用户框选裁剪区域
    返回配置字典
    """
    crop_configs = {}
    
    print("="*60)
    print("相机裁剪区域校准工具")
    print("操作指南:")
    print("1. 鼠标拖动: 框选感兴趣区域 (ROI)")
    print("2. 按 SPACE 或 ENTER: 确认选择并进入下一个相机")
    print("3. 按 c: 取消当前选择，重置框选")
    print("4. 如果画面没出来，请检查USB连接")
    print("="*60)

    for name, idx in camera_indices.items():
        print(f"\n>>> 正在连接相机: {name} (Index: {idx})...")
        cap = cv2.VideoCapture(idx)
        
        # 等待相机预热，读取几帧
        for _ in range(10):
            ret, frame = cap.read()
            
        if not ret:
            print(f"[Error] 无法读取相机 {name}，跳过。")
            cap.release()
            continue

        print(f"请在弹出的窗口中框选 {name} 的区域...")
        
        # 使用 OpenCV 自带的 ROI 选择工具
        # 返回格式: (x, y, w, h)
        roi = cv2.selectROI(f"Select ROI for {name}", frame, showCrosshair=True, fromCenter=False)
        
        # 关闭当前窗口
        cv2.destroyWindow(f"Select ROI for {name}")
        cap.release()

        # 如果用户直接关闭窗口没选，通常返回 (0,0,0,0)
        if roi[2] == 0 or roi[3] == 0:
            print(f"[Warn] 未选择 {name} 的区域，将使用全图。")
            crop_configs[name] = None
        else:
            x, y, w, h = roi
            print(f"已确认 {name}: x={x}, y={y}, w={w}, h={h}")
            crop_configs[name] = [x, y, w, h]
            
    print("\n" + "="*60)
    print("校准完成！请将以下字典复制到你的主程序中：")
    print("="*60)
    print("CROP_CONFIGS = {")
    for name, params in crop_configs.items():
        if params:
            print(f"    '{name}': {params},  # [x, y, w, h]")
        else:
            print(f"    '{name}': None,  # 无裁剪")
    print("}")
    print("="*60)

if __name__ == "__main__":
    # 请确认这里的索引与你主程序一致
    cameras = {
        "cam_high": 4,
        "cam_left_wrist": 0,
        "cam_right_wrist": 2
    }
    
    find_camera_crops(cameras)