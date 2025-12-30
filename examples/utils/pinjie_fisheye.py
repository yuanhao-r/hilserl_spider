import cv2
import argparse
import numpy as np

def read_fisheye_cameras(camera_indices=[0, 1, 2], scale_factor=0.5):
    """
    读取多个鱼眼相机视频流，拼接画面并缩小显示
    
    参数:
        camera_indices: 相机索引列表，默认[0,1,2]对应三个相机
        scale_factor: 缩放因子（0-1），0.5表示缩小到原尺寸的50%
    """
    # 存储相机捕获对象
    caps = []
    # 存储每个相机是否成功打开的状态
    camera_status = []
    # 单个相机的原始分辨率（可根据实际情况调整）
    ORIG_WIDTH = 1280
    ORIG_HEIGHT = 720
    # 缩放后的分辨率
    SCALED_WIDTH = int(ORIG_WIDTH * scale_factor)
    SCALED_HEIGHT = int(ORIG_HEIGHT * scale_factor)

    # 初始化所有相机
    for idx in camera_indices:
        try:
            # 打开相机（Linux使用V4L2后端）
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                # 设置相机参数（鱼眼相机通用配置）
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, ORIG_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ORIG_HEIGHT)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                caps.append(cap)
                camera_status.append(True)
                print(f"✅ 成功打开相机 {idx}")
                print(f"  原始分辨率: {ORIG_WIDTH}x{ORIG_HEIGHT}")
                print(f"  缩放后分辨率: {SCALED_WIDTH}x{SCALED_HEIGHT}")
                print(f"  帧率: {cap.get(cv2.CAP_PROP_FPS)}")
            else:
                caps.append(None)
                camera_status.append(False)
                print(f"❌ 无法打开相机 {idx}")
        except Exception as e:
            caps.append(None)
            camera_status.append(False)
            print(f"❌ 相机 {idx} 初始化失败: {str(e)}")
    
    # 检查是否有至少一个相机可用
    if not any(camera_status):
        print("❌ 没有可用的相机，程序退出")
        return
    
    # 定义裁剪函数（可根据需要调整裁剪区域）
    def crop_frame(frame):
        # 示例裁剪区域，可注释掉使用原图
        # return frame[150:390, 500:820]  # 裁剪指定区域
        return frame  # 不裁剪，返回原图
    
    print(f"\n📷 开始显示拼接画面（缩放{scale_factor*100}%），按 'q' 键退出")
    print("⚠️  若某个相机无画面，会显示黑色填充区域")
    
    while True:
        # 存储每个相机缩放后的帧
        frames_scaled = []
        
        # 读取每个相机的帧并处理
        for i, (cap, status) in enumerate(zip(caps, camera_status)):
            if status and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # 1. 裁剪帧（可选）
                    frame = crop_frame(frame)
                    # 2. 缩小帧尺寸
                    frame_scaled = cv2.resize(frame, (SCALED_WIDTH, SCALED_HEIGHT))
                    # 3. 添加相机编号文字标注（适配缩小后的尺寸）
                    cv2.putText(frame_scaled, f"Camera {camera_indices[i]}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frames_scaled.append(frame_scaled)
                else:
                    # 读取失败时生成黑色帧（缩放后尺寸）
                    frames_scaled.append(np.zeros((SCALED_HEIGHT, SCALED_WIDTH, 3), dtype=np.uint8))
            else:
                # 相机未打开时生成黑色帧（缩放后尺寸）
                frames_scaled.append(np.zeros((SCALED_HEIGHT, SCALED_WIDTH, 3), dtype=np.uint8))
        
        # 画面拼接：水平拼接（缩放后总宽度=3*SCALED_WIDTH，高度=SCALED_HEIGHT）
        if len(frames_scaled) == 1:
            combined_frame = frames_scaled[0]
        elif len(frames_scaled) == 2:
            combined_frame = cv2.hconcat(frames_scaled)
        elif len(frames_scaled) >= 3:
            # 只取前3个相机拼接
            combined_frame = cv2.hconcat(frames_scaled[:3])
        
        # 可选：再次整体缩放拼接后的画面（若仍过大）
        # combined_frame = cv2.resize(combined_frame, (1920, 480))  # 强制缩放到1920x480
        
        # 显示拼接后的画面
        cv2.imshow('Multi-Fisheye Camera Feed (Press Q to Exit)', combined_frame)
        
        # 按q键退出（增加等待时间，降低CPU占用）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放所有相机资源
    for cap in caps:
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()
    print("\n🔌 已释放所有相机资源，程序退出")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='读取多个鱼眼相机视频流并缩小拼接显示')
    parser.add_argument('--indices', nargs='+', type=int, default=[0, 2, 4],
                        help='相机索引列表，例如 --indices 0 1 2（默认）')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='画面缩放因子（0-1），默认0.5（50%），建议0.3-0.7')
    args = parser.parse_args()
    
    # 确保缩放因子合法
    if not (0 < args.scale <= 1):
        print("❌ 缩放因子必须在0-1之间，使用默认值0.5")
        args.scale = 0.5
    
    # 确保至少传入一个相机索引
    if len(args.indices) == 0:
        print("❌ 请指定至少一个相机索引，例如 --indices 0")
    else:
        # 最多处理3个相机
        read_fisheye_cameras(args.indices[:3], args.scale)
