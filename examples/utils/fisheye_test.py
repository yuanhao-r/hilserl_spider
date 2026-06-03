import cv2
import argparse

def read_fisheye_camera(camera_indexes=[0]):
    """
    读取鱼眼相机视频流并显示
    
    参数:
        camera_index: 相机索引，通常从0开始
    """
    # 打开相机
    caps = []
    for idx in camera_indexes:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)  # 使用V4L2后端，适用于Linux系统
        # 检查相机是否成功打开
        if not cap.isOpened():
            print(f"无法打开相机，索引: {idx}")
            return
        
        # 对于鱼眼相机，可以尝试设置更高的分辨率
        # 注意：分辨率需要相机支持，不同相机支持的分辨率不同
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("成功打开鱼眼相机，按 'q' 键退出")
        print(f"当前分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"设置帧率: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"设置fourcc: {cap.get(cv2.CAP_PROP_FOURCC)}")

        caps.append(cap)
        
    image_crop = None # lambda img: img[150:390, 500:820]
    # image_crop = lambda img: img[600:1166, 278:784]
    # image_crop = lambda img: img[278:684, 600:1066]

    
    while True:
        # 读取一帧
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            
            # 检查是否成功读取帧
            if not ret:
                print("无法接收帧 (可能已到达流末尾)。退出...")
                break
            
            cropped_frame = image_crop(frame) if image_crop is not None else frame
            
            # 显示原始鱼眼图像
            cv2.imshow('Fisheye Camera Feed %d' % i, cropped_frame)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) == ord('q'):
                break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 解析命令行参数，方便指定相机索引
    parser = argparse.ArgumentParser(description='读取鱼眼相机视频流')
    parser.add_argument('--index', type=str, default="0", help='相机索引，默认为0')
    args = parser.parse_args()
    
    indexes = [ int(s) for s in args.index.split(",") ]
    read_fisheye_camera(indexes)
