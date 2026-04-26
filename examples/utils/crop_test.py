import cv2
import argparse


def main(camera_index: int = 0):
    """
    打开指定编号的鱼眼相机，在图像上用鼠标画框，并在终端打印对应的裁剪范围。

    用法：
      python fisheye_crop_tool.py --index 8

    操作说明：
      - 鼠标左键按下并拖动：画矩形框
      - 松开左键：锁定当前框，并在终端打印裁剪代码
      - 按 'c' 键：清除当前框
      - 按 'q' 或 ESC：退出
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"无法打开相机，索引: {camera_index}")
        return

    # 尝试设置常用分辨率和参数（与你现有 fisheye_test 一致）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("成功打开鱼眼相机，按住鼠标左键拖动画框，松开后终端会输出裁剪范围。")
    print("按 'c' 清除当前框，按 'q' 或 ESC 退出。")
    print(f"当前分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    window_name = "Fisheye Crop Tool"
    cv2.namedWindow(window_name)

    # 鼠标交互状态
    drawing = False
    ix, iy = -1, -1
    rect = None  # (x1, y1, x2, y2)

    def clamp(v, low, high):
        return max(low, min(high, v))

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, ix, iy, rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            rect = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # 实时更新矩形，但真正打印在 BUTTONUP 时
            x1, y1 = ix, iy
            x2, y2 = x, y
            rect = (x1, y1, x2, y2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = ix, iy
            x2, y2 = x, y
            # 规范化坐标为左上/右下 & 防止越界
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            x1 = clamp(x1, 0, w - 1)
            x2 = clamp(x2, 0, w)
            y1 = clamp(y1, 0, h - 1)
            y2 = clamp(y2, 0, h)
            rect = (x1, y1, x2, y2)

            # 在终端打印裁剪信息（与 config 里的 lambda 一致：img[y1:y2, x1:x2]）
            print("\n================ 裁剪结果 ================")
            print(f"图像尺寸: 高={h}, 宽={w}")
            print(f"矩形像素坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print("可直接复制到 EnvConfig.IMAGE_CROP 中使用，例如：")
            print(f'    "wrist_2": lambda img: img[{y1}:{y2}, {x1}:{x2}],')
            print("==========================================\n")

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧 (可能已到达流末尾)。退出...")
            break

        vis = frame.copy()
        if rect is not None:
            x1, y1, x2, y2 = rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # 'q' 或 ESC
            break
        elif key == ord("c"):
            rect = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="鱼眼相机裁剪范围可视化工具")
    parser.add_argument("--index", type=int, default=0, help="相机索引，默认为 0")
    args = parser.parse_args()
    main(args.index)

