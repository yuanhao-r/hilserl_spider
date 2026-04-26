"""
Gemini 2D 检测 + 奥比中光 D2C 对齐深度 → 3D 目标点（机械臂用）

流程：
1. 奥比中光相机开启彩色 + 深度，软件 D2C 对齐（与 sync_align / object_detection_sw_align 一致）
2. 按空格截取一帧，保存彩色图并调用 Gemini 得到目标 2D 点（如插排插孔）
3. 在对齐的深度图上取该像素深度，经 2d_to_2d + 2d_to_3d 得到彩色相机系下的 3D 点，作为机械臂目标点

依赖：pyorbbecsdk、google-genai、examples/utils.py（frame_to_bgr_image）
"""
# 避免 Gemini 请求头中出现非 ASCII（如 LANG=zh_CN 导致）引发 UnicodeEncodeError
import os
if "LANG" in os.environ and not os.environ.get("LANG", "").startswith("C") and not os.environ.get("LANG", "").startswith("en_US"):
    os.environ["LANG"] = "en_US.UTF-8"
    os.environ["LC_ALL"] = "en_US.UTF-8"

import json
import re
import sys
import tempfile
import threading
import numpy as np
import cv2
# 奥比中光 + examples 工具
# ---------- 自动路径修复逻辑 ----------
# 1. 获取当前文件所在目录: .../examples/experiments/ram_insertion/
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 向上跳两级找到 examples 目录: .../examples/
_EXAMPLES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
# 3. 向上跳三级找到根目录: .../
_ROOT_DIR = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))

# 将这些路径加入 sys.path，解决 utils 和其他模块的导入问题
for _p in [_EXAMPLES_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ------------------------------------

# 尝试导入奥比中光 SDK
try:
    from pyorbbecsdk import (
        Pipeline, Config, AlignFilter, OBStreamType,
        OBSensorType, OBFormat, OBFrameAggregateOutputMode,
    )
    import pyorbbecsdk as ob
except ImportError as e:
    print("错误: 无法导入 pyorbbecsdk，请检查 SDK 是否正确安装。")
    raise e

# 本地兜底：避免 utils.frame_to_bgr_image 导入失败时直接返回 None 导致相机恒 frozen
def _frame_to_bgr_image_fallback(frame):
    width = int(frame.get_width())
    height = int(frame.get_height())
    color_format = frame.get_format()
    data = np.ascontiguousarray(np.asarray(frame.get_data(), dtype=np.uint8))

    if color_format == OBFormat.RGB:
        if data.size < height * width * 3:
            return None
        image = data[: height * width * 3].reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        if data.size < height * width * 3:
            return None
        return data[: height * width * 3].reshape((height, width, 3))
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.YUYV:
        if data.size < height * width * 2:
            return None
        image = data[: height * width * 2].reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        if data.size < height * width * 2:
            return None
        image = data[: height * width * 2].reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.I420:
        if data.size < (height * width * 3) // 2:
            return None
        image = data[: (height * width * 3) // 2].reshape((height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    if color_format == OBFormat.NV12:
        if data.size < (height * width * 3) // 2:
            return None
        image = data[: (height * width * 3) // 2].reshape((height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        if data.size < (height * width * 3) // 2:
            return None
        image = data[: (height * width * 3) // 2].reshape((height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21)
    print(f"警告: 不支持的 Orbbec 像素格式: {color_format}")
    return None


# 尝试从 examples/utils/utils.py 导入工具函数；失败时使用本地兜底转换
try:
    from utils.utils import frame_to_bgr_image as _frame_to_bgr_image_external
except ImportError:
    try:
        from utils import frame_to_bgr_image as _frame_to_bgr_image_external
    except ImportError:
        _frame_to_bgr_image_external = None

if _frame_to_bgr_image_external is None:
    print("警告: 未导入 utils.frame_to_bgr_image，使用本地像素转换兜底。")
    frame_to_bgr_image = _frame_to_bgr_image_fallback
else:
    frame_to_bgr_image = _frame_to_bgr_image_external

# Gemini（可选依赖：仅在执行 detect_hole_3d_one_shot 时才需要）
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:
    genai = None
    types = None

    class ClientError(Exception):
        pass

# ---------- 配置 ----------
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDTmotxn0msTnP8fbv9hBPiXdjfVkTP4RA")
GEMINI_API_KEY = "AIzaSyA6HhvKtAxe7YhVa9t2VOYxACsgW8igq3c"
PROMPT = """
Point to no more than 10 items in the image. The label returned
should be an identifying name for the object detected.
The answer should follow the json format: [{"point": <point>,
"label": <label1>}, ...]. The points are in [y, x] format
normalized to 0-1000.
"""
MIN_DEPTH_MM = 20
MAX_DEPTH_MM = 10000

# 标注图保存路径（按空格截帧并检测后保存）
OUTPUT_ANNOTATED_IMAGE = "output_annotated_3d.jpg"

# “无效深度/转换失败”含义：该像素处深度为 0 或超出有效范围(20mm~10000mm)，
# 无法换算成 3D。常见原因：反光/镜面、遮挡、超出深度视场、物体过远/过近、插孔内黑洞等。

# 预览缩放（仅影响显示，截帧与 3D 仍用原图）：<1 可提高帧率
PREVIEW_SCALE = 0.75


def draw_and_save_annotated_image(
    img_bgr: np.ndarray,
    items: list[dict],
    output_path: str,
    point_radius: int = 10,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    chosen_pixel: tuple[int, int] | None = None,
) -> None:
    """
    在图上绘制每个点的 2D 位置、名称与 3D 坐标（或「无深度」），并保存。
    items 中每项需有 "label", "pixel"(x,y), "point_3d_mm"(x,y,z) 或 None。
    chosen_pixel: 若给出 (x,y)，则该像素对应的点会加粗黄圈并标注「选中」。
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    colors_ok = [
        (0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 100, 100),
        (100, 255, 255), (255, 100, 255), (200, 200, 0),
    ]
    color_fail = (0, 0, 255)  # 无深度用红色
    color_chosen = (0, 255, 255)  # 选中用黄色
    for i, r in enumerate(items):
        x_px, y_px = r["pixel"]
        label = r.get("label", str(i))
        pt3 = r.get("point_3d_mm")
        is_chosen = chosen_pixel is not None and (x_px, y_px) == chosen_pixel
        if is_chosen:
            color = color_chosen
        elif pt3 is not None:
            color = colors_ok[i % len(colors_ok)]
        else:
            color = color_fail
        thickness = 3 if is_chosen else 2
        cv2.circle(out, (x_px, y_px), point_radius, color, thickness)
        cv2.circle(out, (x_px, y_px), point_radius + 2, color, 1)
        if pt3 is not None:
            text = f"{label} ({pt3[0]:.0f},{pt3[1]:.0f},{pt3[2]:.0f})mm"
        else:
            text = f"{label} [无深度]"
        if is_chosen:
            text += " [选中]"
        # 标签放在点右上方，避免超出图像
        tx, ty = x_px + point_radius + 4, y_px - 6
        if ty < 20:
            ty = y_px + point_radius + 14
        tx = max(2, min(tx, w - 250))
        ty = max(18, min(ty, h - 4))
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.imwrite(output_path, out)
    print(f"已保存标注图: {output_path}")


def parse_points_json(text: str) -> list[dict]:
    """从 Gemini 返回文本解析 [{"point": [y,x], "label": "..."}, ...]，点归一化 0-1000"""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "point" in data:
        return [data]
    if isinstance(data, dict):
        for key in ("points", "items", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def invert_extrinsic(d2c: ob.OBExtrinsic) -> ob.OBExtrinsic:
    """深度→彩色外参求逆，得到彩色→深度，用于 2d_to_2d( color_2d -> depth_2d )。"""
    R = np.array(d2c.rot, dtype=np.float32)
    if R.size == 9:
        R = R.reshape(3, 3)
    t = np.array(d2c.transform, dtype=np.float32).ravel()
    R_inv = R.T
    t_inv = -R_inv @ t
    c2d = ob.OBExtrinsic()
    c2d.rot = R_inv
    c2d.transform = t_inv
    return c2d


def get_calibration_from_profiles(color_profile, depth_profile):
    """从 pipeline 的 stream profile 取内参、畸变、外参（不依赖帧）。"""
    try:
        color_profile = color_profile.as_video_stream_profile()
    except Exception:
        pass
    try:
        depth_profile = depth_profile.as_video_stream_profile()
    except Exception:
        pass
    return {
        "color_intrinsics": color_profile.get_intrinsic(),
        "color_distortion": color_profile.get_distortion(),
        "depth_intrinsics": depth_profile.get_intrinsic(),
        "depth_distortion": depth_profile.get_distortion(),
        "d2c_extrinsic": depth_profile.get_extrinsic_to(color_profile),
    }


def get_frame_data(color_frame, depth_frame):
    """从帧取内参、畸变、外参、深度数据（与 examples/coordinate_transform.py 一致）。"""
    color_frame = color_frame.as_video_frame()
    depth_frame = depth_frame.as_video_frame()
    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()
    color_profile = color_frame.get_stream_profile()
    depth_profile = depth_frame.get_stream_profile()
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()
    color_distortion = color_profile.as_video_stream_profile().get_distortion()
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
    depth_distortion = depth_profile.as_video_stream_profile().get_distortion()
    d2c_extrinsic = depth_profile.get_extrinsic_to(color_profile)
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
        depth_height, depth_width
    )
    return {
        "color_intrinsics": color_intrinsics,
        "color_distortion": color_distortion,
        "depth_intrinsics": depth_intrinsics,
        "depth_distortion": depth_distortion,
        "d2c_extrinsic": d2c_extrinsic,
        "depth_data": depth_data,
        "depth_scale": depth_frame.get_depth_scale(),
        "depth_width": depth_width,
        "depth_height": depth_height,
    }


class OrbbecCapture:
    """
    与 FisheyeCapture 同接口：供 env 的 VideoCapture 包装，在 record_demos / run_actor 中持续提供 wrist_1 图像。
    - read() -> (ret, frame): frame 为 BGR numpy，与 OpenCV 一致。
    - get_aligned_frame() -> (color_bgr, frame_data): 供 detect_hole_3d_one_shot 复用同一 pipeline，避免双开。

    注意：pyorbbecsdk 的 Pipeline 不支持多线程同时调用 wait_for_frames。环境中 VideoCapture 会在
    后台线程持续调用 read()，而 go_to_reset 时主线程会调用 get_aligned_frame()，必须用 _lock 串行化，
    否则会出现相机掉线、Device is deactivated/disconnected。
    """
    @staticmethod
    def list_connected_devices():
        """
        列出当前可见的 Orbbec 设备信息，便于按序列号选择。
        返回: [{"index": i, "serial_number": "...", "name": "..."}]
        """
        devices = []
        try:
            ctx = ob.Context()
            device_list = ctx.query_devices()
            for i in range(device_list.get_count()):
                dev = device_list.get_device_by_index(i)
                info = dev.get_device_info()
                devices.append(
                    {
                        "index": i,
                        "serial_number": info.get_serial_number(),
                        "name": info.get_name(),
                    }
                )
        except Exception:
            return []
        return devices

    @staticmethod
    def _select_device(serial_number=None, device_index=None):
        ctx = ob.Context()
        device_list = ctx.query_devices()
        count = device_list.get_count()
        if count <= 0:
            raise RuntimeError("未检测到 Orbbec 设备，请检查 USB 连接。")

        if serial_number is not None:
            sn = str(serial_number).strip()
            if sn == "":
                raise ValueError("serial_number 为空字符串。")
            try:
                return device_list.get_device_by_serial_number(sn)
            except Exception:
                for i in range(count):
                    dev = device_list.get_device_by_index(i)
                    info = dev.get_device_info()
                    if info.get_serial_number() == sn:
                        return dev
                available = [
                    device_list.get_device_by_index(i).get_device_info().get_serial_number()
                    for i in range(count)
                ]
                raise RuntimeError(
                    f"未找到序列号为 {sn} 的 Orbbec 设备。当前可用设备: {available}"
                )

        if device_index is None:
            idx = 0
        else:
            idx = int(device_index)
        if idx < 0 or idx >= count:
            raise RuntimeError(
                f"device_index={idx} 越界，当前设备数量={count}（有效范围: 0~{count - 1}）。"
            )
        return device_list.get_device_by_index(idx)

    def __init__(self, name, dim=(1280, 720), serial_number=None, device_index=None):
        self.name = name
        self.dim = dim
        self._lock = threading.Lock()
        selected_device = None
        if serial_number is not None or device_index is not None:
            selected_device = self._select_device(
                serial_number=serial_number, device_index=device_index
            )
            self._pipeline = Pipeline(selected_device)
        else:
            self._pipeline = Pipeline()
        config = Config()
        try:
            profile_list = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            self._color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
            config.enable_stream(self._color_profile)
            profile_list = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            self._depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(self._depth_profile)
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception as e:
            print(f"OrbbecCapture 流配置错误: {e}")
            raise
        self._pipeline.start(config)
        self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self._pipeline_calibration = None
        try:
            self._pipeline_calibration = get_calibration_from_profiles(
                self._color_profile, self._depth_profile
            )
        except Exception:
            pass
        # 预热
        for _ in range(30):
            self._pipeline.wait_for_frames(100)
        if selected_device is not None:
            info = selected_device.get_device_info()
            print(
                "OrbbecCapture (wrist_1) 已启动: "
                f"name={info.get_name()} serial={info.get_serial_number()}"
            )
        else:
            print("OrbbecCapture (wrist_1) 已启动")

    def _get_one_frame(self, timeout_ms=500):
        """取一帧对齐后的 color + depth，返回 (color_bgr, aligned_depth_frame, frame_data) 或 (None, None, None)。"""
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if not frames:
            return None, None, None
        color_frame_raw = frames.get_color_frame()
        depth_frame_raw = frames.get_depth_frame()
        if not color_frame_raw or not depth_frame_raw:
            return None, None, None
        frames = self._align_filter.process(frames)
        if not frames:
            return None, None, None
        frames = frames.as_frame_set()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None
        color_bgr = frame_to_bgr_image(color_frame)
        if color_bgr is None:
            return None, None, None
        color_bgr = np.ascontiguousarray(color_bgr.copy())
        if self._pipeline_calibration is not None:
            frame_data = dict(self._pipeline_calibration)
            frame_data["depth_data"] = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            ).reshape(depth_frame.get_height(), depth_frame.get_width())
            frame_data["depth_scale"] = depth_frame.get_depth_scale()
            frame_data["depth_width"] = depth_frame.get_width()
            frame_data["depth_height"] = depth_frame.get_height()
        else:
            try:
                frame_data = get_frame_data(color_frame_raw, depth_frame_raw)
                if frame_data is not None:
                    frame_data["depth_data"] = np.frombuffer(
                        depth_frame.get_data(), dtype=np.uint16
                    ).reshape(depth_frame.get_height(), depth_frame.get_width())
                    frame_data["depth_scale"] = depth_frame.get_depth_scale()
                    frame_data["depth_width"] = depth_frame.get_width()
                    frame_data["depth_height"] = depth_frame.get_height()
            except Exception:
                frame_data = None
        return color_bgr, depth_frame, frame_data

    def read(self):
        """与 FisheyeCapture 一致：返回 (ret, frame)，frame 为 BGR。用于 env 的 get_im() 采集 wrist_1。"""
        # 机械臂运动时 USB/负载可能导致偶发超时，内部重试 + 稍长超时，减少误报 camera frozen
        with self._lock:
            color_bgr, _, _ = self._get_one_frame(timeout_ms=1000)
            if color_bgr is None:
                color_bgr, _, _ = self._get_one_frame(timeout_ms=1000)
            if color_bgr is None:
                color_bgr, _, _ = self._get_one_frame(timeout_ms=1500)
        if color_bgr is None:
            return False, None
        h, w = color_bgr.shape[:2]
        if (w, h) != self.dim:
            color_bgr = cv2.resize(color_bgr, self.dim)
        return True, np.ascontiguousarray(color_bgr)

    def get_aligned_frame(self):
        """返回 (color_bgr, frame_data)，供 detect_hole_3d_one_shot 复用同一 pipeline。"""
        with self._lock:
            color_bgr, _, frame_data = self._get_one_frame(timeout_ms=5000)
        if color_bgr is None or frame_data is None:
            return None, None
        return np.ascontiguousarray(color_bgr.copy()), frame_data

    def close(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass
        print("OrbbecCapture 已关闭")


def pixel_to_3d_in_color_frame(
    u_px: int,
    v_px: int,
    depth_mm: float,
    frame_data: dict,
) -> tuple[float, float, float] | None:
    """
    彩色像素 (u_px, v_px) + 深度 depth_mm -> 彩色相机坐标系下的 3D (x,y,z) 毫米。
    先 2d_to_2d( color -> depth ) 得到深度图像素，再 2d_to_3d 得到 3D。
    """
    if depth_mm <= 0 or depth_mm < MIN_DEPTH_MM or depth_mm > MAX_DEPTH_MM:
        return None
    ci = frame_data["color_intrinsics"]
    cd = frame_data["color_distortion"]
    di = frame_data["depth_intrinsics"]
    dd = frame_data["depth_distortion"]
    d2c = frame_data["d2c_extrinsic"]
    c2d = invert_extrinsic(d2c)

    try:
        depth_2d = ob.transformation2dto2d(
            ob.OBPoint2f(float(u_px), float(v_px)),
            float(depth_mm),
            ci, cd, di, dd, c2d,
        )
        pt3 = ob.transformation2dto3d(
            depth_2d,
            float(depth_mm),
            di,
            d2c,
        )
        return (pt3.x, pt3.y, pt3.z)
    except Exception:
        return None


def sanitize_api_key(key: str) -> str:
    """去掉 BOM、首尾空白和非 ASCII，避免请求头编码错误。"""
    if not key:
        return key
    key = key.strip().strip("\ufeff")
    return "".join(c for c in key if ord(c) < 128)


def request_gemini_2d_points(image_bgr: np.ndarray, api_key: str) -> list[dict]:
    """用 Gemini 对 BGR 图做 2D 点检测，返回 [{"point": [y,x], "label": "..."}, ...]，点 0-1000。"""
    if genai is None or types is None:
        raise RuntimeError(
            "google-genai 未安装，无法调用 Gemini。当前项目若只使用 Orbbec RGB，可忽略此功能。"
        )
    _, buf = cv2.imencode(".jpg", image_bgr)
    image_bytes = buf.tobytes()
    key = sanitize_api_key(api_key)
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=120_000),
    )
    response = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = response.text or ""
    print(f"DEBUG: Gemini raw response: '{text}'") 
    return parse_points_json(text)


def capture_one_aligned_frame(timeout_ms=5000, max_attempts=30):
    """
    启动 Orbbec 管线，采集一帧 D2C 对齐后的彩色+深度，返回 (color_bgr, frame_data)。
    frame_data 包含 depth_data, depth_scale, 内参/外参等，用于 pixel_to_3d_in_color_frame。
    返回 (None, None) 表示失败。
    """
    pipeline = Pipeline()
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    except Exception as e:
        print("Orbbec 流配置错误:", e)
        return None, None

    pipeline.start(config)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    print("相机正在预热（调整曝光）...")
    for _ in range(100):  # 跳过前 20 帧
        pipeline.wait_for_frames(100)
    pipeline_calibration = None
    print("相机预热wancheng...")
    try:
        pipeline_calibration = get_calibration_from_profiles(color_profile, depth_profile)
    except Exception:
        pass

    color_bgr = None
    frame_data = None
    for _ in range(max_attempts):
        frames = pipeline.wait_for_frames(timeout_ms)
        if not frames:
            continue
        color_frame_raw = frames.get_color_frame()
        depth_frame_raw = frames.get_depth_frame()
        if not color_frame_raw or not depth_frame_raw:
            continue
        frames = align_filter.process(frames)
        if not frames:
            continue
        frames = frames.as_frame_set()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        color_bgr = frame_to_bgr_image(color_frame)
        if color_bgr is None:
            continue
        # 拷贝一份，避免 pipeline.stop() 后帧内存被释放导致图像变黑
        color_bgr = np.ascontiguousarray(color_bgr.copy())
        if pipeline_calibration is not None:
            frame_data = dict(pipeline_calibration)
            frame_data["depth_data"] = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            ).reshape(depth_frame.get_height(), depth_frame.get_width())
            frame_data["depth_scale"] = depth_frame.get_depth_scale()
            frame_data["depth_width"] = depth_frame.get_width()
            frame_data["depth_height"] = depth_frame.get_height()
        else:
            try:
                frame_data = get_frame_data(color_frame_raw, depth_frame_raw)
                if frame_data is not None:
                    frame_data["depth_data"] = np.frombuffer(
                        depth_frame.get_data(), dtype=np.uint16
                    ).reshape(depth_frame.get_height(), depth_frame.get_width())
                    frame_data["depth_scale"] = depth_frame.get_depth_scale()
                    frame_data["depth_width"] = depth_frame.get_width()
                    frame_data["depth_height"] = depth_frame.get_height()
            except Exception:
                frame_data = None
        break

    pipeline.stop()
    if color_bgr is None or frame_data is None:
        return None, None
    # 再次确保返回的是独立拷贝，防止后续使用时底层缓冲被复用
    return np.ascontiguousarray(color_bgr.copy()), frame_data


# 默认只从这些 label 中选目标，并取相机系 x 最小的点
# 未传 target_labels 时的默认（如单独跑本脚本）；run_actor/record_demos 用 config.ORBBEC_TARGET_LABELS
DEFAULT_TARGET_LABELS = ("white object", "white", "power adapter", "charger", "electrical outlet", "power strip")


def detect_hole_3d_one_shot(
    api_key=None,
    prefer_label="插孔",
    target_labels=None,
    selection="max_x",
    save_debug_image: bool = True,
    debug_output_dir: str | None = None,
    orbbec_capture=None,
    use_fake_target: bool = False,
):
    """
    一次性：Orbbec 采图 + Gemini 2D 检测 + 深度转 3D，返回目标点在彩色相机系下的 3D 坐标 (x,y,z) 单位 mm。
    - use_fake_target: 为 True 时直接返回固定坐标（带 ±2mm 随机），不采图、不调 API，供测试用。
    - orbbec_capture / target_labels / selection: 同上。
    失败返回 None。
    """
    
    # 测试模式：固定坐标 + ±2mm 随机，不采图、不调 API（config.ORBBEC_USE_FAKE_TARGET=True 或 环境变量 USE_FAKE_GEMINI_TARGET=1）
    if use_fake_target or os.environ.get("USE_FAKE_GEMINI_TARGET", "").strip().lower() in ("1", "true", "yes"):
        base_x, base_y, base_z = 41, 30, 480
        noise = np.random.uniform(-2, 2, size=3)
        fake_target = (base_x + noise[0], base_y + noise[1], base_z + noise[2])
        print(f"!!! [测试模式] 模拟坐标(±2mm 抖动): X:{fake_target[0]:.2f}, Y:{fake_target[1]:.2f}, Z:{fake_target[2]:.2f} mm，未调用 Gemini !!!")
        return fake_target

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key or api_key == "你的_GEMINI_API_KEY":
        print("detect_hole_3d_one_shot: 未设置 GEMINI_API_KEY")
        return None

    if orbbec_capture is not None:
        color_bgr, frame_data = orbbec_capture.get_aligned_frame()
    else:
        color_bgr, frame_data = capture_one_aligned_frame()
    if color_bgr is None or frame_data is None:
        print("detect_hole_3d_one_shot: 采集一帧失败")
        return None

    print("detect_hole_3d_one_shot: 采图完成，正在调用 Gemini API（约 10~30 秒）...")
    try:
        points_2d = request_gemini_2d_points(color_bgr, api_key)
    except ClientError as e:
        if getattr(e, "status_code", None) == 429:
            print(
                "detect_hole_3d_one_shot: Gemini 429 配额用尽。免费版 gemini-robotics-er-1.5-preview 每日约 20 次，"
                "每次 reset/验证 = 1 次。请明日再试或开通付费。详见 https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        else:
            raise
        return None
    print("detect_hole_3d_one_shot: Gemini 返回")
    if not points_2d:
        print("detect_hole_3d_one_shot: Gemini 未解析到 2D 点")
        return None

    labels_filter = target_labels if target_labels is not None else DEFAULT_TARGET_LABELS
    labels_filter = [s.lower() for s in labels_filter]

    h, w = color_bgr.shape[:2]
    depth_data = frame_data["depth_data"]
    depth_scale = frame_data["depth_scale"]
    candidates = []
    all_items = []  # 所有检测点（含无深度的），用于标注图
    for item in points_2d:
        point = item.get("point")
        label = item.get("label", "")
        if not point or len(point) < 2:
            continue
        y_norm, x_norm = float(point[0]), float(point[1])
        x_px = int((x_norm / 1000.0) * w)
        y_px = int((y_norm / 1000.0) * h)
        x_px = max(0, min(x_px, w - 1))
        y_px = max(0, min(y_px, h - 1))
        raw = depth_data[y_px, x_px]
        depth_mm = float(raw) * depth_scale
        if depth_mm <= 0 or depth_mm < MIN_DEPTH_MM or depth_mm > MAX_DEPTH_MM:
            roi = depth_data[
                max(0, y_px - 2) : min(depth_data.shape[0], y_px + 3),
                max(0, x_px - 2) : min(depth_data.shape[1], x_px + 3),
            ]
            valid = roi[(roi > 0) & (roi * depth_scale >= MIN_DEPTH_MM) & (roi * depth_scale <= MAX_DEPTH_MM)]
            depth_mm = float(np.median(valid)) * depth_scale if valid.size else 0.0
        pt3 = pixel_to_3d_in_color_frame(x_px, y_px, depth_mm, frame_data) if depth_mm > 0 else None
        all_items.append({"label": label, "pixel": (x_px, y_px), "point_3d_mm": pt3})
        if pt3 is not None:
            candidates.append({"label": label, "pixel": (x_px, y_px), "point_3d_mm": pt3})

    if not candidates:
        print("detect_hole_3d_one_shot: 无有效深度 3D 点")
        if save_debug_image and all_items:
            _save_detect_debug_image(color_bgr, all_items, None, debug_output_dir)
        return None

    filtered = [c for c in candidates if any(kw in (c.get("label") or "").lower() for kw in labels_filter)]
    if not filtered:
        filtered = candidates

    # 相机系 3D 为 (x, y, z)，point_3d_mm[0]=x, [1]=y, [2]=z
    if selection == "min_x":
        chosen = min(filtered, key=lambda c: c["point_3d_mm"][0])
    elif selection == "max_x":
        chosen = max(filtered, key=lambda c: c["point_3d_mm"][0])
    elif prefer_label:
        chosen = None
        for c in filtered:
            if prefer_label in (c.get("label") or ""):
                chosen = c
                break
        if chosen is None:
            chosen = filtered[0]
    else:
        chosen = filtered[0]

    pt3 = chosen["point_3d_mm"]
    chosen_pixel = chosen.get("pixel")
    print(f"detect_hole_3d_one_shot: 选用 {chosen['label']} -> ({pt3[0]:.1f}, {pt3[1]:.1f}, {pt3[2]:.1f}) mm")

    if save_debug_image and all_items:
        _save_detect_debug_image(color_bgr, all_items, chosen_pixel, debug_output_dir)

    return tuple(pt3)


def _save_detect_debug_image(
    color_bgr: np.ndarray,
    all_items: list[dict],
    chosen_pixel: tuple[int, int] | None,
    output_dir: str | None,
) -> None:
    """将检测结果标注到图上并保存，文件名带时间戳便于多次 debug 对比。"""
    from datetime import datetime
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detect_debug_{timestamp}.jpg"
    output_path = os.path.join(output_dir, filename)
    # 确保为 uint8 且值域 0–255，避免因 dtype/范围错误导致保存为黑图
    img = np.asarray(color_bgr)
    if img.size == 0:
        print("detect_debug: 图像为空，跳过保存")
        return
    if img.dtype != np.uint8:
        if img.max() <= 1.0 and img.min() >= 0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if np.all(img == 0):
        print("detect_debug: 警告：图像全黑，可能为帧缓冲已释放或相机未正确曝光，仍保存供排查")
    draw_and_save_annotated_image(img, all_items, output_path, chosen_pixel=chosen_pixel)


def run_camera_gemini_3d():
    """主流程：相机 D2C 对齐 -> 按空格截帧 -> Gemini 2D -> 取深度 -> 输出 3D。"""
    pipeline = Pipeline()
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    except Exception as e:
        print("流配置错误:", e)
        return

    pipeline.start(config)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    # 从 pipeline 的 stream profile 一次性取标定（避免帧 profile 不可用）
    pipeline_calibration = None
    try:
        pipeline_calibration = get_calibration_from_profiles(color_profile, depth_profile)
        print("已从 pipeline 获取内参/外参")
    except Exception as e:
        print("从 pipeline 获取标定失败:", e)
        print("将尝试从每帧获取（若仍失败请检查相机固件/驱动）")

    # if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyDTmotxn0msTnP8fbv9hBPiXdjfVkTP4RA":
    if not GEMINI_API_KEY:
        print("请设置 GEMINI_API_KEY 或修改脚本中的 GEMINI_API_KEY")
        pipeline.stop()
        return

    window_name = "Orbbec + Gemini 2D->3D (Space=截帧并检测, q=退出)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_color_bgr = None
    last_depth_frame = None
    last_frame_data = None

    while True:
        frames = pipeline.wait_for_frames(1000)
        if not frames:
            continue
        color_frame_raw = frames.get_color_frame()
        depth_frame_raw = frames.get_depth_frame()
        if not color_frame_raw or not depth_frame_raw:
            continue
        frames = align_filter.process(frames)
        if not frames:
            continue
        frames = frames.as_frame_set()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_bgr = frame_to_bgr_image(color_frame)
        if color_bgr is None:
            continue

        last_color_bgr = color_bgr.copy()
        last_depth_frame = depth_frame
        # 标定：优先用 pipeline 启动时取的，否则每帧从原始帧取
        if pipeline_calibration is not None:
            last_frame_data = dict(pipeline_calibration)
            last_frame_data["depth_data"] = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            ).reshape(depth_frame.get_height(), depth_frame.get_width())
            last_frame_data["depth_scale"] = depth_frame.get_depth_scale()
            last_frame_data["depth_width"] = depth_frame.get_width()
            last_frame_data["depth_height"] = depth_frame.get_height()
        else:
            try:
                last_frame_data = get_frame_data(color_frame_raw, depth_frame_raw)
                if last_frame_data is not None:
                    last_frame_data["depth_data"] = np.frombuffer(
                        depth_frame.get_data(), dtype=np.uint16
                    ).reshape(depth_frame.get_height(), depth_frame.get_width())
                    last_frame_data["depth_scale"] = depth_frame.get_depth_scale()
                    last_frame_data["depth_width"] = depth_frame.get_width()
                    last_frame_data["depth_height"] = depth_frame.get_height()
            except Exception:
                last_frame_data = None

        # 取深度图用于显示（可选）；预览缩小可提高帧率
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width()
        )
        depth_scale = depth_frame.get_depth_scale()
        depth_mm = depth_data.astype(np.float32) * depth_scale
        depth_mm = np.where(
            (depth_mm > MIN_DEPTH_MM) & (depth_mm < MAX_DEPTH_MM),
            depth_mm, 0
        )
        depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
        show_color, show_depth = color_bgr, depth_vis
        if PREVIEW_SCALE != 1.0 and PREVIEW_SCALE > 0:
            w, h = int(color_bgr.shape[1] * PREVIEW_SCALE), int(color_bgr.shape[0] * PREVIEW_SCALE)
            show_color = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            show_depth = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_LINEAR)
        show = cv2.addWeighted(show_color, 0.6, show_depth, 0.4, 0)
        cv2.putText(
            show, "Space: capture & Gemini 2D->3D | q: quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.imshow(window_name, show)

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break
        if key != ord(" "):
            continue

        # 空格：截帧 + Gemini + 3D
        if last_frame_data is None:
            print("无法获取内参/外参，跳过本帧")
            continue
        print("正在请求 Gemini 2D 检测...")
        points_2d = request_gemini_2d_points(last_color_bgr, GEMINI_API_KEY)
        if not points_2d:
            print("未解析到 2D 点")
            continue

        h, w = last_color_bgr.shape[:2]
        depth_data = last_frame_data["depth_data"]
        depth_scale = last_frame_data["depth_scale"]
        results = []
        all_items = []  # 含成功与失败，用于标注图
        for item in points_2d:
            point = item.get("point")
            label = item.get("label", "")
            if not point or len(point) < 2:
                continue
            y_norm, x_norm = float(point[0]), float(point[1])
            x_px = int((x_norm / 1000.0) * w)
            y_px = int((y_norm / 1000.0) * h)
            x_px = max(0, min(x_px, w - 1))
            y_px = max(0, min(y_px, h - 1))

            raw = depth_data[y_px, x_px]
            depth_mm = float(raw) * depth_scale
            if depth_mm <= 0 or depth_mm < MIN_DEPTH_MM or depth_mm > MAX_DEPTH_MM:
                roi = depth_data[
                    max(0, y_px - 2) : min(depth_data.shape[0], y_px + 3),
                    max(0, x_px - 2) : min(depth_data.shape[1], x_px + 3),
                ]
                valid = roi[(roi > 0) & (roi * depth_scale >= MIN_DEPTH_MM) & (roi * depth_scale <= MAX_DEPTH_MM)]
                depth_mm = float(np.median(valid)) * depth_scale if valid.size else 0.0

            pt3 = pixel_to_3d_in_color_frame(x_px, y_px, depth_mm, last_frame_data) if depth_mm > 0 else None
            if pt3 is not None:
                results.append({"label": label, "pixel": (x_px, y_px), "depth_mm": depth_mm, "point_3d_mm": pt3})
                all_items.append({"label": label, "pixel": (x_px, y_px), "point_3d_mm": pt3})
                print(f"  {label}: pixel=({x_px},{y_px}) depth={depth_mm:.1f}mm -> 3D(彩色系) = ({pt3[0]:.2f}, {pt3[1]:.2f}, {pt3[2]:.2f}) mm")
            else:
                all_items.append({"label": label, "pixel": (x_px, y_px), "point_3d_mm": None})
                print(f"  {label}: pixel=({x_px},{y_px}) depth={depth_mm:.1f}mm -> 无效深度/转换失败")

        if all_items:
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_ANNOTATED_IMAGE)
            draw_and_save_annotated_image(last_color_bgr, all_items, out_path)
        if results:
            print("--- 机械臂目标点（彩色相机坐标系，单位 mm）---")
            for r in results:
                print(f"  {r['label']}: x={r['point_3d_mm'][0]:.3f} y={r['point_3d_mm'][1]:.3f} z={r['point_3d_mm'][2]:.3f}")

    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    run_camera_gemini_3d()
