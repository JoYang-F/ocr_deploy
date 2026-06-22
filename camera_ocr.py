"""
海康工业相机实时OCR检测识别脚本

架构：双线程 Producer-Consumer
  - 采集线程：持续从海康相机取帧 → Bayer→BGR 转换 → 写入共享帧缓存
  - OCR 线程：从共享帧缓存取最新帧 → DBNet 检测 → CRNN 识别 → 写入共享结果缓存
  - 主线程：取最新帧 + 最新结果 → 叠加检测框/文本 → 实时显示

依赖：MvImport (海康 SDK)、inference (OCR 引擎)

"""

import os
import sys
import time
import queue
import threading
import argparse
from ctypes import c_ubyte, cast, POINTER
from datetime import datetime

import cv2
import numpy as np
import yaml

from MvImport.MvCameraControl_class import *
from inference import OCRInference


def _resolve_cfg_path(rel_path: str) -> str:
    """将相对路径解析为脚本所在目录的绝对路径。"""
    if os.path.isabs(rel_path):
        return rel_path
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, rel_path))


# ============================================================
# 海康相机（自包含）
# ============================================================

class HikCamera:
    """海康工业相机：初始化 SDK → 枚举 → 创建句柄 → 取流 → 像素格式转换（Bayer→BGR）。"""

    def __init__(self):
        self._cam = None
        self._payload_size = 0
        self._data_buf = None
        self._width = 0
        self._height = 0

    # ---------- 属性 ----------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ---------- 生命周期 ----------

    def start(self):
        """初始化 SDK、枚举设备、打开第一台相机、开始取流。"""

        # 1. SDK 初始化
        MvCamera.MV_CC_Initialize()

        # 2. 创建相机实例并枚举设备
        self._cam = MvCamera()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = self._cam.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            raise RuntimeError("未找到海康相机！请检查相机连接和电源。")

        # 3. 创建句柄
        stDevInfo = deviceList.pDeviceInfo[0].contents
        ret = self._cam.MV_CC_CreateHandle(stDevInfo)
        if ret != 0:
            raise RuntimeError(f"创建句柄失败! 错误码: 0x{ret:08X}")

        # 4. 打开设备
        ret = self._cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self._cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"打开设备失败! 错误码: 0x{ret:08X}")

        # 5. 连续采集模式
        self._cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        # 6. 获取 PayloadSize 并分配接收缓存
        stParam = MVCC_INTVALUE()
        self._cam.MV_CC_GetIntValue("PayloadSize", stParam)
        self._payload_size = stParam.nCurValue
        self._data_buf = (c_ubyte * self._payload_size)()

        # 7. 取第一帧以获取分辨率
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != 0:
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"开始取流失败! 错误码: 0x{ret:08X}")

    def get_frame(self) -> tuple:
        """获取一帧 BGR 彩色图像。

        Returns:
            (success: bool, image: np.ndarray) — BGR (H,W,3) uint8
        """
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ret = self._cam.MV_CC_GetOneFrameTimeout(
            self._data_buf, self._payload_size, stFrameInfo, 1000
        )
        if ret != 0:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
        self._width, self._height = w, h

        # 通过 SDK 内置转换接口将任意像素格式转为 BGR8_Packed
        stParam = MV_CC_PIXEL_CONVERT_PARAM()
        stParam.nWidth = w
        stParam.nHeight = h
        stParam.enSrcPixelType = stFrameInfo.enPixelType
        stParam.pSrcData = cast(self._data_buf, POINTER(c_ubyte))
        stParam.nSrcDataLen = stFrameInfo.nFrameLen
        stParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed

        dst_size = w * h * 3
        dst_buf = (c_ubyte * dst_size)()
        stParam.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
        stParam.nDstBufferSize = dst_size

        ret = self._cam.MV_CC_ConvertPixelType(stParam)
        if ret != 0:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        image_bgr = np.frombuffer(dst_buf, count=dst_size, dtype=np.uint8).reshape((h, w, 3))
        return True, image_bgr

    def stop(self):
        """停止取流、关闭设备、反初始化 SDK。"""
        if self._cam is not None:
            self._cam.MV_CC_StopGrabbing()
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()


# ============================================================
# 工具函数
# ============================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    """生成毫秒级时间戳字符串，避免文件名冲突。"""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"


def _draw_status(image: np.ndarray, lines: list):
    """在图像左上角叠加半透明状态信息（原地修改）。"""
    overlay = image.copy()
    for i, text in enumerate(lines):
        y = 35 + i * 28
        cv2.putText(overlay, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)


def _draw_ocr_results(image: np.ndarray, results: list):
    """在图像上绘制 OCR 检测框和识别文本（原地修改）。"""
    for r in results:
        box = np.array(r["bbox"], dtype=np.int32)
        cv2.polylines(image, [box.reshape(-1, 1, 2)], isClosed=True,
                      color=(0, 0, 255), thickness=2)
        label = f"{r['text']} {r['score']:.2f}"
        x, y = int(box[0][0]), int(box[0][1]) - 8
        # 文字底色
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(image, (x, y - th - 2), (x + tw + 4, y + 2),
                      (0, 0, 255), -1)
        cv2.putText(image, label, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)


# ============================================================
# 线程函数
# ============================================================

def _capture_loop(cam: HikCamera, frame_lock: threading.Lock,
                  ocr_event: threading.Event, stop_event: threading.Event,
                  stats: dict):
    """采集线程：持续从海康相机取帧，写入共享缓存，通知 OCR 线程。"""
    fps_t0 = time.time()
    fps_count = 0

    while not stop_event.is_set():
        ok, frame = cam.get_frame()
        if not ok:
            time.sleep(0.001)
            continue

        # 写入共享帧（主线程显示用）
        with frame_lock:
            stats["latest_frame"] = frame.copy()
            stats["frame_count"] += 1

        # 通知 OCR 线程有新帧可处理
        with frame_lock:
            stats["ocr_pending_frame"] = frame.copy()
        ocr_event.set()

        # 采集 FPS
        fps_count += 1
        if fps_count >= 30:
            elapsed = time.time() - fps_t0
            stats["capture_fps"] = fps_count / elapsed
            fps_t0 = time.time()
            fps_count = 0

    cam.stop()


def _ocr_loop(ocr: OCRInference, frame_lock: threading.Lock,
              ocr_event: threading.Event, stop_event: threading.Event,
              stats: dict):
    """OCR 线程：等待新帧 → 执行 detection + recognition → 写回共享结果。"""
    while not stop_event.is_set():
        # 等待采集线程通知，超时 200ms 以响应 stop_event
        signaled = ocr_event.wait(timeout=0.2)
        if not signaled:
            continue
        ocr_event.clear()

        # 取最新待处理帧
        with frame_lock:
            frame = stats.pop("ocr_pending_frame", None)

        if frame is None:
            continue

        try:
            t0 = time.time()
            results = ocr.predict(frame)
            t_cost = time.time() - t0

            stats["ocr_fps"] = 1.0 / max(t_cost, 0.001)
            stats["ocr_latency_ms"] = t_cost * 1000

            with frame_lock:
                stats["latest_results"] = results
                stats["ocr_frame"] = frame  # 对应结果的原帧（供保存用）

        except Exception as e:
            stats["ocr_error"] = str(e)


# ============================================================
# 主流程
# ============================================================

def main():
    # ---- 解析 CLI 参数 ----
    parser = argparse.ArgumentParser(description="海康工业相机 实时 OCR 检测识别")
    parser.add_argument("-c", "--config", default="config.yml",
                        help="OCR 配置文件路径（默认 config.yml）")
    parser.add_argument("--display-scale", type=float, default=None,
                        help="预览缩放比例（覆盖 config.yml 中的 capture.display_scale）")
    parser.add_argument("--save-dir", default=None,
                        help="检出文本时图片保存目录（覆盖 config.yml 中的 capture.save_dir）")
    parser.add_argument("--no-auto-save", action="store_true",
                        help="关闭检出文本时自动保存图片（覆盖 config.yml）")
    args = parser.parse_args()

    # ---- 读取配置文件 ----
    config_path = _resolve_cfg_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    capture_cfg = cfg.get("capture", {})

    # 配置值优先顺序: CLI 参数 > YAML > 默认值
    save_dir = args.save_dir or capture_cfg.get("save_dir", "./captures_ocr")
    jpeg_quality = capture_cfg.get("jpeg_quality", 95)
    display_scale = args.display_scale or capture_cfg.get("display_scale", 0.5)
    capture_auto_save = capture_cfg.get("auto_save", True)  # YAML 中的自动保存配置

    _ensure_dir(save_dir)

    # ========== 加载 OCR 模型 ==========
    print("=== 海康相机实时 OCR 检测识别 ===")
    print(f"配置文件: {config_path}")
    print(f"保存目录: {save_dir}  |  预览缩放: {display_scale}")
    print("正在加载 OCR 模型 ...")
    try:
        ocr = OCRInference(config_path)
    except Exception as e:
        print(f"[错误] OCR 模型加载失败: {e}")
        sys.exit(1)

    det_prov = ocr.det.sess.get_providers()
    rec_prov = ocr.rec.sess.get_providers()
    print(f"模型加载完成！Det: {det_prov[0]}, Rec: {rec_prov[0]}, "
          f"字符集: {len(ocr.rec)} 类")

    # ========== 初始化海康相机 ==========
    cam = HikCamera()
    print("正在初始化海康相机 ...")
    try:
        cam.start()
    except RuntimeError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    print(f"相机就绪: {cam.width}x{cam.height}")
    print("=" * 55)
    print("  q — 退出        s — 手动抓取保存")
    print("  r — 切换 OCR    a — 切换自动保存")
    print(f"  保存目录: {save_dir}/")
    print("=" * 55)

    # ========== 共享状态 ==========
    frame_lock = threading.Lock()
    ocr_event = threading.Event()
    stop_event = threading.Event()

    stats = {
        "latest_frame": None,        # 最新采集帧（主线程显示用）
        "latest_results": [],        # 最新 OCR 结果列表
        "ocr_frame": None,           # OCR 结果对应的原帧
        "ocr_pending_frame": None,   # 待 OCR 处理的最新帧
        "frame_count": 0,            # 累计采集帧数
        "capture_fps": 0.0,          # 采集帧率
        "ocr_fps": 0.0,              # OCR 推理帧率
        "ocr_latency_ms": 0.0,       # OCR 单次推理耗时 (ms)
        "ocr_error": None,           # OCR 异常信息
    }

    # ========== 启动线程 ==========
    capture_thread = threading.Thread(
        target=_capture_loop,
        args=(cam, frame_lock, ocr_event, stop_event, stats),
        daemon=True, name="Capture"
    )
    ocr_thread = threading.Thread(
        target=_ocr_loop,
        args=(ocr, frame_lock, ocr_event, stop_event, stats),
        daemon=True, name="OCR"
    )
    capture_thread.start()
    ocr_thread.start()

    # ========== 主循环：显示 + 按键 ==========
    ocr_enabled = True              # OCR 开关
    auto_save_enabled = (not args.no_auto_save) and capture_auto_save  # 自动保存开关
    saved_count = 0
    last_saved_texts = set()        # 去重：避免同一帧反复保存

    try:
        while True:
            # 取最新帧和最新结果
            with frame_lock:
                frame = stats["latest_frame"]
                results = stats["latest_results"] if ocr_enabled else []
                ocr_frame_for_save = stats.get("ocr_frame")

            # 显示
            if frame is not None:
                h, w = frame.shape[:2]
                show = frame.copy()

                # 叠加 OCR 结果
                if results:
                    _draw_ocr_results(show, results)

                # 自动保存：检出文本时保存（去重，每批次只存一次）
                if auto_save_enabled and ocr_enabled and results and ocr_frame_for_save is not None:
                    texts_key = frozenset(r["text"] for r in results)
                    if texts_key != last_saved_texts:
                        last_saved_texts = texts_key
                        ts = _timestamp()
                        img_path = os.path.join(save_dir, f"{ts}.jpg")
                        txt_path = os.path.join(save_dir, f"{ts}.txt")
                        cv2.imwrite(img_path, ocr_frame_for_save,
                                    [cv2.IMWRITE_jpeg_quality, jpeg_quality])
                        with open(txt_path, "w", encoding="utf-8") as f:
                            for r in results:
                                f.write(f"{r['text']}\t{r['score']:.3f}\n")
                        saved_count += 1
                        print(f"[自动保存] {ts}.jpg ({len(results)} 条文本, 累计 {saved_count})")

                # 状态信息
                ocr_label = "ON" if ocr_enabled else "OFF"
                auto_label = "ON" if auto_save_enabled else "OFF"
                lines = [
                    f"Capture FPS: {stats['capture_fps']:.1f}  |  "
                    f"OCR FPS: {stats['ocr_fps']:.1f}  |  Latency: {stats['ocr_latency_ms']:.0f}ms",
                    f"Saves: {saved_count}  |  OCR: {ocr_label}  |  AutoSave: {auto_label}",
                    "q:Quit  s:Snap  r:OCR  a:AutoSave",
                ]
                _draw_status(show, lines)

                # 缩放并显示
                pw, ph = int(w * display_scale), int(h * display_scale)
                preview = cv2.resize(show, (pw, ph))
                cv2.imshow("Hikvision OCR", preview)

            # 显示 OCR 错误
            if stats.get("ocr_error"):
                print(f"[OCR 异常] {stats['ocr_error']}")
                stats["ocr_error"] = None

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 手动抓取
                with frame_lock:
                    snap = stats["latest_frame"]
                if snap is not None:
                    ts = _timestamp()
                    fpath = os.path.join(save_dir, f"snap_{ts}.jpg")
                    cv2.imwrite(fpath, snap, [cv2.IMWRITE_jpeg_quality, jpeg_quality])
                    saved_count += 1
                    print(f"[手动抓取] snap_{ts}.jpg (累计 {saved_count})")
            elif key == ord('r'):
                ocr_enabled = not ocr_enabled
                state = "开启" if ocr_enabled else "关闭"
                print(f"[OCR] {state}")
            elif key == ord('a'):
                auto_save_enabled = not auto_save_enabled
                state = "开启" if auto_save_enabled else "关闭"
                print(f"[自动保存] {state}")

    finally:
        # ========== 清理 ==========
        print("正在停止线程 ...")
        stop_event.set()
        ocr_event.set()  # 唤醒 OCR 线程使其检测 stop_event
        capture_thread.join(timeout=3.0)
        ocr_thread.join(timeout=3.0)
        cv2.destroyAllWindows()
        print(f"已退出。本次共保存 {saved_count} 张图片。")
        if saved_count > 0:
            print(f"图片目录: {os.path.abspath(save_dir)}/")


if __name__ == "__main__":
    main()