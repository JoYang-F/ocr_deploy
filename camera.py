"""
海康彩色工业相机 / 电脑摄像头 图像采集工具

支持两种来源（通过 --source 指定）：
  gige   — 海康工业相机，原厂 Bayer 格式转 BGR 彩色（默认）
  webcam — 电脑摄像头，直接输出 BGR 彩色

功能：
  - 实时预览彩色相机画面
  - 按 s 键：单张抓取，保存到 ./captures/
  - 按 空格：切换自动连续保存模式
  - 按 q 键：退出

配合使用：
  python camera.py --source webcam                    # 采集 webcam 图片
  python inference.py -c config.yml -i ./captures/    # 对采集的图片批量 OCR
"""

import os
import sys
import time
import argparse
from ctypes import c_ubyte, cast, POINTER
from datetime import datetime

import cv2
import numpy as np

from MvImport.MvCameraControl_class import *

# ============================================================
# 配置
# ============================================================
SAVE_DIR = "./captures"          # 图片保存目录
JPEG_QUALITY = 95                # JPEG 保存质量 (1-100)
PREVIEW_SCALE = 0.5              # 预览窗口缩放比例
WEBCAM_WIDTH = 1280              # 电脑摄像头默认分辨率宽度
WEBCAM_HEIGHT = 720              # 电脑摄像头默认分辨率高度


# ============================================================
# 相机抽象基类
# ============================================================

class CameraCapture:
    """统一相机采集接口，所有相机实现必须提供相同方法。"""

    def start(self):
        raise NotImplementedError

    def get_frame(self) -> tuple[bool, np.ndarray]:
        """获取一帧 BGR 图像。子类实现。"""
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


# ============================================================
# 电脑摄像头实现
# ============================================================

class WebcamCapture(CameraCapture):
    """通过 OpenCV 打开电脑摄像头（USB webcam），直接输出 BGR 彩色图像。"""

    def __init__(self, device: int = 0, width: int = WEBCAM_WIDTH, height: int = WEBCAM_HEIGHT):
        self.device = device
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    @property
    def name(self) -> str:
        return "Webcam"

    def start(self):
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 device={self.device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 尝试设置 30fps
        self._cap.set(cv2.CAP_PROP_FPS, 30)

    def get_frame(self) -> tuple[bool, np.ndarray]:
        if self._cap is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        ret, frame = self._cap.read()
        return ret, frame

    def stop(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ============================================================
# 海康工业相机实现（保留原有逻辑）
# ============================================================

class HikCapture(CameraCapture):
    """海康工业相机，通过 MvImport SDK 采集并转换为 BGR 彩色。"""

    @property
    def name(self) -> str:
        return "Hikvision"

    def start(self):
        self._cam = MvCamera()
        MvCamera.MV_CC_Initialize()

        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = self._cam.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            raise RuntimeError("未找到海康相机！")

        stDeviceList = deviceList.pDeviceInfo[0].contents
        ret = self._cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            raise RuntimeError(f"创建句柄失败! 错误码: 0x{ret:08X}")

        ret = self._cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"打开设备失败! 错误码: 0x{ret:08X}")

        self._cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        stParam = MVCC_INTVALUE()
        self._cam.MV_CC_GetIntValue("PayloadSize", stParam)
        self._payload_size = stParam.nCurValue
        self._data_buf = (c_ubyte * self._payload_size)()

        ret = self._cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"开始取流失败! 错误码: 0x{ret:08X}")

    def get_frame(self) -> tuple[bool, np.ndarray]:
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ret = self._cam.MV_CC_GetOneFrameTimeout(
            self._data_buf, self._payload_size, stFrameInfo, 1000
        )
        if ret != 0:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        h, w = stFrameInfo.nHeight, stFrameInfo.nWidth

        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
        stConvertParam.nWidth = w
        stConvertParam.nHeight = h
        stConvertParam.pSrcData = cast(self._data_buf, POINTER(c_ubyte))
        stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
        stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed

        dst_buf_size = w * h * 3
        dst_buf = (c_ubyte * dst_buf_size)()
        stConvertParam.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
        stConvertParam.nDstBufferSize = dst_buf_size

        ret_convert = self._cam.MV_CC_ConvertPixelType(stConvertParam)
        if ret_convert != 0:
            print(f"像素格式转换失败! 错误码: 0x{ret_convert:08X}")
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        img_array = np.frombuffer(dst_buf, count=dst_buf_size, dtype=np.uint8)
        image_bgr = img_array.reshape((h, w, 3))
        return True, image_bgr

    def stop(self):
        self._cam.MV_CC_StopGrabbing()
        self._cam.MV_CC_CloseDevice()
        self._cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp_filename(ext=".jpg"):
    """生成带毫秒时间戳的文件名，避免重名"""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}{ext}"


def draw_status(img: np.ndarray, text_lines: list):
    """在图像左上角绘制半透明状态信息（原地修改 img）"""
    overlay = img.copy()
    y0 = 35
    for i, line in enumerate(text_lines):
        y = y0 + i * 30
        cv2.putText(overlay, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(overlay, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="相机图像采集工具")
    parser.add_argument(
        "--source",
        choices=["gige", "webcam"],
        default="gige",
        help="采集来源: gige=海康工业相机(默认), webcam=电脑摄像头",
    )
    args = parser.parse_args()

    ensure_dir(SAVE_DIR)

    # ---- 创建相机实例 ----
    if args.source == "webcam":
        cam: CameraCapture = WebcamCapture()
        print(f"[{cam.name}] 正在打开摄像头 ...")
    else:
        cam = HikCapture()
        print(f"[{cam.name}] 正在初始化工业相机 ...")

    try:
        cam.start()
    except RuntimeError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    print("=" * 50)
    print(f"  相机采集已启动 ({cam.name})")
    print("  s     — 单张抓取保存")
    print("  空格  — 切换自动连续保存")
    print("  q     — 退出")
    print(f"  保存目录: {SAVE_DIR}/")
    print("=" * 50)

    auto_save = False        # 自动连续保存开关
    saved_count = 0           # 本次运行已保存张数
    fps_t0 = time.time()
    fps_counter = 0
    fps_text = "FPS: --"
    image_bgr = None           # 当前帧，供 s 键抓取使用

    while True:
        ret, frame = cam.get_frame()

        if ret:
            try:
                image_bgr = frame
                h, w = image_bgr.shape[:2]

                # FPS 统计
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed = time.time() - fps_t0
                    fps_text = f"FPS: {fps_counter / elapsed:.1f}"
                    fps_t0 = time.time()
                    fps_counter = 0

                # 自动保存模式
                if auto_save:
                    fname = timestamp_filename()
                    fpath = os.path.join(SAVE_DIR, fname)
                    cv2.imwrite(fpath, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    saved_count += 1
                    print(f"[自动保存] {fname}  (累计 {saved_count} 张)")

                # 预览（叠加状态信息到副本，原图保持干净用于保存）
                auto_label = "ON" if auto_save else "OFF"
                status = [
                    f"Saves: {saved_count} | Auto: {auto_label}",
                    fps_text,
                    "s:Grab  Space:Auto  q:Quit",
                ]
                preview_frame = image_bgr.copy()
                draw_status(preview_frame, status)

                pw, ph = int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)
                preview = cv2.resize(preview_frame, (pw, ph))
                cv2.imshow(f"{cam.name} Capture", preview)

            except Exception as e:
                print(f"帧处理异常: {e}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 单张抓取
            if image_bgr is not None:
                fname = timestamp_filename()
                fpath = os.path.join(SAVE_DIR, fname)
                cv2.imwrite(fpath, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                saved_count += 1
                print(f"[单张抓取] {fname}  (累计 {saved_count} 张)")
        elif key == ord(' '):
            auto_save = not auto_save
            state = "开启" if auto_save else "关闭"
            print(f"[自动保存] {state}")

    # ---- 清理 ----
    cam.stop()
    cv2.destroyAllWindows()
    print(f"已退出，本次共保存 {saved_count} 张图片到 {os.path.abspath(SAVE_DIR)}/")


if __name__ == "__main__":
    main()