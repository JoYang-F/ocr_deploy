"""
海康彩色工业相机图像采集工具（支持原厂 Bayer 格式转 BGR 彩色）

功能：
  - 实时预览彩色相机画面
  - 按 s 键：单张抓取，保存到 ./captures/
  - 按 空格：切换自动连续保存模式
  - 按 q 键：退出

配合使用：
  python inference.py -c config.yml -i ./captures/    # 对采集的图片批量 OCR
"""

import os
import time
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
    """在图像左上角绘制半透明状态信息"""
    overlay = img.copy()
    y0 = 35  # 【关键修改】把起始 Y 坐标从 10 改为 35，给第一行文字顶部留出空间
    for i, line in enumerate(text_lines):
        y = y0 + i * 30  # 行间距设为 30
        cv2.putText(overlay, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(overlay, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)


# ============================================================
# 主流程
# ============================================================

def main():
    ensure_dir(SAVE_DIR)

    # ---- 初始化相机 ----
    cam = MvCamera()
    MvCamera.MV_CC_Initialize()

    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = cam.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("未找到海康相机！")
        MvCamera.MV_CC_Finalize()
        return

    stDeviceList = deviceList.pDeviceInfo[0].contents
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"创建句柄失败! 错误码: 0x{ret:08X}")
        MvCamera.MV_CC_Finalize()
        return

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"打开设备失败! 错误码: 0x{ret:08X}")
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    stParam = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    payload_size = stParam.nCurValue
    data_buf = (c_ubyte * payload_size)()

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"开始取流失败! 错误码: 0x{ret:08X}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return

    print("=" * 50)
    print("  相机采集已启动 (支持彩色解析)")
    print("  s     — 单张抓取保存")
    print("  空格  — 切换自动连续保存")
    print("  q     — 退出")
    print(f"  保存目录: {SAVE_DIR}/")
    print("=" * 50)

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    auto_save = False        # 自动连续保存开关
    saved_count = 0           # 本次运行已保存张数
    fps_t0 = time.time()
    fps_counter = 0
    fps_text = "FPS: --"

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, stFrameInfo, 1000)

        if ret == 0:
            try:
                h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
                
                # ========================================================
                # 核心修改区：调用原厂 API 将任意格式转换为 BGR 彩色
                # ========================================================
                # 1. 准备转换参数结构体
                stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                stConvertParam.nWidth = w
                stConvertParam.nHeight = h
                stConvertParam.pSrcData = cast(data_buf, POINTER(c_ubyte))
                stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
                
                # 目标格式设为 BGR8_Packed (OpenCV 的默认彩色格式)
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                
                # 2. 分配目标内存 (宽 * 高 * 3个通道)
                dst_buf_size = w * h * 3
                dst_buf = (c_ubyte * dst_buf_size)()
                stConvertParam.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
                stConvertParam.nDstBufferSize = dst_buf_size
                
                # 3. 执行转换
                ret_convert = cam.MV_CC_ConvertPixelType(stConvertParam)
                
                if ret_convert == 0:
                    # 转换成功，直接转为 numpy 数组并 reshape
                    img_array = np.frombuffer(dst_buf, count=dst_buf_size, dtype=np.uint8)
                    image_bgr = img_array.reshape((h, w, 3))
                else:
                    print(f"像素格式转换失败! 错误码: 0x{ret_convert:08X}")
                    continue
                # ========================================================

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

                # 预览
                auto_label = "ON" if auto_save else "OFF"
                status = [
                    f"Saves: {saved_count} | Auto: {auto_label}",
                    fps_text,
                    "s:Grab  Space:Auto  q:Quit",
                ]
                draw_status(image_bgr, status)

                pw, ph = int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)
                preview = cv2.resize(image_bgr, (pw, ph))
                cv2.imshow("Hikvision Capture", preview)

            except Exception as e:
                print(f"帧处理异常: {e}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 单张抓取
            try:
                fname = timestamp_filename()
                fpath = os.path.join(SAVE_DIR, fname)
                cv2.imwrite(fpath, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                saved_count += 1
                print(f"[单张抓取] {fname}  (累计 {saved_count} 张)")
            except NameError:
                pass  # 还没取到第一帧
        elif key == ord(' '):
            auto_save = not auto_save
            state = "开启" if auto_save else "关闭"
            print(f"[自动保存] {state}")

    # ---- 清理 ----
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()
    cv2.destroyAllWindows()
    print(f"已退出，本次共保存 {saved_count} 张图片到 {os.path.abspath(SAVE_DIR)}/")


if __name__ == "__main__":
    main()