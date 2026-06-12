"""
图像预处理模块（无 PyTorch 依赖，仅 numpy + opencv）

提供检测和识别两套预处理管线，与训练时完全一致：
  - DetPreprocess: ResizeForTest → NormalizeImage(ImageNet 统计量)
  - RecPreprocess: RecResizeImg → 等比缩放 + [-1,1] 归一化 + 右填充

注意：检测和识别使用不同的归一化参数！
"""

import math
import cv2
import numpy as np


# ============================================================
# 检测预处理：ResizeForTest + NormalizeImage(ImageNet)
# ============================================================

class DetPreprocess:
    """文本检测预处理管线。

    1. ResizeForTest: 将长边缩放到 long_size，尺寸对齐到 stride 的倍数
    2. NormalizeImage: ImageNet 均值/标准差归一化，HWC → CHW
    """

    # ImageNet 统计量（与训练时 operators.py 一致）
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, long_size: int = 960):
        """
        Args:
            long_size: 图片长边缩放目标尺寸
        """
        self.long_size = long_size

    def __call__(self, image: np.ndarray) -> dict:
        """
        Args:
            image: BGR 图像 (H, W, 3), uint8

        Returns:
            {"image": np.ndarray (1,3,H',W') float32,
             "src_scale": np.ndarray (2,) int32 [orig_H, orig_W]}
        """
        src_h, src_w = image.shape[:2]

        # ---- Step 1: ResizeForTest ----
        ratio = self.long_size / max(src_h, src_w)
        resize_h = int(src_h * ratio)
        resize_w = int(src_w * ratio)

        # 对齐到 stride=128（DBNet 下采样倍数的整数倍）
        stride = 128
        resize_h = (resize_h + stride - 1) // stride * stride
        resize_w = (resize_w + stride - 1) // stride * stride

        resized = cv2.resize(image, (resize_w, resize_h))

        # ---- Step 2: NormalizeImage (ImageNet stats) ----
        x = resized.astype(np.float32) * (1.0 / 255.0)
        x = (x - self.MEAN) / self.STD
        x = x.transpose(2, 0, 1)  # HWC → CHW

        # 添加 batch 维度
        x = np.expand_dims(x, axis=0).astype(np.float32)

        return {
            "image": x,
            "src_scale": np.array([src_h, src_w], dtype=np.int32),
        }


# ============================================================
# 识别预处理：RecResizeImg → [-1, 1] 归一化 + 右填充
# ============================================================

class RecPreprocess:
    """文本识别预处理管线。

    将裁剪后的文本行图像缩放到固定高度（默认 32），保持宽高比，
    归一化到 [-1, 1]，并右填充到 max_width。

    与训练时 data_loader/img_aug/rec_img_aug.py 的 resize_norm_img 完全一致。
    """

    def __init__(self, image_shape: tuple = (3, 32, 320)):
        """
        Args:
            image_shape: (C, H, max_W) 目标尺寸
        """
        self.img_c, self.img_h, self.img_w = image_shape

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: BGR 裁剪文本行 (h, w, 3), uint8

        Returns:
            np.ndarray (1, 3, self.img_h, self.img_w) float32, 值域 [-1, 1]
        """
        h, w = image.shape[:2]
        ratio = w / float(h)

        # 等比缩放到高度=img_h，宽度不超过 img_w
        if math.ceil(self.img_h * ratio) > self.img_w:
            resized_w = self.img_w
        else:
            resized_w = int(math.ceil(self.img_h * ratio))

        resized = cv2.resize(image, (resized_w, self.img_h))
        resized = resized.astype(np.float32)
        resized = resized.transpose(2, 0, 1) / 255.0
        resized -= 0.5
        resized /= 0.5  # → [-1, 1]

        # 右填充到 max_width
        padded = np.zeros((self.img_c, self.img_h, self.img_w), dtype=np.float32)
        padded[:, :, 0:resized_w] = resized

        # 添加 batch 维度
        return np.expand_dims(padded, axis=0).astype(np.float32)


# ============================================================
# 工具函数：透视裁剪（与原 lite_ocr.py 一致）
# ============================================================

def rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """根据四个角点做透视变换，裁出正立的文本行图像。

    Args:
        img: BGR 原图 (H, W, 3)
        points: 四个角点 (4, 2) float32, 顺序：左上→右上→右下→左下

    Returns:
        cropped: 透视校正后的 BGR 文本行图像
    """
    points = points.astype(np.float32)
    left   = int(np.min(points[:, 0]))
    right  = int(np.max(points[:, 0]))
    top    = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))

    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] -= left
    points[:, 1] -= top

    crop_w = int(np.linalg.norm(points[0] - points[1]))
    crop_h = int(np.linalg.norm(points[0] - points[3]))

    pts_std = np.float32([
        [0, 0],
        [crop_w, 0],
        [crop_w, crop_h],
        [0, crop_h],
    ])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(
        img_crop, M, (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 如果高度 ≥ 2*宽度，说明是竖排文字，旋转90°
    h, w = dst.shape[:2]
    if h * 1.0 / w >= 2:
        dst = np.rot90(dst)

    return dst
