"""
后处理模块（无 PyTorch 依赖，仅 numpy + opencv + pyclipper + shapely）

提供：
  - CharacterMapper: 字符 ↔ 索引 双向映射
  - DBPostProcess:  DBNet 概率图 → 文本框坐标
  - CTCDecoder:     CRNN CTC 输出 → 文本字符串
"""

import json
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


# ============================================================
# 字符映射
# ============================================================

class CharacterMapper:
    """字符 ↔ 索引 双向映射。

    从 JSON 文件加载字符集，格式：{"<BLANK>": 0, "A": 1, "B": 2, ...}
    """

    def __init__(self, char_json_path: str):
        with open(char_json_path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.blank_token = self.char2idx.get("<BLANK>", 0)

    def __len__(self) -> int:
        return len(self.char2idx)


# ============================================================
# DBNet 后处理：概率图 → 文本框
# ============================================================

class DBPostProcess:
    """DBNet 后处理：从概率图中提取文本框。

    步骤：
        1. 二值化（prob > thresh）
        2. findContours 找轮廓
        3. 对每个轮廓：最小外接矩形 → unclip 膨胀 → 坐标映射回原图
        4. 按 box_thresh 过滤低置信度框
    """

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.6,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def __call__(self, pred: np.ndarray, src_scale: np.ndarray):
        """
        Args:
            pred: 概率图 (1, 1, H, W) 或 (1, H, W)
            src_scale: 原始图像尺寸 (H, W)

        Returns:
            boxes: list of np.ndarray, 每个 (4, 2) int16
            scores: list of float
        """
        if pred.ndim == 4:
            pred = pred[0, 0, :, :]
        elif pred.ndim == 3:
            pred = pred[0, :, :]

        segmentation = pred > self.thresh
        orig_h, orig_w = int(src_scale[0]), int(src_scale[1])

        boxes, scores = self._boxes_from_bitmap(pred, segmentation, orig_w, orig_h)
        return boxes, scores

    def _boxes_from_bitmap(self, pred, bitmap, dest_w, dest_h):
        h, w = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for idx in range(num_contours):
            contour = contours[idx].squeeze(1)
            if contour.ndim != 2 or contour.shape[0] < 4:
                continue

            points, min_side = self._get_mini_boxes(contour)
            if min_side < self.min_size:
                continue

            score = self._box_score_fast(pred, contour)
            if score < self.box_thresh:
                continue

            # unclip 膨胀
            expanded = self._unclip(points, self.unclip_ratio).reshape(-1, 2)
            box, min_side = self._get_mini_boxes(expanded)
            if min_side < self.min_size + 2:
                continue

            # 映射回原图坐标
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)

            boxes.append(box.astype(np.int16))
            scores.append(score)

        return boxes, scores

    @staticmethod
    def _unclip(points, unclip_ratio):
        poly = Polygon(points)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    @staticmethod
    def _get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # 按左上→右上→右下→左下排序
        if points[1][1] > points[0][1]:
            idx_0, idx_3 = 0, 1
        else:
            idx_0, idx_3 = 1, 0
        if points[3][1] > points[2][1]:
            idx_1, idx_2 = 2, 3
        else:
            idx_1, idx_2 = 3, 2

        box = np.array([points[idx_0], points[idx_1], points[idx_2], points[idx_3]])
        return box, min(bounding_box[1])

    @staticmethod
    def _box_score_fast(bitmap, box):
        h, w = bitmap.shape[:2]
        box = box.copy()
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


# ============================================================
# CRNN CTC 解码：概率矩阵 → 文本
# ============================================================

class CTCDecoder:
    """CRNN CTC 解码器。

    将模型输出的 log_softmax 概率矩阵解码为文本。
    处理流程：exp → argmax → 去重 → 移除 <BLANK>
    """

    def __init__(self, char_mapper: CharacterMapper):
        self.char2idx = char_mapper.char2idx
        self.idx2char = char_mapper.idx2char
        self.ignored_tokens = {char_mapper.blank_token}

    def __call__(self, preds: np.ndarray) -> list:
        """
        Args:
            preds: CRNN 输出 (T, N, classes_num) log_softmax

        Returns:
            list of (text, confidence) tuples, 长度 = batch_size
        """
        # log_softmax → prob
        probs = np.exp(preds)

        # 沿类别维度取 argmax 和 max
        indices = probs.argmax(axis=2)   # (T, N)
        confs   = probs.max(axis=2)      # (T, N)

        # 转置为 (N, T)
        indices = indices.transpose(1, 0)
        confs   = confs.transpose(1, 0)

        results = []
        for batch_idx in range(indices.shape[0]):
            text, conf = self._decode_one(indices[batch_idx], confs[batch_idx])
            results.append((text, conf))
        return results

    def _decode_one(self, idx_seq: np.ndarray, conf_seq: np.ndarray) -> tuple:
        """解码单条序列：去重 + 移除 <BLANK>"""
        chars = []
        confs = []
        prev_idx = -1
        for i in range(len(idx_seq)):
            idx = int(idx_seq[i])
            if idx in self.ignored_tokens:
                prev_idx = -1
                continue
            if idx == prev_idx:  # CTC 去重
                continue
            chars.append(self.idx2char[idx])
            confs.append(float(conf_seq[i]))
            prev_idx = idx

        text = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return text, avg_conf
