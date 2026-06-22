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
    """CRNN CTC 解码器（最小修复版）。

    设计原则：
        1. 默认完全保留原代码的解码路径：输入按 (T, N, C) 处理；
        2. 默认仍按 log_softmax → exp 处理；
        3. 不跳过正常字符，避免身份证号这种连续数字被漏掉；
        4. 只在 idx 越界时做有限兜底，并打印清楚原因。

    如果你的 ONNX 输出不是 (T, N, C)，可以在初始化时显式指定 layout：
        CTCDecoder(char_mapper, layout="NTC")  # (N, T, C)
        CTCDecoder(char_mapper, layout="NCT")  # (N, C, T)
    """

    def __init__(
        self,
        char_mapper: CharacterMapper,
        layout: str = "TNC",
        from_log_softmax: bool = True,
        debug: bool = False,
        strict: bool = False,
    ):
        self.char2idx = char_mapper.char2idx
        self.idx2char = {int(k): v for k, v in char_mapper.idx2char.items()}
        self.blank_token = int(char_mapper.blank_token)
        self.ignored_tokens = {self.blank_token}

        self.layout = layout.upper()
        self.from_log_softmax = from_log_softmax
        self.debug = debug
        self.strict = strict
        self._warned_missing = False

    def __call__(self, preds: np.ndarray) -> list:
        """
        Args:
            preds: CRNN 输出。默认按 (T, N, C) log_softmax 处理。

        Returns:
            list of (text, confidence) tuples, 长度 = batch_size
        """
        preds = np.asarray(preds)
        preds = self._normalize_layout(preds)

        # 默认保持原来的行为：log_softmax → prob
        if self.from_log_softmax:
            probs = np.exp(preds)
        else:
            probs = self._softmax(preds, axis=2)

        indices = probs.argmax(axis=2)   # (T, N)
        confs = probs.max(axis=2)        # (T, N)

        # 转置为 (N, T)，保持原代码逻辑
        indices = indices.transpose(1, 0)
        confs = confs.transpose(1, 0)

        if self.debug:
            print(
                f"[CTCDecoder] layout={self.layout}, preds(T,N,C)={preds.shape}, "
                f"dict_len={len(self.idx2char)}, blank={self.blank_token}, "
                f"max_idx={int(indices.max())}, min_idx={int(indices.min())}"
            )

        results = []
        for batch_idx in range(indices.shape[0]):
            text, conf = self._decode_one(indices[batch_idx], confs[batch_idx])
            results.append((text, conf))
        return results

    def _normalize_layout(self, preds: np.ndarray) -> np.ndarray:
        """统一转成原代码使用的 (T, N, C)。"""
        if preds.ndim == 2:
            # 单张图片输出：默认 (T, C)，补 batch 维 → (T, 1, C)
            if self.layout == "TC":
                return preds[:, None, :]
            if self.layout == "CT":
                return preds.T[:, None, :]
            # 未指定时，优先按 (T, C)，避免影响原识别逻辑
            return preds[:, None, :]

        if preds.ndim != 3:
            raise ValueError(f"CTCDecoder only supports 2D/3D output, got shape={preds.shape}")

        if self.layout == "TNC":
            return preds
        if self.layout == "NTC":
            return preds.transpose(1, 0, 2)
        if self.layout == "NCT":
            return preds.transpose(2, 0, 1)
        if self.layout == "TCN":
            return preds.transpose(0, 2, 1)
        if self.layout == "CTN":
            return preds.transpose(1, 2, 0)
        if self.layout == "CNT":
            return preds.transpose(2, 1, 0)

        raise ValueError(
            f"Unsupported layout={self.layout}. Use one of: "
            "TNC, NTC, NCT, TCN, CTN, CNT, TC, CT"
        )

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _get_char(self, idx: int):
        """按原索引取字符；仅在越界时尝试 idx-1 兜底。"""
        ch = self.idx2char.get(idx)
        if ch is not None:
            return ch

        # 常见情况：模型类别比字典多一个，或 blank / 字符表偏移 1 位。
        # 这里只在原 idx 找不到时才尝试 idx-1，不会影响正常数字识别。
        alt_idx = idx - 1
        alt_ch = self.idx2char.get(alt_idx)
        if alt_ch is not None and alt_idx not in self.ignored_tokens:
            if self.debug and not self._warned_missing:
                print(
                    f"[WARN] idx={idx} not in idx2char, fallback to idx-1={alt_idx}. "
                    f"dict_len={len(self.idx2char)}"
                )
                self._warned_missing = True
            return alt_ch

        msg = (
            f"idx={idx} not found in idx2char. "
            f"dict_len={len(self.idx2char)}, blank={self.blank_token}. "
            "Usually this means rec.onnx and char_json do not match, "
            "or the output layout is not TNC/NTC/NCT as configured."
        )

        if self.strict:
            raise KeyError(msg)

        if not self._warned_missing:
            print("[WARN] " + msg)
            self._warned_missing = True
        return None

    def _decode_one(self, idx_seq: np.ndarray, conf_seq: np.ndarray) -> tuple:
        """解码单条序列：去重 + 移除 <BLANK>。"""
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

            ch = self._get_char(idx)
            if ch is None or ch == "<BLANK>":
                prev_idx = -1
                continue

            chars.append(ch)
            confs.append(float(conf_seq[i]))
            prev_idx = idx

        text = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return text, avg_conf
