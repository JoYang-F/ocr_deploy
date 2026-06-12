"""
文本识别推理（CRNN + ONNX Runtime）

可独立运行，不依赖 PyTorch。

使用示例：
    from rec_inference import RecInference

    # 方式 1：从配置文件加载
    rec = RecInference.from_config("config.yml")

    # 方式 2：直接传参
    rec = RecInference(
        model_path="./models/CRNN.onnx",
        char_json_path="./models/num_chars_38.json",
    )

    # 识别单张文本行图像
    text, score = rec.predict(cropped_image)  # → ("AB123", 0.95)

    # 批量识别
    results = rec.predict_batch([crop1, crop2])

CLI 用法：
    python rec_inference.py -m ./models/CRNN.onnx --char-json ./models/num_chars_38.json -i crop.jpg
    python rec_inference.py -c config.yml -i ./crops/
"""

import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as rt
import yaml

from preprocess import RecPreprocess
from postprocess import CharacterMapper, CTCDecoder


# ============================================================
# 工具函数
# ============================================================

def _resolve_path(rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, rel_path))


def _get_ort_providers() -> list:
    available = rt.get_available_providers()
    preferred = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in preferred if p in available]


# ============================================================
# RecInference
# ============================================================

class RecInference:
    """CRNN 文本识别推理。

    输入裁剪后的文本行图像 → 输出识别文本和置信度。
    """

    def __init__(
        self,
        model_path: str = "./models/CRNN.onnx",
        char_json_path: str = "./models/num_chars_38.json",
        image_shape: tuple = (3, 32, 320),
    ):
        self.model_path = model_path
        self.char_json_path = char_json_path
        self.image_shape = tuple(image_shape)

        abs_path = _resolve_path(model_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {abs_path}")
        char_abs = _resolve_path(char_json_path)
        if not os.path.exists(char_abs):
            raise FileNotFoundError(f"字符映射表不存在: {char_abs}")

        self.sess = rt.InferenceSession(abs_path, providers=_get_ort_providers())
        self.char_mapper = CharacterMapper(char_json_path)
        self.decoder = CTCDecoder(self.char_mapper)

    @classmethod
    def from_config(cls, config_path: str) -> "RecInference":
        """从 YAML 配置文件加载。

        配置文件需包含 rec 段：
            rec:
              model_path: ./models/CRNN.onnx
              char_json_path: ./models/num_chars_38.json
              image_shape: [3, 32, 320]
        """
        config_path = _resolve_path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rec_cfg = cfg.get("rec", cfg)
        return cls(
            model_path=_resolve_path(rec_cfg.get("model_path", "./models/CRNN.onnx")),
            char_json_path=_resolve_path(rec_cfg.get("char_json_path", "./models/num_chars_38.json")),
            image_shape=tuple(rec_cfg.get("image_shape", [3, 32, 320])),
        )

    # ---- 推理 ----

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """识别单张文本行图像中的文字。

        Args:
            image: 裁剪后的文本行 BGR 图像 (h, w, 3), uint8。
                   高度缩放到 32，宽度保持宽高比（上限 320）。

        Returns:
            (text, score): 识别文本和置信度 (0~1)
        """
        if image is None or image.size == 0:
            return "", 0.0

        h, w = image.shape[:2]
        scale = h * 1.0 / self.image_shape[1]
        max_w = int(w / scale)
        if max_w < 4:
            return "", 0.0

        prep = RecPreprocess(
            image_shape=(self.image_shape[0], self.image_shape[1], max_w)
        )
        rec_input = prep(image)
        ort_inputs = {self.sess.get_inputs()[0].name: rec_input}
        ort_outputs = self.sess.run(None, ort_inputs)

        decoded = self.decoder(ort_outputs[0])
        if decoded:
            return decoded[0]
        return "", 0.0

    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """批量识别多张文本行图像。

        Args:
            images: 裁剪后的文本行 BGR 图像列表，每张 (h, w, 3) uint8，
                    各图像尺寸可以不同

        Returns:
            [(text, score), ...]
        """
        return [self.predict(img) for img in images]

    def predict_file(self, image_path: str) -> Tuple[str, float]:
        """从文件路径读取裁剪图像并识别"""
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.predict(img)

    def predict_files(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """批量识别多个图像文件。

        Args:
            image_paths: 图像文件路径列表

        Returns:
            [(text, score), ...]
        """
        return [self.predict_file(p) for p in image_paths]

    # ---- 信息 ----

    def __len__(self) -> int:
        return len(self.char_mapper)

    @property
    def input_name(self) -> str:
        return self.sess.get_inputs()[0].name

    @property
    def char_list(self) -> list:
        return list(self.char_mapper.char2idx.keys())

    @property
    def info(self) -> dict:
        return {
            "model_path": self.model_path,
            "char_json_path": self.char_json_path,
            "image_shape": self.image_shape,
            "num_classes": len(self.char_mapper),
            "providers": self.sess.get_providers(),
        }


# ============================================================
# CLI
# ============================================================

def _collect_images(path: str) -> list:
    img_ext = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
    paths = []
    if os.path.isfile(path):
        paths.append(path)
    elif os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if os.path.splitext(f)[-1][1:].lower() in img_ext:
                paths.append(os.path.join(path, f))
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRNN 文本识别推理")
    parser.add_argument("-c", "--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("-m", "--model", default="./models/CRNN.onnx", help="ONNX 模型路径")
    parser.add_argument("--char-json", default="./models/num_chars_38.json", help="字符映射表 JSON")
    parser.add_argument("-i", "--image", default=None, help="输入图像路径或目录（裁剪后的文本行，覆盖配置文件）")
    args = parser.parse_args()

    # 构建推理器
    if args.config:
        config_path = _resolve_path(args.config)
        rec = RecInference.from_config(config_path)
        # 尝试从配置文件读取输入路径
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        input_cfg = cfg.get("input", {})
        image_path = args.image or input_cfg.get("image_dir_or_path")
    else:
        rec = RecInference(model_path=args.model, char_json_path=args.char_json)
        image_path = args.image

    if not image_path:
        print("请指定输入图像路径，二选一：")
        print("  1) 编辑 config.yml → input.image_dir_or_path")
        print("  2) CLI 参数: -i <path>")
        exit(1)

    print(f"[INFO] 模型: {rec.model_path}, 字符集大小: {len(rec)}")
    print(f"[INFO] Providers: {rec.sess.get_providers()}")

    img_paths = _collect_images(image_path)
    if not img_paths:
        print(f"未找到图像: {image_path}")
        exit(1)
    print(f"共 {len(img_paths)} 张图像\n")

    for img_path in img_paths:
        t0 = time.time()
        text, score = rec.predict_file(img_path)
        elapsed = (time.time() - t0) * 1000
        print(f"[{os.path.basename(img_path)}] text={text!r}  "
              f"score={score:.3f}  {elapsed:.1f}ms")
