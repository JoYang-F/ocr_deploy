"""
文本检测推理（DBNet + ONNX Runtime）

可独立运行，不依赖 PyTorch。

使用示例：
    from det_inference import DetInference

    # 方式 1：从配置文件加载
    det = DetInference.from_config("config.yml")

    # 方式 2：直接传参
    det = DetInference(model_path="./models/DBNet.onnx", long_size=960)

    # 推理
    result = det.predict(image)  # → {"boxes": [...], "scores": [...]}

    # 可视化
    vis = det.draw_boxes(image, result["boxes"], result["scores"])
    cv2.imwrite("output.jpg", vis)

CLI 用法：
    python det_inference.py -m ./models/DBNet.onnx -i test.jpg -o ./results
    python det_inference.py -c config.yml -i ./images/
"""

import os
import time
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as rt
import yaml

from preprocess import DetPreprocess
from postprocess import DBPostProcess


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
# DetInference
# ============================================================

class DetInference:
    """DBNet 文本检测推理。

    输入 BGR 图像 → 输出检测到的文本框坐标和置信度。
    """

    def __init__(
        self,
        model_path: str = "./models/DBNet.onnx",
        long_size: int = 960,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.6,
    ):
        self.model_path = model_path
        self.long_size = long_size
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

        abs_path = _resolve_path(model_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {abs_path}")
        self.sess = rt.InferenceSession(abs_path, providers=_get_ort_providers())

        self.preprocess = DetPreprocess(long_size=self.long_size)
        self.postprocess = DBPostProcess(
            thresh=self.thresh,
            box_thresh=self.box_thresh,
            max_candidates=self.max_candidates,
            unclip_ratio=self.unclip_ratio,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "DetInference":
        """从 YAML 配置文件加载。

        配置文件需包含 det 段：
            det:
              model_path: ./models/DBNet.onnx
              long_size: 960
              thresh: 0.3
              box_thresh: 0.6
              max_candidates: 1000
              unclip_ratio: 1.6
        """
        config_path = _resolve_path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        det_cfg = cfg.get("det", cfg)
        return cls(
            model_path=_resolve_path(det_cfg.get("model_path", "./models/DBNet.onnx")),
            long_size=det_cfg.get("long_size", 960),
            thresh=det_cfg.get("thresh", 0.3),
            box_thresh=det_cfg.get("box_thresh", 0.6),
            max_candidates=det_cfg.get("max_candidates", 1000),
            unclip_ratio=det_cfg.get("unclip_ratio", 1.6),
        )

    # ---- 推理 ----

    def predict(self, image: np.ndarray) -> dict:
        """检测图像中的文本区域。

        Args:
            image: BGR 图像 (H, W, 3), uint8

        Returns:
            {
                "boxes":  list of np.ndarray — 每个 (4, 2) int16，四个角点（左上→右上→右下→左下）
                "scores": list of float — 每个框的置信度
            }
        """
        if image is None or image.size == 0:
            return {"boxes": [], "scores": []}

        data = self.preprocess(image)
        ort_inputs = {self.sess.get_inputs()[0].name: data["image"]}
        ort_outputs = self.sess.run(None, ort_inputs)
        boxes, scores = self.postprocess(ort_outputs[0], data["src_scale"])
        return {"boxes": boxes, "scores": scores}

    def predict_file(self, image_path: str) -> dict:
        """从文件路径读取图像并检测"""
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.predict(img)

    def predict_batch(self, images: List[np.ndarray]) -> List[dict]:
        """批量检测多张图像。

        Args:
            images: BGR 图像列表，每张 (H, W, 3) uint8，
                    各图像尺寸可以不同

        Returns:
            [{"boxes": [...], "scores": [...]}, ...]
        """
        return [self.predict(img) for img in images]

    def predict_files(self, image_paths: List[str]) -> List[dict]:
        """批量检测多个图像文件。

        Args:
            image_paths: 图像文件路径列表

        Returns:
            [{"boxes": [...], "scores": [...]}, ...]
        """
        return [self.predict_file(p) for p in image_paths]

    # ---- 可视化 ----

    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: list,
        scores: Optional[list] = None,
    ) -> np.ndarray:
        """在图像上绘制检测框。

        Args:
            image: 原始 BGR 图像
            boxes:  predict() 返回的 boxes 列表
            scores: 可选，置信度列表（标注在框上）

        Returns:
            带红色检测框的 BGR 图像
        """
        vis = image.copy()
        for i, box in enumerate(boxes):
            pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            if scores and i < len(scores):
                s = float(np.mean(scores[i]))
                cv2.putText(
                    vis, f"{s:.3f}",
                    (int(box[0][0]), int(box[0][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                )
        return vis

    # ---- 信息 ----

    @property
    def input_name(self) -> str:
        return self.sess.get_inputs()[0].name

    @property
    def info(self) -> dict:
        return {
            "model_path": self.model_path,
            "long_size": self.long_size,
            "thresh": self.thresh,
            "box_thresh": self.box_thresh,
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

    parser = argparse.ArgumentParser(description="DBNet 文本检测推理")
    parser.add_argument("-c", "--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("-m", "--model", default="./models/DBNet.onnx", help="ONNX 模型路径")
    parser.add_argument("-i", "--image", default=None, help="输入图像路径或目录（覆盖配置文件）")
    parser.add_argument("-o", "--output", default="./results_det", help="输出目录")
    parser.add_argument("--long-size", type=int, default=960, help="长边尺寸")
    parser.add_argument("--thresh", type=float, default=0.3, help="二值化阈值")
    parser.add_argument("--box-thresh", type=float, default=0.6, help="文本框置信度阈值")
    args = parser.parse_args()

    # 构建推理器
    if args.config:
        config_path = _resolve_path(args.config)
        det = DetInference.from_config(config_path)
        # 尝试从配置文件读取输入路径
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        input_cfg = cfg.get("input", {})
        image_path = args.image or input_cfg.get("image_dir_or_path")
        if args.output == "./results_det":
            args.output = cfg.get("result", {}).get("save_dir", args.output)
    else:
        det = DetInference(
            model_path=args.model,
            long_size=args.long_size,
            thresh=args.thresh,
            box_thresh=args.box_thresh,
        )
        image_path = args.image

    if not image_path:
        print("请指定输入图像路径，二选一：")
        print("  1) 编辑 config.yml → input.image_dir_or_path")
        print("  2) CLI 参数: -i <path>")
        exit(1)

    print(f"[INFO] 模型: {det.model_path}, long_size={det.long_size}, "
          f"thresh={det.thresh}, box_thresh={det.box_thresh}")
    print(f"[INFO] Providers: {det.sess.get_providers()}")

    img_paths = _collect_images(image_path)
    if not img_paths:
        print(f"未找到图像: {image_path}")
        exit(1)
    print(f"共 {len(img_paths)} 张图像\n")

    os.makedirs(args.output, exist_ok=True)
    total_boxes = 0

    for img_path in img_paths:
        t0 = time.time()
        result = det.predict_file(img_path)
        elapsed = (time.time() - t0) * 1000
        n = len(result["boxes"])
        total_boxes += n

        name = os.path.basename(img_path)
        print(f"[{name}] {n} 个文本框, {elapsed:.1f}ms")
        for box, score in zip(result["boxes"], result["scores"]):
            print(f"  [{float(np.mean(score)):.3f}] {box.tolist()}")

        # 保存可视化
        img = cv2.imread(img_path)
        if img is not None:
            vis = det.draw_boxes(img, result["boxes"], result["scores"])
            out_name = os.path.splitext(name)[0] + ".jpg"
            cv2.imwrite(os.path.join(args.output, out_name), vis)

    print(f"\n总计: {total_boxes} 个文本框")
