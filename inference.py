"""
端到端 OCR 推理（检测 + 识别管线）

内部组合 DetInference + RecInference：
    BGR 图像 → DetInference（文本框检测）
            → 逐框 rotate_crop → RecInference（文本识别）
            → [{bbox, text, score}, ...]

使用示例：
    from inference import OCRInference

    ocr = OCRInference("config.yml")
    results = ocr.predict(image)

    # 也可以单独访问内部推理器
    ocr.det.predict(image)
    ocr.rec.predict(crop)

CLI 用法：
    python inference.py -c config.yml -i test.jpg
    python inference.py -c config.yml -i test.jpg --det
    python inference.py -c config.yml -i crop.jpg --rec
"""

import os
import time
import json
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from preprocess import rotate_crop_image
from det_inference import DetInference
from rec_inference import RecInference


# ============================================================
# 工具函数
# ============================================================

def _resolve_path(rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, rel_path))


# ============================================================
# OCRInference — 端到端检测+识别
# ============================================================

class OCRInference:
    """端到端 OCR 推理引擎。

    内部由 DetInference + RecInference 组成。
    """

    def __init__(
        self,
        config_path: str = "config.yml",
        *,
        det: Optional[DetInference] = None,
        rec: Optional[RecInference] = None,
    ):
        """
        Args:
            config_path: YAML 配置文件路径
            det: 可选，已构建的 DetInference 实例（覆盖配置文件）
            rec: 可选，已构建的 RecInference 实例（覆盖配置文件）
        """
        config_path = _resolve_path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.config = cfg
        self.result_cfg = cfg.get("result", {})

        self.det = det if det is not None else DetInference.from_config(config_path)
        self.rec = rec if rec is not None else RecInference.from_config(config_path)

    @classmethod
    def from_instances(cls, det: DetInference, rec: RecInference) -> "OCRInference":
        """用已有的 DetInference 和 RecInference 实例构建。

        Args:
            det: 检测推理器
            rec: 识别推理器
        """
        inst = object.__new__(cls)
        inst.config = {"result": {"save_dir": "./results", "visualize": False}}
        inst.result_cfg = inst.config["result"]
        inst.det = det
        inst.rec = rec
        return inst

    # ---- 推理 ----

    def predict(self, image: np.ndarray) -> List[dict]:
        """对单张图像执行端到端 OCR。

        Args:
            image: BGR 图像 (H, W, 3), uint8

        Returns:
            [{"bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
              "text": "识别的文本",
              "det_score": 0.85,   # 检测置信度
              "rec_score": 0.95,   # 识别置信度
              "score": 0.95}, ...] # 综合分数（当前取 rec_score）
        """
        if image is None or image.size == 0:
            return []

        det_result = self.det.predict(image)
        boxes = det_result["boxes"]
        scores = det_result["scores"]
        if len(boxes) == 0:
            return []

        results = []
        for box, det_score in zip(boxes, scores):
            text, rec_score = self._recognize_one(image, box)
            if text.strip():
                det_conf = float(np.mean(det_score))
                rec_conf = float(np.mean(rec_score))
                results.append({
                    "bbox": box.tolist(),
                    "text": text,
                    "det_score": round(det_conf, 3),
                    "rec_score": round(rec_conf, 3),
                    "score": round(rec_conf, 3),
                })
        return results

    def predict_batch(self, images: List[np.ndarray]) -> List[List[dict]]:
        """批量端到端推理。

        Args:
            images: BGR 图像列表，每张 (H, W, 3) uint8，各图像尺寸可以不同

        Returns:
            [[{bbox, text, score}, ...], ...] — 每张图像的结果列表
        """
        return [self.predict(img) for img in images]

    def predict_file(self, image_path: str) -> List[dict]:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.predict(img)

    def predict_files(self, image_paths: List[str]) -> List[List[dict]]:
        """批量端到端推理多个图像文件。

        Args:
            image_paths: 图像文件路径列表

        Returns:
            [[{bbox, text, score}, ...], ...] — 每个文件的结果列表
        """
        return [self.predict_file(p) for p in image_paths]

    # ---- 内部 ----

    def _recognize_one(self, image: np.ndarray, box: np.ndarray) -> Tuple[str, float]:
        crop = rotate_crop_image(image, box.astype(np.float32))
        if crop.size == 0:
            return "", 0.0
        return self.rec.predict(crop)

    # ---- 结果保存 ----

    def save_results(
        self,
        results: List[dict],
        image: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None,
        image_name: str = "result",
    ):
        out_dir = output_dir or _resolve_path(
            self.result_cfg.get("save_dir", "./results")
        )
        os.makedirs(out_dir, exist_ok=True)

        txt_path = os.path.join(out_dir, f"{image_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if image is not None and self.result_cfg.get("visualize", True):
            vis = self._draw_results(image, results)
            cv2.imwrite(os.path.join(out_dir, f"{image_name}.jpg"), vis)

    @staticmethod
    def _draw_results(image: np.ndarray, results: List[dict]) -> np.ndarray:
        vis = image.copy()
        for r in results:
            box = np.array(r["bbox"], dtype=np.int32)
            cv2.polylines(vis, [box.reshape(-1, 1, 2)], isClosed=True,
                          color=(0, 0, 255), thickness=2)
            cv2.putText(vis, r["text"], (box[3][0], box[3][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return vis


# ============================================================
# 便捷函数
# ============================================================

def create_ocr(
    det_model: str,
    rec_model: str,
    char_json: str,
    long_size: int = 960,
    **kwargs,
) -> OCRInference:
    """快速创建 OCRInference 实例（无需 YAML 配置文件）。"""
    det = DetInference(
        model_path=det_model,
        long_size=long_size,
        thresh=kwargs.get("thresh", 0.3),
        box_thresh=kwargs.get("box_thresh", 0.6),
        max_candidates=kwargs.get("max_candidates", 1000),
        unclip_ratio=kwargs.get("unclip_ratio", 1.6),
    )
    rec = RecInference(
        model_path=rec_model,
        char_json_path=char_json,
        image_shape=tuple(kwargs.get("image_shape", [3, 32, 320])),
    )
    return OCRInference.from_instances(det, rec)


# ============================================================
# CLI
# ============================================================

def _collect_images(path: str) -> List[str]:
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

    parser = argparse.ArgumentParser(description="端到端 OCR 推理（ONNX Runtime）")
    parser.add_argument("-c", "--config", default="config.yml", help="YAML 配置文件路径")
    parser.add_argument("-i", "--image", default=None, help="输入图像路径或目录（覆盖配置文件）")
    parser.add_argument("-o", "--output", default=None, help="输出目录（覆盖配置文件）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--det", action="store_true", help="仅文本检测")
    mode.add_argument("--rec", action="store_true", help="仅文本识别")
    args = parser.parse_args()

    # 解析配置文件
    config_path = _resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 输入路径：CLI 参数优先，否则用配置文件
    input_cfg = cfg.get("input", {})
    image_path = args.image or input_cfg.get("image_dir_or_path")
    if not image_path:
        print("请指定输入图像路径，二选一：")
        print("  1) 编辑 config.yml → input.image_dir_or_path")
        print("  2) CLI 参数: -i <path>")
        exit(1)

    # 模式：CLI 参数优先，否则用配置文件
    det = args.det or input_cfg.get("mode") == "det"
    rec = args.rec or input_cfg.get("mode") == "rec"

    # 未指定 -o 时，根据 mode 自动选输出目录
    if args.output is None:
        mode_dir = {
            "det": "./results_det",
            "rec": "./results_rec",
            "e2e": "./results_e2e",
        }
        mode = "det" if det else ("rec" if rec else "e2e")
        args.output = mode_dir[mode]

    img_paths = _collect_images(image_path)
    if not img_paths:
        print(f"未找到图像: {image_path}")
        exit(1)
    print(f"配置: {config_path}")
    print(f"共 {len(img_paths)} 张图像")

    # -- 仅检测 --
    if det:
        det = DetInference.from_config(config_path)
        total_time = 0.0
        total_boxes = 0
        total_score = 0.0
        for p in img_paths:
            t0 = time.time()
            result = det.predict_file(p)
            elapsed = time.time() - t0
            name = os.path.basename(p)
            boxes = result["boxes"]
            scores = result["scores"]
            total_time += elapsed
            total_boxes += len(boxes)
            total_score += sum(float(np.mean(s)) for s in scores)
            print(f"[{name}]  {len(boxes)} 个文本框 用时{elapsed:.3f}s")
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                img = cv2.imread(p)
                if img is not None:
                    vis = det.draw_boxes(img, boxes, scores)
                    cv2.imwrite(os.path.join(args.output, name), vis)
            for box, score in zip(boxes, scores):
                print(f" 文本框位置{box.tolist()}  置信度：{score:.4f}")
        # ---- 汇总统计 ----
        avg_time = total_time / len(img_paths)
        avg_score = total_score / total_boxes if total_boxes > 0 else 0.0
        print(f"\n{'='*50}")
        print(f" 图像数量:{len(img_paths)}  文本框总数:{total_boxes}")
        print(f" 总用时:{total_time:.3f}s   平均用时:{avg_time:.3f}s   平均准确度:{avg_score:.4f}")        
        print(f"{'='*50}")
        exit(0)

    # -- 仅识别 --
    if rec:
        rec = RecInference.from_config(config_path)

        # 像 det 分支一样：先创建输出目录。
        # rec 模式不保存可视化图片，而是把每张图的识别结果写入同一个 result.txt。
        out_dir = _resolve_path(args.output or "./results_rec")
        os.makedirs(out_dir, exist_ok=True)
        result_txt = os.path.join(out_dir, "result.txt")

        total_time = 0.0
        total_score = 0.0
        total_texts = 0

        with open(result_txt, "w", encoding="utf-8") as f:
            for p in img_paths:
                t0 = time.time()
                text, score = rec.predict_file(p)
                elapsed = time.time() - t0

                score = float(score)
                total_time += elapsed
                if text.strip():
                    total_score += score
                    total_texts += 1

                # 每张图片一行，包含图片路径、识别文本、置信度、耗时。
                # 使用 JSON Lines，后续用 Python / Excel / pandas 都好读取。
                item = {
                    "image_path": os.path.abspath(p).replace("\\", "/"),
                    "text": text,
                    "score": round(score, 6),
                    "time": round(elapsed, 6),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

                name = os.path.basename(p)
                print(f"[{name}] text={text!r}  score={float(score):.3f}  用时{elapsed:.3f}s")

        # ---- 汇总统计 ----
        avg_time = total_time / len(img_paths)
        avg_score = total_score / total_texts if total_texts > 0 else 0.0
        print(f"\n{'='*50}")
        print(f" 图像数量:{len(img_paths)}  有效文本数:{total_texts}")
        print(f" 总用时:{total_time:.3f}s   平均用时:{avg_time:.3f}s   平均置信度:{avg_score:.4f}")
        print(f" 识别结果已保存至:{result_txt}")
        print(f"{'='*50}")
        exit(0)


    # -- 端到端 --
    ocr = OCRInference(config_path)
    all_results = []  # 汇总所有图像的结果
    total_time = 0.0
    total_det_score = 0.0
    total_rec_score = 0.0
    total_boxes = 0
    for p in img_paths:
        t0 = time.time()
        results = ocr.predict_file(p)
        elapsed = time.time() - t0
        total_time += elapsed
        file_label = os.path.basename(p)
        name = os.path.splitext(file_label)[0]
        ocr.save_results(results, cv2.imread(p), output_dir=args.output, image_name=name)
        total_boxes += len(results)
        # 直接逐条输出识别结果
        for r in results:
            total_det_score += r["det_score"]
            total_rec_score += r["rec_score"]
            r = {"file_name": os.path.basename(p), **r}
            all_results.append(r)
            print(f"[{file_label}] ➡ 识别结果：{r['text']}  score：{r['score']:.4f}  time：{elapsed:.3f}s")
        if len(results) == 0:
            print(f"[{file_label}] ➡ 未检测到文本  time：{elapsed:.3f}s")

    # 汇总写入 result.txt
    if all_results:
        out_dir = args.output or ocr.result_cfg.get("save_dir", "./results")
        out_dir = _resolve_path(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        agg_path = os.path.join(out_dir, "result.txt")
        with open(agg_path, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n汇总结果已保存至: {agg_path}  (共 {len(all_results)} 条)")

    # ---- 端到端汇总统计 ----
    n_images = len(img_paths)
    avg_time = total_time / n_images
    avg_det_score = total_det_score / total_boxes if total_boxes > 0 else 0.0
    avg_rec_score = total_rec_score / total_boxes if total_boxes > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"  图像数量:       {n_images}")
    print(f"  检出文本框:     {total_boxes} 个")
    print(f"  平均用时/图:    {avg_time:.3f}s")
    print(f"  总用时:         {total_time:.3f}s")
    print(f"  平均检测置信度: {avg_det_score:.4f}")
    print(f"  平均识别置信度: {avg_rec_score:.4f}")
    print(f"{'='*50}")
