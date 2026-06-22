<div align="center">

# 📄 OCR 推理部署包

**基于 DBNet + CRNN 的端到端 OCR 推理引擎**

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/onnxruntime-%3E%3D1.7.0-orange.svg)](https://github.com/microsoft/onnxruntime)
[![Code size](https://img.shields.io/github/languages/code-size/JoYang-F/ocr_deploy)](https://github.com/JoYang-F/ocr_deploy)
[![Last Commit](https://img.shields.io/github/last-commit/JoYang-F/ocr_deploy)](https://github.com/JoYang-F/ocr_deploy)

<br>

> 🚀 **纯 ONNX Runtime 推理 · 无需 PyTorch · 轻量高效**
>
> 将训练好的 DBNet（文本检测）和 CRNN（文本识别）模型一键部署到任意环境。  
> 支持 **CLI 命令行** 和 **Python API** 两种使用方式，配合 `camera_ocr.py` 可实时 OCR。

</div>

---

## ✨ 特性一览

| 特性 | 说明 |
|------|------|
| ⚡ **轻量部署** | 纯 ONNX Runtime 推理，无需 PyTorch，依赖仅 ~100MB |
| 🖥️ **跨平台** | 支持 Windows / Linux / macOS，CPU 和 GPU（CUDA / TensorRT） |
| 🎯 **端到端管线** | DBNet 检测 → 透视裁剪 → CRNN 识别，一站式输出 |
| 📦 **两种使用方式** | CLI 命令行 / Python API 任选 |
| 🔧 **灵活配置** | 支持 YAML 配置文件或直接传参，阈值 / 尺寸均可调 |
| 📷 **工业相机** | 海康相机采集 + 实时 OCR，双线程架构 |

---

## 🧠 技术架构

```
┌─────────────────────────────────────────────────────┐
│                   输入图像 (BGR)                      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────┐ │
│  │         文本检测 — DBNet + ONNX                  │ │
│  │  det_inference.py → DetPreprocess → ONNX →      │ │
│  │  DBPostProcess → 文本框坐标                      │ │
│  └─────────────────────┬───────────────────────────┘ │
│                        │                              │
│                        ▼                              │
│  ┌─────────────────────────────────────────────────┐ │
│  │  透视裁剪 rotate_crop_image (preprocess.py)      │ │
│  └─────────────────────┬───────────────────────────┘ │
│                        │                              │
│                        ▼                              │
│  ┌─────────────────────────────────────────────────┐ │
│  │         文本识别 — CRNN + ONNX                   │ │
│  │  rec_inference.py → RecPreprocess → ONNX →      │ │
│  │  CTCDecoder → 文本字符串                         │ │
│  └─────────────────────┬───────────────────────────┘ │
│                        │                              │
│                        ▼                              │
│  ┌─────────────────────────────────────────────────┐ │
│  │   结果输出 (bbox + text + score)                 │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**核心管线：** 输入 BGR 图像 → [DBNet 检测文本框] → [逐框透视裁剪] → [CRNN 识别文字] → [{bbox, text, score}, ...]

---

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖（约 100MB）：

| 包 | 用途 |
|----|------|
| `onnxruntime` | ONNX 推理引擎 |
| `numpy` | 数值计算 |
| `opencv-python` | 图像处理 |
| `pyclipper` | 文本框轮廓膨胀 |
| `shapely` | 多边形几何计算 |
| `pyyaml` | 配置文件解析 |

#### 可选：GPU 加速

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 2. 准备模型

ONNX 模型文件较大（~200MB），未直接包含在仓库中。

**方式一：从 Release 下载（推荐）**

前往 [Releases](https://github.com/JoYang-F/ocr_deploy/releases) 页面下载最新的模型包，解压到 `models/` 目录：

```bash
# 示例：下载 v1.0 版本的模型
curl -L -o models.zip https://github.com/JoYang-F/ocr_deploy/releases/download/v1.0/models.zip
unzip models.zip -d models/
```

**方式二：自行导出**

如果有训练环境，使用 `export.py` 导出 ONNX 模型放入 `models/` 目录：

```
models/
├── DBNet_res34.onnx       ← 检测模型（DBNet）
├── CRNN_res18.onnx        ← 识别模型（CRNN）
└── num_chars_11.json      ← 字符映射表
```

> 模型导出方法参见 [deploy.md](deploy.md) 或训练项目中的 `export.py`。

### 3. 配置

编辑 `config.yml` 设置输入路径：

```yaml
input:
  image_dir_or_path: ./test    # 输入图像目录或单张图像路径
  mode: e2e                    # 运行模式：e2e / det / rec
```

### 4. 运行推理

```bash
# 端到端 OCR（检测 + 识别）
python inference.py -c config.yml

# 指定图像
python inference.py -c config.yml -i test.jpg

# 仅检测
python det_inference.py -c config.yml

# 仅识别（需裁剪好的文本行图像）
python rec_inference.py -c config.yml
```

---

## 使用方式

### 🖥️ CLI 命令行

#### 端到端 OCR

```bash
python inference.py -c config.yml                        # 使用配置文件输入路径
python inference.py -c config.yml -i ./test.jpg          # 单张图片
python inference.py -c config.yml -i ./images/ -o ./out  # 目录批量
python inference.py -c config.yml --det                  # 仅文本检测模式
python inference.py -c config.yml --rec                  # 仅文本识别模式
```

#### 仅检测

```bash
python det_inference.py -c config.yml                                    # 配置文件
python det_inference.py -m ./models/DBNet_res34.onnx -i test.jpg         # 直接传参
python det_inference.py -m ./models/DBNet_res34.onnx -i test.jpg -o ./out
```

#### 仅识别

```bash
python rec_inference.py -c config.yml                                              # 配置文件
python rec_inference.py -m ./models/CRNN_res18.onnx --char-json ./models/num_chars_11.json -i crop.jpg
```

所有 CLI 都会输出详细日志并自动汇总统计信息（总用时、平均置信度等）。

### 🐍 Python API

#### 端到端 OCR

```python
import cv2
from inference import OCRInference

# 方式 1：从配置文件加载
ocr = OCRInference("config.yml")

# 方式 2：直接传入实例
from inference import create_ocr
ocr = create_ocr(
    det_model="./models/DBNet_res34.onnx",
    rec_model="./models/CRNN_res18.onnx",
    char_json="./models/num_chars_11.json",
)

# 推理
image = cv2.imread("test.jpg")
results = ocr.predict(image)  # [{"bbox": ..., "text": ..., "score": ...}, ...]

# 批量
results = ocr.predict_batch([img1, img2])

# 从文件
results = ocr.predict_file("test.jpg")
results = ocr.predict_files(["a.jpg", "b.jpg"])

# 保存结果（文本 JSON + 可视化图像）
ocr.save_results(results, image, output_dir="./output")
```

#### 独立检测

```python
from det_inference import DetInference

det = DetInference.from_config("config.yml")
# 或直接传参：
# det = DetInference(model_path="./models/DBNet_res34.onnx", long_size=960)

result = det.predict(image)    # → {"boxes": [...], "scores": [...]}
vis = det.draw_boxes(image, result["boxes"], result["scores"])
```

#### 独立识别

```python
from rec_inference import RecInference

rec = RecInference.from_config("config.yml")

text, score = rec.predict(cropped_img)  # → ("AB123", 0.95)
text, score = rec.predict_file("crop.jpg")
results = rec.predict_batch([crop1, crop2])
```

---

## 配置文件说明

`config.yml` 完整配置项：

```yaml
# --- 输入 ---
input:
  image_dir_or_path: ./test    # 图像目录或单张图像路径
  mode: e2e                    # 运行模式：e2e（端到端）/ det（仅检测）/ rec（仅识别）

# --- 文本检测模型 (DBNet) ---
det:
  model_path: ./models/DBNet_res34.onnx    # ONNX 检测模型路径
  long_size: 960                            # 推理时图片长边尺寸（越大越准但越慢）
  thresh: 0.3                               # 概率图二值化阈值 ↓ 低则更多框
  box_thresh: 0.6                           # 文本框置信度阈值 ↓ 低则保留更多
  max_candidates: 1000                      # 最大候选框数
  unclip_ratio: 1.6                         # 文本框膨胀比例 ↓ 小则框更紧

# --- 文本识别模型 (CRNN) ---
rec:
  model_path: ./models/CRNN_res18.onnx      # ONNX 识别模型路径
  char_json_path: ./models/num_chars_11.json # 字符映射表 JSON
  image_shape: [3, 32, 320]                 # 识别输入尺寸 [C, H, max_W]

# --- 结果输出 ---
result:
  save_dir: ./results                        # 结果保存目录
  visualize: true                            # 是否保存可视化图像
```

---

## 🔌 工业相机采集

| 工具 | 说明 |
|------|------|
| `camera.py` | 相机画面实时预览 + 手动/自动抓取保存图像 |
| `camera_ocr.py` | 实时采集 + OCR 检测识别 + 可视化显示（双线程） |

### camera.py — 图像采集

```bash
python camera.py
```

| 按键 | 功能 |
|------|------|
| `s` | 单张抓取，保存到 `./captures/` |
| `空格` | 切换自动连续保存模式 |
| `q` | 退出 |

采集的图片可离线进行 OCR 推理：

```bash
python inference.py -c config.yml -i ./captures/
```

### camera_ocr.py — 实时 OCR

```bash
python camera_ocr.py -c config.yml
```

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `s` | 手动抓取当前画面及 OCR 结果 |
| `r` | 切换 OCR 开关 |
| `a` | 切换自动保存（检出文本时自动存图 + txt） |

> ⚠️ **依赖说明：** 相机工具依赖海康 MVS SDK。SDK 文件（`MvImport/`）未包含在仓库中，  
> 请从 [海康机器人官网](https://www.hikrobotics.com/machinevision) 下载安装，  
> 然后将 `MvImport/` 目录复制到本项目根目录即可运行。

---

## 部署到其他机器

`deploy/` 目录完全自包含，复制到目标机器即可运行：

```bash
# 1. 拷贝目录
scp -r deploy/ user@target:/path/

# 2. 目标机器上安装依赖
cd /path/deploy/
pip install -r requirements.txt

# 3. 运行
python inference.py -c config.yml -i test.jpg
```

> 目标机器**完全不需要 PyTorch**，也无需训练代码和数据集。

详细部署步骤、Docker 容器化等参见 [deploy.md](deploy.md)。

---

## 常见问题

<details>
<summary><b>Q: 模型推理报错 "ONNX 模型不存在"</b></summary>

确保 ONNX 模型文件已导出并放在 `models/` 目录下：

```bash
ls models/
# DBNet_res34.onnx  CRNN_res18.onnx  num_chars_11.json
```

模型导出需在训练环境中执行 `export.py`，参见 [deploy.md](deploy.md)。
</details>

<details>
<summary><b>Q: 识别结果全是乱码</b></summary>

检查字符映射 JSON 是否与训练时使用的一致。字符集不匹配会导致解码错误。更换字符集必须重新训练模型。
</details>

<details>
<summary><b>Q: 检测框不准 / 漏检严重</b></summary>

尝试调整 `config.yml` 中的检测参数：

```yaml
det:
  thresh: 0.2          # 降低二值化阈值，检出更多文本区域
  box_thresh: 0.5      # 降低框置信度阈值，保留更多候选框
  unclip_ratio: 1.5    # 减小膨胀比例，让框更紧贴文本
```
</details>

<details>
<summary><b>Q: CPU 推理太慢怎么办？</b></summary>

1. 减小 `long_size`（如 640），牺牲精度换速度
2. 安装 `onnxruntime-openvino`（Intel CPU 优化）
3. 使用 MobileNetV3 等轻量骨干网络（比 ResNet34 快 3x）
4. 启用 ONNX INT8 量化
</details>

<details>
<summary><b>Q: 如何换用自定义字符集？</b></summary>

```bash
# 1. 准备字符映射 JSON
echo '{"<BLANK>":0,"A":1,"B":2, ...}' > my_chars.json

# 2. 用新字符集重新训练识别模型
# 3. 导出 ONNX 并指定新字符集
python export.py rec --pth model.pth --output CRNN.onnx --char_json my_chars.json

# 4. 更新部署目录中的 config.yml 和字符集文件
```
</details>

---

## 相关资源

- [详细部署指南](deploy.md) — 模型导出、Docker、GPU 加速、性能调优
- [DBNet 论文](https://arxiv.org/abs/1911.08947) — Real-time Scene Text Detection with Differentiable Binarization
- [CRNN 论文](https://arxiv.org/abs/1507.05717) — An End-to-End Trainable Neural Network for Image-based Sequence Recognition

---

## License

本项目基于 [MIT License](LICENSE) 开源，可以自由使用、修改、商用。

Copyright (c) 2024 JoYang-F
