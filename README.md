# OCR 模型部署包

将训练好的 DBNet（检测）和 CRNN（识别）模型部署到任意环境进行推理。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 最小依赖仅 `onnxruntime + numpy + opencv-python + pyclipper + shapely + pyyaml`，约 100MB。
> 无需安装 PyTorch。

如需 HTTP 服务，额外安装：
```bash
pip install fastapi uvicorn python-multipart
```

### 2. 导出 ONNX 模型

从项目根目录执行：

```bash
# 检测模型
python export.py det \
    --pth ./output/model_det/best_hmean.pth \
    --output deploy/models/DBNet.onnx \
    --backbone resnet34

# 识别模型（需指定训练的字符集 JSON）
python export.py rec \
    --pth ./output/model_rec/best_acc.pth \
    --output deploy/models/CRNN.onnx \
    --backbone rec_mobilenet_v3 \
    --char_json ./data_loader/rec/num_chars_38.json
```

### 3. 配置输入路径

编辑 `config.yml`，设置输入路径：

```yaml
input:
  image_dir_or_path: /path/to/your/images    # ← 改为实际路径
```

或者运行时通过 `-i` 覆盖：

```bash
python inference.py -c config.yml -i /path/to/image.jpg
```

### 4. 测试推理

```bash
# 端到端 OCR（检测 + 识别）
python inference.py -c config.yml                              # 使用配置文件中的路径
python inference.py -c config.yml -i ./test.jpg                # CLI 覆盖

# 仅文本检测
python det_inference.py -c config.yml                          # 配置文件
python det_inference.py -m ./models/DBNet.onnx -i test.jpg    # 直接传参

# 仅文本识别
python rec_inference.py -c config.yml                          # 配置文件
python rec_inference.py -m ./models/CRNN.onnx --char-json ./models/num_chars_38.json -i crop.jpg
```

### 5. 启动 HTTP 服务

```bash
# 开发模式
python ocr_service.py

# 生产模式
uvicorn ocr_service:app --host 0.0.0.0 --port 8000 --workers 4
```

API 文档自动生成：`http://localhost:8000/docs`

```bash
# 测试
curl -X POST http://localhost:8000/predict     -F "file=@test.jpg"   # 端到端
curl -X POST http://localhost:8000/detect      -F "file=@test.jpg"   # 仅检测
curl -X POST http://localhost:8000/recognize   -F "file=@crop.jpg"   # 仅识别
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `det_inference.py` | **独立文本检测推理**（DBNet），可单独运行 |
| `rec_inference.py` | **独立文本识别推理**（CRNN），可单独运行 |
| `inference.py` | 端到端 OCR 推理类（组合 det + rec） |
| `preprocess.py` | 图像预处理（检测 ImageNet 归一化 / 识别 [-1,1] 归一化） |
| `postprocess.py` | 后处理（DB 文本框提取 + CTC 解码 + 字符映射） |
| `export.py` | ONNX 导出脚本（位于项目根目录，需要 PyTorch，训练环境执行一次即可） |
| `ocr_service.py` | FastAPI HTTP 服务封装 |
| `config.yml` | 推理配置文件（模型路径、阈值等） |
| `requirements.txt` | 最小推理依赖 |
| `models/` | ONNX 模型 + 字符集 JSON 存放目录 |

---

## 代码集成

### 仅文本检测

```python
from det_inference import DetInference

det = DetInference.from_config("config.yml")
# 或直接传参：
# det = DetInference(model_path="./models/DBNet.onnx", long_size=960)

# 单张图像 — numpy ndarray (H, W, 3) uint8
result = det.predict(image)             # → {"boxes": [...], "scores": [...]}

# 批量图像 — list of ndarray，各张尺寸可以不同
results = det.predict_batch([img1, img2, img3])

# 单文件 / 多文件 — 传路径字符串
result = det.predict_file("test.jpg")
results = det.predict_files(["a.jpg", "b.jpg", "c.jpg"])

# 可视化
vis = det.draw_boxes(image, result["boxes"], result["scores"])
```

### 仅文本识别

```python
from rec_inference import RecInference

rec = RecInference.from_config("config.yml")
# 或直接传参：
# rec = RecInference(model_path="./models/CRNN.onnx", char_json_path="./models/num_chars_38.json")

# 单张文本行 — numpy ndarray (h, w, 3) uint8
text, score = rec.predict(crop)         # → ("AB123", 0.95)

# 批量文本行 — list of ndarray，各张尺寸可以不同
results = rec.predict_batch([crop1, crop2, crop3])  # → [("AB", 0.9), ("CD", 0.8), ...]

# 单文件 / 多文件
text, score = rec.predict_file("crop.jpg")
results = rec.predict_files(["a.jpg", "b.jpg"])
```

### 端到端检测 + 识别

```python
from det_inference import DetInference
from rec_inference import RecInference
from inference import OCRInference

# 方式 1：配置文件
ocr = OCRInference("config.yml")

# 方式 2：注入已有实例
det = DetInference(model_path="./models/DBNet.onnx")
rec = RecInference(model_path="./models/CRNN.onnx", char_json_path="./models/num_chars_38.json")
ocr = OCRInference.from_instances(det, rec)

# 推理
results = ocr.predict(image)             # 单张
results = ocr.predict_batch([img1, img2]) # 批量
results = ocr.predict_file("test.jpg")    # 单文件
results = ocr.predict_files(["a.jpg","b.jpg"])  # 多文件

for r in results:
    print(f"[{r['score']:.3f}] {r['text']} → {r['bbox']}")

# 也可直接访问内部推理器
ocr.det.predict(image)   # 单独检测
ocr.rec.predict(crop)    # 单独识别
```

---

## 部署到其他机器

只需拷贝 `deploy/` 目录（训练环境的 `export.py` 可省略）：

```
deploy/
├── models/               ← ONNX 模型 + 字符集 JSON
├── det_inference.py       ← 独立检测
├── rec_inference.py       ← 独立识别
├── inference.py           ← 端到端
├── preprocess.py
├── postprocess.py
├── ocr_service.py         ← 可选，HTTP 服务
├── config.yml
├── requirements.txt
├── deploy.md              ← 详细部署指南
└── README.md
```

目标机器执行：
```bash
cd deploy/
pip install -r requirements.txt

# 检测
python det_inference.py -m ./models/DBNet.onnx -i test.jpg

# 识别
python rec_inference.py -m ./models/CRNN.onnx --char-json ./models/num_chars_38.json -i crop.jpg

# 端到端
python inference.py -c config.yml -i test.jpg
```

---

## 常见问题

**Q: 更换字符集后需要重新训练吗？**
A: 需要。CRNN 的 CTC Head 输出维度由字符集大小决定，更换字符集后必须重新训练识别模型。

**Q: 如何加速 CPU 推理？**
A: 安装 `onnxruntime-openvino` 或使用 ONNX 量化（INT8）可大幅提升 CPU 推理速度。

**Q: 如何加速 GPU 推理？**
A: 将 `onnxruntime` 替换为 `onnxruntime-gpu`，并确保 CUDA 可用。
