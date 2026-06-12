# OCR 模型部署指南

本文档详细介绍如何将训练好的 DBNet + CRNN 模型从训练环境部署到任意机器上进行推理。

---

## 概述

```
┌────────────── 训练环境（需要 PyTorch）──────────────┐
│                                                      │
│  1. 训练模型 → .pth 检查点                           │
│  2. export.py 导出 → .onnx 模型                      │
│                                                      │
└──────────────────────┬───────────────────────────────┘
                       │ 拷贝 deploy/ 目录
                       ▼
┌────────────── 部署环境（无需 PyTorch）──────────────┐
│                                                      │
│  3. 安装最小依赖（约 100MB）                         │
│  4. 推理：CLI / Python API / HTTP 服务               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 第一步：训练环境 — 导出 ONNX 模型

> ⚠️ 此步骤在训练机器上执行，**需要 PyTorch 环境**。

### 1.1 确认检查点文件

训练完成后，检查点位于 `output/` 目录：

```bash
ls output/model_det/    # 检测模型：best_hmean.pth, latest.pth 等
ls output/model_rec/    # 识别模型：best_acc.pth, latest.pth 等
```

### 1.2 确认模型架构

查看训练配置文件，确认导出参数：

```bash
cat config/train/det.yml   # 查看 Backbone.name（resnet34 / det_mobilenet_v3）
cat config/train/rec.yml   # 查看 Backbone.name（rec_mobilenet_v3 / rec_resnet18 等）
```

### 1.3 导出 ONNX

```bash
# 导出检测模型
python export.py det \
    --pth ./output/model_det/best_hmean.pth \
    --output deploy/models/DBNet.onnx \
    --backbone resnet34

# 导出识别模型
python export.py rec \
    --pth ./output/model_rec/best_acc.pth \
    --output deploy/models/CRNN.onnx \
    --backbone rec_mobilenet_v3 \
    --char_json ./data_loader/rec/num_chars_38.json
```

**参数对照表**：

| 训练 Backbone | `--backbone` 参数 |
|---------------|-------------------|
| DBNet + ResNet18 | `resnet18` |
| DBNet + ResNet34 | `resnet34` |
| DBNet + MobileNetV3 | `det_mobilenet_v3` |
| CRNN + ResNet18 | `rec_resnet18` |
| CRNN + ResNet34 | `rec_resnet34` |
| CRNN + MobileNetV3 | `rec_mobilenet_v3` |

> 也可用 `--config` 直接从训练 YAML 读取架构：
> ```bash
> python export.py det --config config/train/det.yml --pth ./output/model_det/best_hmean.pth --output deploy/models/DBNet.onnx
> ```

### 1.4 验证导出

导出成功会打印模型输入输出 shape：

```
[DET] 输入 shape: (1, 3, 224, 224) → 输出 shape: (1, 1, 224, 224)
[DET] ✅ ONNX 模型已导出: deploy/models/DBNet.onnx

[REC] 输入 shape: (1, 3, 32, 320) → 输出 shape: (79, 1, 38)
[REC] ✅ ONNX 模型已导出: deploy/models/CRNN.onnx
```

### 1.5 确认 deploy 目录完整

```bash
ls -la deploy/
# ├── models/DBNet.onnx       ← 已导出
# ├── models/CRNN.onnx        ← 已导出
# ├── models/num_chars_38.json ← 字符映射表
# ├── det_inference.py
# ├── rec_inference.py
# ├── inference.py
# ├── preprocess.py
# ├── postprocess.py
# ├── ocr_service.py
# ├── config.yml
# └── requirements.txt
```

---

## 第二步：拷贝到目标机器

### 2.1 打包

```bash
# 方式 A：直接拷贝整个 deploy/ 目录（Windows / Linux / macOS）
cp -r deploy/ /path/to/target/

# 方式 B：打包传输
tar -czf ocr_deploy.tar.gz deploy/
scp ocr_deploy.tar.gz user@target:/path/
```

### 2.2 目标机器解压

```bash
tar -xzf ocr_deploy.tar.gz
cd deploy/
```

> `deploy/` 目录完全自包含，不依赖父项目代码。

---

## 第三步：目标机器 — 安装依赖

### 3.1 Python 环境

最低要求 **Python 3.8+**。

```bash
python --version       # Python 3.8.0+
pip --version
```

### 3.2 安装推理依赖（最小，约 100MB）

```bash
cd deploy/
pip install -r requirements.txt
```

依赖清单：

| 包 | 用途 | 大小 |
|----|------|------|
| `onnxruntime` | ONNX 推理引擎 | ~30MB |
| `numpy` | 数值计算 | ~20MB |
| `opencv-python` | 图像处理 | ~40MB |
| `pyclipper` | 文本框轮廓膨胀 | ~1MB |
| `shapely` | 多边形几何计算 | ~3MB |
| `pyyaml` | 配置文件解析 | ~1MB |

> **总计约 100MB**，远小于 PyTorch 环境（~2GB+）。

### 3.3 （可选）GPU 推理

如果有 NVIDIA GPU 且已装 CUDA：

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 3.4 （可选）HTTP 服务

如需 REST API 推理服务：

```bash
pip install fastapi uvicorn python-multipart
```

---

## 第四步：配置

### 4.1 编辑配置文件

```bash
vim config.yml
```

关键配置项：

```yaml
# 输入：推理的图片目录
input:
  image_dir_or_path: /data/my_images    # ← 改为实际路径
  mode: e2e                            # e2e（端到端）/ det / rec

# 检测模型
det:
  model_path: ./models/DBNet.onnx       # ← 确认模型文件路径
  long_size: 960                        # 长边尺寸（越大越准但越慢）


# 识别模型
rec:
  model_path: ./models/CRNN.onnx        # ← 确认模型文件路径
  char_json_path: ./models/num_chars_38.json

# 输出
result:
  save_dir: ./results                   # ← 改为需要的输出目录
  visualize: true
```

### 4.2 快速验证

放入一张测试图到 `./images/`（或配置的输入目录），然后：

```bash
# 端到端（检测+识别）
python inference.py -c config.yml

# 或直接指定图片
python inference.py -c config.yml -i test.jpg
```

---

## 第五步：运行推理

### 方式 A：命令行

```bash
# === 端到端 OCR ===
python inference.py -c config.yml                          # 使用配置文件中的输入目录
python inference.py -c config.yml -i ./test.jpg            # 单张图片
python inference.py -c config.yml -i ./test_dir/ -o ./out  # 目录 → 指定输出

# === 仅检测 ===
python det_inference.py -c config.yml                      # 配置文件
python det_inference.py -m ./models/DBNet.onnx -i test.jpg # 直接传参

# === 仅识别 ===
python rec_inference.py -c config.yml                      # 配置文件
python rec_inference.py -m ./models/CRNN.onnx --char-json ./models/num_chars_38.json -i crop.jpg
```

### 方式 B：Python API

```python
import cv2
from inference import OCRInference

# 初始化
ocr = OCRInference("config.yml")

# 单张推理
image = cv2.imread("test.jpg")
results = ocr.predict(image)

# 批量推理
images = [cv2.imread(f) for f in ["a.jpg", "b.jpg"]]
batch_results = ocr.predict_batch(images)

# 处理结果
for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")
    print(f"  坐标: {r['bbox']}")

# 保存结果（文本 + 可视化图像）
ocr.save_results(results, image, output_dir="./output")
```

```python
# 单独使用检测
from det_inference import DetInference
det = DetInference.from_config("config.yml")
result = det.predict(image)
vis = det.draw_boxes(image, result["boxes"], result["scores"])
cv2.imwrite("det_result.jpg", vis)
```

```python
# 单独使用识别
from rec_inference import RecInference
rec = RecInference.from_config("config.yml")
text, score = rec.predict(cropped_text_line)
print(f"{text} ({score:.3f})")
```

### 方式 C：HTTP 服务

```bash
# 启动服务
python ocr_service.py
# 或
uvicorn ocr_service:app --host 0.0.0.0 --port 8000
```

API 调用：

```bash
# 端到端 OCR
curl -X POST http://localhost:8000/predict -F "file=@test.jpg"

# 仅检测
curl -X POST http://localhost:8000/detect -F "file=@test.jpg"

# 仅识别
curl -X POST http://localhost:8000/recognize -F "file=@crop.jpg"

# 健康检查
curl http://localhost:8000/health
```

返回格式：

```json
{
  "results": [
    {
      "bbox": [[10, 20], [100, 20], [100, 50], [10, 50]],
      "text": "AB123",
      "score": 0.953
    }
  ],
  "elapsed_ms": 245.3
}
```

浏览器打开 `http://localhost:8000/docs` 查看交互式 API 文档。

---

## 第六步：结果说明

### 输出文件

```
results/
├── test1.txt      ← 识别文本结果（JSON 格式，每行一条）
├── test1.jpg      ← 可视化图像（红框 + 绿色识别文本）
├── test2.txt
├── test2.jpg
└── ...
```

### 文本框坐标格式

`bbox` 为四个角点，顺序为 **左上 → 右上 → 右下 → 左下**：

```
[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

### 置信度说明

- `score`: 检测置信度 × 识别置信度的综合分数，范围 0~1
- 值越高越可信，建议过滤 `score < 0.5` 的结果

---

## 性能参考

| 场景 | 硬件 | 检测耗时 | 识别耗时（每框） | 总耗时（10框） |
|------|------|---------|---------------|-------------|
| CPU | Intel i7-12700 | ~200ms | ~15ms | ~350ms |
| CPU | Apple M1 | ~150ms | ~10ms | ~250ms |
| GPU | NVIDIA T4 | ~30ms | ~3ms | ~60ms |

> 检测使用 DBNet-ResNet34 + long_size=960；识别使用 CRNN-MobileNetV3。

---

## Docker 部署（可选）

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型
COPY . .

# 启动服务
CMD ["python", "ocr_service.py"]
```

### 构建和运行

```bash
docker build -t ocr-service .
docker run -p 8000:8000 -v $(pwd)/models:/app/models ocr-service
```

---

## 常见问题

### Q: 模型推理报错 "ONNX 模型不存在"

确保模型文件已从训练环境导出并放在 `deploy/models/` 下：

```bash
ls deploy/models/
# DBNet.onnx  CRNN.onnx  num_chars_38.json
```

### Q: 识别结果全是错的

检查字符集 JSON 是否与训练时一致。更换字符集必须重新训练模型。

### Q: 检测框不准 / 漏检

调整 `config.yml` 中的阈值：

```yaml
det:
  thresh: 0.2          # 降低二值化阈值（检出更多，可能误检）
  box_thresh: 0.5      # 降低框置信度阈值（保留更多框）
  unclip_ratio: 1.5    # 减小膨胀比例（框更紧）
```

### Q: CPU 推理太慢

1. 减小 `long_size`（如 640），牺牲精度换速度
2. 安装 `onnxruntime-openvino`（Intel CPU 优化）
3. 使用小骨干网络（MobileNetV3 比 ResNet34 快 3x）

### Q: 更换自己的字符集

```bash
# 1. 准备字符映射 JSON
echo '{"<BLANK>":0,"我":1,"是":2,...}' > my_chars.json

# 2. 用新字符集重新训练识别模型
python train.py -c config/train/rec.yml  # 修改 character_json_path

# 3. 导出 ONNX（指定新字符集）
python export.py rec \
    --pth ./output/model_rec/best_acc.pth \
    --output deploy/models/CRNN.onnx \
    --backbone rec_mobilenet_v3 \
    --char_json my_chars.json

# 4. 替换部署目录中的字符集文件
cp my_chars.json deploy/models/
```

### Q: 多 Worker 部署时内存不足

每个 worker 会加载一份模型副本。可以使用共享内存或减少 worker 数量：

```bash
uvicorn ocr_service:app --workers 2  # 只开 2 个 worker
```
