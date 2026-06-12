# 模型文件目录

请将导出的 ONNX 模型文件放置在此目录下：

- `DBNet.onnx`   — 文本检测模型
- `CRNN.onnx`    — 文本识别模型
- `num_chars_38.json` — 字符映射表

## 导出模型

从项目根目录执行：

```bash
# 导出检测模型 (以 ResNet34 骨干为例)
python export.py det \
    --pth ./output/model_det/best_hmean.pth \
    --output deploy/models/DBNet.onnx \
    --backbone resnet34

# 导出识别模型 (以 MobileNetV3 骨干为例)
python export.py rec \
    --pth ./output/model_rec/best_acc.pth \
    --output deploy/models/CRNN.onnx \
    --backbone rec_mobilenet_v3 \
    --char_json ./data_loader/rec/num_chars_38.json
```
