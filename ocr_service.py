"""
FastAPI OCR 推理服务

启动方式：
    # 开发模式
    python ocr_service.py

    # 生产模式（多 worker）
    uvicorn ocr_service:app --host 0.0.0.0 --port 8000 --workers 4

API 端点：
    POST /predict          — 端到端 OCR（检测+识别）
    POST /predict_batch    — 批量 OCR
    POST /detect           — 仅文本检测
    POST /recognize        — 仅文本识别
    GET  /health           — 健康检查
    GET  /info             — 模型信息

依赖（可选）：
    pip install fastapi uvicorn python-multipart
"""

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import cv2

# FastAPI 为可选依赖
try:
    from fastapi import FastAPI, File, UploadFile, Query, HTTPException
    import uvicorn
except ImportError:
    raise ImportError(
        "请安装 FastAPI 依赖: pip install fastapi uvicorn python-multipart"
    )

from inference import DetInference, RecInference, OCRInference


# ============================================================
# 应用初始化
# ============================================================

CONFIG_PATH = os.environ.get("OCR_CONFIG", "config.yml")

# 启动时加载模型
ocr: Optional[OCRInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动加载模型，关闭释放资源"""
    global ocr
    print(f"[startup] 加载配置: {CONFIG_PATH}")
    ocr = OCRInference(CONFIG_PATH)
    print("[startup] 模型加载完成 ✓")
    yield
    # 关闭时无需额外清理（ONNX Runtime 自动释放）


app = FastAPI(
    title="OCR Service",
    description="基于 DBNet + CRNN 的文本检测识别服务",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 请求/响应模型
# ============================================================

from pydantic import BaseModel, Field


class TextBox(BaseModel):
    bbox: List[List[float]] = Field(..., description="四个角点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    text: str = Field(..., description="识别文本")
    score: float = Field(..., description="置信度")


class PredictResponse(BaseModel):
    results: List[TextBox] = Field(default_factory=list, description="识别结果列表")
    elapsed_ms: float = Field(..., description="推理耗时 (毫秒)")
    image_id: Optional[str] = Field(None, description="图片标识")


class BatchPredictResponse(BaseModel):
    items: List[PredictResponse] = Field(..., description="每张图片的结果")


class HealthResponse(BaseModel):
    status: str = "ok"
    models: dict


# ============================================================
# API 端点
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    global ocr
    if ocr is None:
        raise HTTPException(503, "模型尚未加载")
    return HealthResponse(
        status="ok",
        models={
            "det": "DBNet",
            "rec": "CRNN",
            "char_classes": len(ocr.rec),
        },
    )


@app.get("/info")
async def info():
    """返回模型配置信息"""
    global ocr
    if ocr is None:
        raise HTTPException(503, "模型尚未加载")
    return {
        "det": ocr.det.info,
        "rec": ocr.rec.info,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="图像文件 (jpg/png/bmp/tif)"),
    image_id: Optional[str] = Query(None, description="图片标识"),
):
    """端到端 OCR：上传图片，返回检测框 + 识别文本"""
    global ocr
    if ocr is None:
        raise HTTPException(503, "模型尚未加载")

    # 读取上传文件
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, f"无法解码图片: {file.filename}")

    # 推理
    t0 = time.time()
    results = ocr.predict(img)
    elapsed = (time.time() - t0) * 1000

    return PredictResponse(
        results=[TextBox(**r) for r in results],
        elapsed_ms=round(elapsed, 1),
        image_id=image_id or file.filename,
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="多张图像文件"),
):
    """批量 OCR：一次上传多张图片"""
    global ocr
    if ocr is None:
        raise HTTPException(503, "模型尚未加载")

    items = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            items.append(PredictResponse(
                results=[], elapsed_ms=0, image_id=f"{file.filename} (decode error)"
            ))
            continue

        t0 = time.time()
        results = ocr.predict(img)
        elapsed = (time.time() - t0) * 1000

        items.append(PredictResponse(
            results=[TextBox(**r) for r in results],
            elapsed_ms=round(elapsed, 1),
            image_id=file.filename,
        ))

    return BatchPredictResponse(items=items)


class DetBox(BaseModel):
    bbox: List[List[float]] = Field(..., description="四个角点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")
    score: float = Field(..., description="置信度")


class DetectResponse(BaseModel):
    boxes: List[DetBox] = Field(default_factory=list, description="检测框列表")
    elapsed_ms: float = Field(..., description="推理耗时 (毫秒)")


class RecognizeResponse(BaseModel):
    text: str = Field(..., description="识别文本")
    score: float = Field(..., description="置信度")
    elapsed_ms: float = Field(..., description="推理耗时 (毫秒)")


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(..., description="图像文件 (jpg/png/bmp/tif)"),
):
    """仅文本检测：上传图片，返回文本框坐标"""
    global ocr
    if ocr is None:
        raise HTTPException(503, "模型尚未加载")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"无法解码图片: {file.filename}")

    t0 = time.time()
    result = ocr.det.predict(img)
    elapsed = (time.time() - t0) * 1000

    boxes = []
    for box, score in zip(result["boxes"], result["scores"]):
        boxes.append(DetBox(
            bbox=box.tolist(),
            score=round(float(np.mean(score)), 3),
        ))

    return DetectResponse(boxes=boxes, elapsed_ms=round(elapsed, 1))


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(..., description="裁剪后的文本行图像"),
):
    """仅文本识别：上传裁剪好的文本行图像，返回识别文字"""
    global ocr
    if ocr is None:
        raise HTTPException(503, "模型尚未加载")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"无法解码图片: {file.filename}")

    t0 = time.time()
    text, score = ocr.rec.predict(img)
    elapsed = (time.time() - t0) * 1000

    return RecognizeResponse(
        text=text,
        score=round(float(score), 3),
        elapsed_ms=round(elapsed, 1),
    )


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "ocr_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
