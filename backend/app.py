"""Solar panel dust segmentation API.

Matches the Kaggle notebook export (UNet_ResNet34, dust_segmenter.onnx):
  input : "image"  float32 NCHW, RGB, 256x256, pixels / 255   (NO mean/std here,
          ImageNet normalization is already inside the model)
  output: "logits" float32 (N, 1, 256, 256) raw logits -> apply sigmoid
"""
import base64
import io
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "dust_segmenter.onnx"))

IMG_SIZE = 256          # same as the notebook
PIXEL_THRESHOLD = 0.5   # same as the notebook
DIRTY_PERCENT = 1.0     # notebook's image-level rule: dirty if >= 1% dust pixels
MAX_SIDE = 640          # size of the overlay returned to the browser

app = FastAPI(title="Solar Panel Dust Detection API", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

session = None
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    INPUT_NAME = session.get_inputs()[0].name
    print("Model loaded:", MODEL_PATH, "| input:", INPUT_NAME, session.get_inputs()[0].shape)
else:
    print("MODEL NOT FOUND:", MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    # Notebook used PIL BILINEAR resize + /255 only.
    arr = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def status_for(pct: float) -> str:
    if pct < DIRTY_PERCENT:
        return "Clean"
    if pct < 15:
        return "Light"
    if pct < 40:
        return "Moderate"
    return "Heavy"


def to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def root():
    return {"message": "Solar Panel Dust Detection API is running", "model_loaded": session is not None}


@app.get("/model-info")
def model_info():
    if session is None:
        return {"model_loaded": False, "model_path": MODEL_PATH}
    return {
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "input": {"name": INPUT_NAME, "shape": str(session.get_inputs()[0].shape)},
        "output": {"name": session.get_outputs()[0].name, "shape": str(session.get_outputs()[0].shape)},
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(500, f"Model not found at {MODEL_PATH}")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    try:
        logits = session.run(None, {INPUT_NAME: preprocess(image)})[0]
    except Exception as e:
        raise HTTPException(500, f"Model inference failed: {e}")

    prob = sigmoid(logits[0, 0])             # (256, 256) dust probability
    mask = prob >= PIXEL_THRESHOLD
    coverage = float(mask.mean() * 100)
    confidence = float(prob[mask].mean() * 100) if mask.any() else float((1 - prob).mean() * 100)

    # Overlay at display size
    shown = image.copy()
    shown.thumbnail((MAX_SIDE, MAX_SIDE))
    w, h = shown.size
    big_mask = np.asarray(
        Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) >= int(PIXEL_THRESHOLD * 255)
    arr = np.asarray(shown, dtype=np.float32)
    arr[big_mask] = 0.45 * arr[big_mask] + 0.55 * np.array([245, 158, 11], dtype=np.float32)
    overlay = Image.fromarray(arr.astype(np.uint8))
    mask_img = Image.fromarray((big_mask * 255).astype(np.uint8), "L")

    return {
        "label": "dirty" if coverage >= DIRTY_PERCENT else "clean",
        "status": status_for(coverage),
        "coverage_percent": round(coverage, 2),
        "confidence": round(confidence, 2),
        "overlay_png_base64": to_b64_png(overlay),
        "mask_png_base64": to_b64_png(mask_img),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)