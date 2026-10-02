from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI(title="Solar Panel Dust Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = (224, 224)
DIRTY_THRESHOLD = 0.5
model = tf.keras.models.load_model("model/final_model.keras")


# ---------- helpers ----------
def load_image(image_bytes):
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


def preprocess_image(image_bytes):
    img = load_image(image_bytes).resize(IMG_SIZE)
    arr = np.array(img)
    return np.expand_dims(arr, axis=0)


def get_patches(img, grid):
    """Split PIL image into grid x grid patches, each resized to model input size."""
    w, h = img.size
    pw, ph = w // grid, h // grid
    patches = []
    for r in range(grid):
        for c in range(grid):
            box = (c * pw, r * ph, (c + 1) * pw, (r + 1) * ph)
            patches.append(np.array(img.crop(box).resize(IMG_SIZE)))
    return np.array(patches)


def coverage_status(pct):
    if pct < 15:
        return "Clean"
    if pct < 40:
        return "Light"
    if pct < 70:
        return "Moderate"
    return "Heavy"


def make_heatmap(img, probs, grid):
    """Overlay red (dirty) / green (clean) translucent boxes on the image. Returns base64 PNG."""
    img = img.copy()
    img.thumbnail((640, 640))
    w, h = img.size
    pw, ph = w // grid, h // grid
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    px = overlay.load()
    for i, p in enumerate(probs):
        r, c = divmod(i, grid)
        color = (255, 0, 0, 110) if p >= DIRTY_THRESHOLD else (0, 200, 0, 90)
        for y in range(r * ph, (r + 1) * ph):
            for x in range(c * pw, (c + 1) * pw):
                px[x, y] = color
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- endpoints ----------
@app.get("/")
def root():
    return {"message": "Solar Panel Dust Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    arr = preprocess_image(image_bytes)

    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "dirty" if prob >= DIRTY_THRESHOLD else "clean"
    confidence = max(prob, 1 - prob)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(prob, 4),
    }


@app.post("/coverage")
async def coverage(file: UploadFile = File(...), grid: int = 6):
    if not 2 <= grid <= 12:
        raise HTTPException(status_code=400, detail="grid must be between 2 and 12")

    image_bytes = await file.read()
    img = load_image(image_bytes)

    patches = get_patches(img, grid)
    probs = model.predict(patches, batch_size=32, verbose=0).flatten()

    dirty_mask = probs >= DIRTY_THRESHOLD
    coverage_pct = float(dirty_mask.mean() * 100)

    return {
        "grid": grid,
        "total_patches": int(grid * grid),
        "dirty_patches": int(dirty_mask.sum()),
        "coverage_percent": round(coverage_pct, 2),
        "status": coverage_status(coverage_pct),
        "patch_scores": np.round(probs.reshape(grid, grid), 3).tolist(),
        "heatmap_png_base64": make_heatmap(img, probs, grid),
    }