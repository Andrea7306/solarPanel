
"""
Solar Panel Dust Segmentation API

Model: U-Net + ResNet34 exported to ONNX.

Input:
    Name: image
    Shape: (1, 3, 256, 256)
    Format: RGB, float32, pixel values / 255

Output:
    Raw segmentation logits with shape (1, 1, 256, 256).
    Sigmoid is applied to obtain probabilities.
"""

import base64
import io
import os

import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your ONNX file is directly inside the backend folder.
# You can override this path using the MODEL_PATH environment variable.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "dust_segmenter.onnx")
)

IMG_SIZE = 256
PIXEL_THRESHOLD = 0.5
DIRTY_PERCENT = 1.0
MAX_SIDE = 640

app = FastAPI(
    title="Solar Panel Dust Detection API",
    version="1.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 2. LOAD ONNX MODEL
# --------------------------------------------------

session = None
INPUT_NAME = None
OUTPUT_NAME = None

if os.path.isfile(MODEL_PATH):
    try:
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        INPUT_NAME = session.get_inputs()[0].name
        OUTPUT_NAME = session.get_outputs()[0].name

        print("ONNX model loaded successfully!")
        print("Model path:", MODEL_PATH)
        print("Input name:", INPUT_NAME)
        print("Input shape:", session.get_inputs()[0].shape)
        print("Output name:", OUTPUT_NAME)
        print("Output shape:", session.get_outputs()[0].shape)

    except Exception as e:
        print("ONNX MODEL LOADING FAILED:", str(e))
        session = None

else:
    print("MODEL NOT FOUND:", MODEL_PATH)


# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess(img: Image.Image) -> np.ndarray:
    """
    Convert image into the model's expected input:
    RGB, float32, NCHW, shape (1, 3, 256, 256).

    Matches the notebook preprocessing: resize and /255.
    """

    img = img.convert("RGB")

    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    arr = np.asarray(img, dtype=np.float32) / 255.0

    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)

    # CHW -> NCHW
    arr = np.expand_dims(arr, axis=0)

    return np.ascontiguousarray(arr, dtype=np.float32)


# --------------------------------------------------
# 4. SIGMOID
# --------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert logits into probabilities safely."""

    x = np.clip(x, -80, 80)

    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------
# 5. CONTAMINATION STATUS
# --------------------------------------------------

def status_for(coverage: float) -> str:
    if coverage < 1.0:
        return "Clean"

    elif coverage < 15.0:
        return "Light"

    elif coverage < 40.0:
        return "Moderate"

    else:
        return "Heavy"


# --------------------------------------------------
# 6. CONVERT IMAGE TO BASE64 PNG
# --------------------------------------------------

def to_b64_png(img: Image.Image) -> str:
    buffer = io.BytesIO()

    img.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# --------------------------------------------------
# 7. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Solar Panel Dust Detection API is running",
        "model_loaded": session is not None
    }


# --------------------------------------------------
# 8. MODEL INFORMATION
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    if session is None:
        return {
            "model_loaded": False,
            "model_path": MODEL_PATH,
            "error": "ONNX model is not loaded. Check the model path and startup logs."
        }

    return {
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "input": {
            "name": INPUT_NAME,
            "shape": str(session.get_inputs()[0].shape)
        },
        "output": {
            "name": OUTPUT_NAME,
            "shape": str(session.get_outputs()[0].shape)
        }
    }


# --------------------------------------------------
# 9. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check if model is available
    if session is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ONNX model is not loaded. "
                f"Expected model location: {MODEL_PATH}"
            )
        )

    # Read uploaded image
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file. Upload a valid JPG or PNG."
        )

    finally:
        await file.close()

    # Preprocess and run model inference
    try:
        input_array = preprocess(image)

        outputs = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: input_array}
        )

        logits = outputs[0]

        if logits.ndim != 4 or logits.shape[0] != 1:
            raise ValueError(
                f"Unexpected model output shape: {logits.shape}"
            )

        if logits.shape[1] != 1:
            raise ValueError(
                f"Expected one output channel, got: {logits.shape[1]}"
            )

        # (1, 1, 256, 256) -> (256, 256)
        prob = sigmoid(logits[0, 0])

    except Exception as e:
        print("MODEL INFERENCE ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    # --------------------------------------------------
    # 10. GENERATE BINARY DUST MASK
    # --------------------------------------------------

    mask = prob >= PIXEL_THRESHOLD

    # Percentage of pixels predicted as dust
    coverage = float(mask.mean() * 100.0)

    # Prediction confidence estimate; not a calibrated probability
    if mask.any():
        confidence = float(prob[mask].mean() * 100.0)
    else:
        confidence = float((1.0 - prob).mean() * 100.0)

    label = (
        "dirty"
        if coverage >= DIRTY_PERCENT
        else "clean"
    )

    status = status_for(coverage)

    # --------------------------------------------------
    # 11. CREATE OVERLAY AND MASK AT DISPLAY SIZE
    # --------------------------------------------------

    shown = image.copy()
    shown.thumbnail((MAX_SIDE, MAX_SIDE))

    width, height = shown.size

    # Resize probability map to match the displayed image.
    probability_image = Image.fromarray(
        np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    )

    probability_image = probability_image.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    # Threshold using the same probability cutoff.
    big_mask = (
        np.asarray(probability_image, dtype=np.uint8)
        >= int(PIXEL_THRESHOLD * 255)
    )

    # Create amber-colored overlay on predicted dust pixels.
    image_array = np.asarray(
        shown,
        dtype=np.float32
    ).copy()

    overlay_color = np.array(
        [245, 158, 11],
        dtype=np.float32
    )

    image_array[big_mask] = (
        0.45 * image_array[big_mask]
        + 0.55 * overlay_color
    )

    overlay = Image.fromarray(
        np.clip(image_array, 0, 255).astype(np.uint8)
    )

    mask_image = Image.fromarray(
        (big_mask * 255).astype(np.uint8),
        mode="L"
    )

    # --------------------------------------------------
    # 12. RETURN API RESPONSE
    # --------------------------------------------------

    return {
        "label": label,
        "status": status,
        "coverage_percent": round(coverage, 2),
        "confidence": round(confidence, 2),
        "overlay_png_base64": to_b64_png(overlay),
        "mask_png_base64": to_b64_png(mask_image)
    }


# --------------------------------------------------
# 13. RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )