# Solar Panel Dust Detection

DATASET LINK - https://universe.roboflow.com/august-4t80o/solar-dust?utm_source=chatgpt.com

Upload a photo of a solar panel and the app highlights the dusty area and reports what percentage of the image it covers.

A U-Net (ResNet34 encoder) segmentation model is trained on Kaggle, exported to ONNX, and served by a FastAPI backend. A single-page frontend sends images to the API and shows the result.

## How it works

```
Browser (index.html)  --image-->  FastAPI (app.py)  -->  ONNX Runtime (UNet-ResNet34)
        ^                                                        |
        +------ coverage %, status, overlay, mask  <-------------+
```

## Project structure

```
solar_dust_app/
├── backend/
│   ├── app.py                  FastAPI server
│   └── models/
│       └── dust_segmenter.onnx Trained model (see "Get the model")
├── frontend/
│   └── index.html              Web UI (no build step)
├── notebooks/
│   └── solar-panel-dust-detection.ipynb   Training notebook (Kaggle)
└── README.md
```

## Get the model

The `.onnx` file is large, so it may not be stored in the repo directly. Place it at:

```
backend/models/dust_segmenter.onnx
```

Download link: _add your Git LFS / Release / Drive link here_

To use a different location, set the `MODEL_PATH` environment variable.

## Run it

Requires Python 3.9+.

```bash
cd backend
pip install fastapi uvicorn onnxruntime pillow numpy python-multipart
python app.py
```

Check the model loaded: open http://127.0.0.1:8000/model-info and look for `"model_loaded": true`.

Then open `frontend/index.html` in a browser (double-click it), choose a panel photo, and click **Analyze image**. Keep the backend terminal running.

Interactive API docs: http://127.0.0.1:8000/docs

## API

`POST /predict` with a multipart form field named `file` (an image).

```json
{
  "label": "dirty",
  "status": "Moderate",
  "coverage_percent": 23.41,
  "confidence": 87.2,
  "overlay_png_base64": "...",
  "mask_png_base64": "..."
}
```

| Field | Meaning |
|---|---|
| `coverage_percent` | Share of image pixels predicted as dust |
| `label` | `dirty` if coverage is at least 1%, otherwise `clean` |
| `status` | Clean (<1%), Light (<15%), Moderate (<40%), Heavy (40%+) |
| `confidence` | Mean model probability over the pixels marked as dust |
| `overlay_png_base64` | Photo with dust highlighted in amber |
| `mask_png_base64` | Binary dust mask (white = dust) |

Other endpoints: `GET /` (health), `GET /model-info` (model input/output shapes).

## Model

- Task: binary semantic segmentation of dust on solar panels
- Dataset: Roboflow `solar-dust` (COCO polygons), 5961 train / 524 validation / 270 test images. Only the `dust` class is used as the mask.
- Input: RGB, 256x256, pixel values divided by 255. ImageNet mean/std normalization is built into the model, so do not normalize again.
- Output: one-channel logits. The backend applies a sigmoid and a 0.5 threshold.
- Training: Adam, BCE + Dice loss, mixed precision, early stopping on validation IoU.

Three architectures were compared. UNet-ResNet34 was selected by validation IoU.

| Model | Test IoU | Test Dice | Precision | Recall | Params |
|---|---|---|---|---|---|
| UNet (scratch) | 0.5647 | 0.7218 | 0.7203 | 0.7234 | 7.8M |
| **UNet-ResNet34** (served) | 0.5705 | 0.7265 | 0.6935 | 0.7629 | 24.4M |
| DeepLabV3+ ResNet34 | 0.5880 | 0.7406 | 0.7188 | 0.7636 | 22.4M |

## Limitations

- Segmentation quality is moderate (test IoU about 0.57). Thin or faint dust can be missed.
- Results depend on the photo: strong glare, shadows, or images that are not mostly solar panel can distort the coverage percentage.
- Coverage is measured over the whole image, not only the panel area.

## Retrain

Open `notebooks/solar-panel-dust-detection.ipynb` on Kaggle, add a Roboflow API key as the secret `roboflow_api_key`, and run all cells. The notebook writes `dust_segmenter.onnx` to its output folder.
