# Solar Panel Dust Detection

DATASET LINK - (https://universe.roboflow.com/august-4t80o/solar-dust)

Upload a photo of a solar panel and the app highlights the dusty area and reports what percentage of the image it covers.

A U-Net (ResNet34 encoder) segmentation model is trained on Kaggle, exported to ONNX, and served by a FastAPI backend. A single-page frontend sends images to the API and shows the result.

## How it works

```text
Browser (index.html)  --image-->  FastAPI (app.py)  -->  ONNX Runtime (UNet-ResNet34)
        ^                                                        |
        +------ coverage %, status, overlay, mask  <-------------+
```

## Project structure

```text
solar_dust_app/
├── backend/
│   ├── app.py
│   ├── dust_segmenter.onnx
│   └── dust_segmenter/
│       └── ... PyTorch model files
│
├── frontend/
│   └── index.html
│
├── models/
│   ├── best_DeepLabV3Plus_ResNet34/
│   │   └── ... model files
│   └── best_UNet_ResNet34/
│       └── ... model files
│
├── kaggle_output/
│   └── dust_segmenter.onnx
│
└── README.md
```

## Get the model

The `.onnx` file is large, so it may not be stored in the repo directly. Place it at:

```text
backend/dust_segmenter.onnx
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

Three architectures were compared. **UNet-ResNet34 was selected based on test IoU.**

| Model | Test Loss | Test IoU | Test Dice | Test Precision | Test Recall |
|---|---:|---:|---:|---:|---:|
| **UNet-ResNet34 (served)** | 0.579944 | **0.735649** | **0.845814** | 0.821875 | 0.877501 |
| DeepLabV3+ ResNet34 | 0.588410 | 0.696955 | 0.818536 | 0.789331 | 0.858587 |
| UNet (scratch) | 0.644043 | 0.676643 | 0.805514 | 0.803852 | 0.815055 |

**Best model based on Test IoU:** UNet-ResNet34.

## Limitations

- Segmentation quality is moderate. Test IoU for the selected model is approximately 0.736. Thin or faint dust can be missed.
- Results depend on the photo: strong glare, shadows, or images that are not mostly solar panel can distort the coverage percentage.
- Coverage is measured over the whole image, not only the panel area.

## Retrain

Open `notebooks/solar-panel-dust-detection.ipynb` on Kaggle, add a Roboflow API key as the secret `roboflow_api_key`, and run all cells. The notebook writes `dust_segmenter.onnx` to its output folder.
