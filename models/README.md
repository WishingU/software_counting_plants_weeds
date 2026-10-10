# Curated models

This directory contains the model checkpoints intentionally distributed with
the project. All `.pt` files are managed by Git LFS through the repository's
`.gitattributes` configuration.

After cloning the repository, download the complete model binaries:

```powershell
git lfs install
git lfs pull
```

A file of approximately 132-133 bytes is an unresolved Git LFS pointer, not a
usable model. Ultralytics `YOLO(model_path)` loads an existing local checkpoint;
it does not download LFS objects or run `git lfs pull`.

## Models used by the Streamlit application

| Application option | File | Purpose | Default settings |
| --- | --- | --- | --- |
| Species Detection | `species/yolov8s_full.pt` | Identify wheat, wild oat, brome grass, and barley grass | `conf=0.40`, `iou=0.70`, `imgsz=768` |
| Broadleaf Detection | `broadleaf/yolo26m_broadleaf.pt` | Detect and count broadleaf weeds | `conf=0.40`, `iou=0.70`, `imgsz=768` |
| Crop & Weed Counting | `counting/yolov8n-100e.pt` | Detect and count crop and weed classes | `conf=0.25`, `iou=0.30`, `imgsz=640` |

Species Detection is selected when the web application opens.

## Additional counting checkpoints

| File | Purpose |
| --- | --- |
| `counting/yolov8n-50e.pt` | Shorter-trained crop/weed checkpoint |
| `counting/yolov8s-comparison.pt` | Larger crop/weed comparison checkpoint |

## Additional species checkpoints

| File | Purpose |
| --- | --- |
| `species/yolov8n-50e.pt` | Four-species baseline |
| `species/yolov8n-100e.pt` | Longer-trained YOLOv8n checkpoint |
| `species/yolov8s-100e.pt` | YOLOv8s comparison checkpoint |

## Repository conventions

- Store curated checkpoints under the appropriate `models/` subdirectory.
- Do not commit entire training-run directories. Generated runs belong under
  `runs/` or `outputs/`, which are ignored by Git.
- Update this file and the application model configuration when adding or
  replacing a deployed checkpoint.
- Keep the checkpoint, confidence, IoU, input size, and dataset split fixed
  when comparing model results.
- Only load `.pt` files from trusted sources. PyTorch checkpoints may contain
  executable serialized content.
