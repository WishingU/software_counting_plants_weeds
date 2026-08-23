# Plant and Weed Detection and Counting

[中文 README](README_cn.md) | [English annotation guide](docs/ANNOTATION_GUIDE.en.md)

This repository provides a reproducible pipeline for whole-plant detection and
counting of four seedling classes:

- wheat
- wild oat
- brome grass
- barley grass

One physical plant is one detection instance and contributes one count. Root
point detection, instance segmentation, and local crop classification are not
part of the current primary pipeline.

## Project layout

```text
E:/cits3200_project/
  src/plant_counter/                # audit, build, train, and evaluation code
  tests/                            # automated tests
  configs/                          # dataset and annotation configuration
  docs/                             # project and annotation documentation
  data/raw/                         # original downloaded data
  data/annotations/                 # source COCO annotations
  data/processed/                   # YOLO images, labels, and build report
  models/                           # pretrained weights
  outputs/runs/                     # training, evaluation, and audit outputs
  archives/                         # compressed backups
  legacy/                           # retired one-off scripts
```

Images, model weights, and generated runs are excluded from Git.

## 0. Environment

```powershell
cd E:\cits3200_project
conda activate plant-count
python -m pip install -e . --no-build-isolation
```

Verify the runtime before training:

```powershell
python -c "import cv2, torch; print(cv2.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If the editable package is not installed, use this temporary fallback:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

## 1. Audit source annotations

```powershell
python -m plant_counter.audit `
  --annotations-dir .\data\annotations `
  --dataset-dir .\data\processed `
  --rules .\configs\annotation_rules.yaml `
  --output-dir .\outputs\runs\annotation_audit
```

The audit writes:

- `annotation_audit.json`: dataset totals, box statistics, and issue counts.
- `annotation_issues.csv`: individual errors and review warnings.

Fix all `error` rows before building the dataset. Inspect `warning` rows, but
do not automatically discard them: large boxes can be valid for mature plants.

## 2. Build YOLO labels

```powershell
python -m plant_counter.dataset `
  --annotations-dir .\data\annotations `
  --dataset-dir .\data\processed `
  --yaml .\configs\grass.yaml `
  --report .\data\processed\build_report.json
```

Expected totals for the current dataset are:

| Split | Images | Instances |
|---|---:|---:|
| train | 910 | 10,442 |
| val | 239 | 2,848 |

The former 60-image `test` directory was a copied validation subset and was
removed during workspace cleanup. A future test set must contain genuinely
independent images.

## 3. Train a model

```powershell
python -m plant_counter.train `
  --data .\configs\grass.yaml `
  --model .\models\yolo26n.pt `
  --epochs 100 `
  --imgsz 1280 `
  --batch -1 `
  --device 0 `
  --project .\outputs\runs\detect `
  --name whole_plant_v1
```

The training wrapper fixes the seed, enables deterministic mode, uses a
Windows-safe worker setting, and stores environment metadata. The best weights
will normally be located at:

```text
E:/cits3200_project/outputs/runs/detect/whole_plant_v1/weights/best.pt
```

## 4. Evaluate detection and counting

```powershell
python -m plant_counter.evaluate '
  --model .\outputs\runs\detect\whole_plant_v1\weights\best.pt `
  --data .\configs\grass.yaml `
  --split val `
  --conf 0.25 `
  --iou 0.7 `
  --device 0 `
  --output-dir .\outputs\runs\evaluation\whole_plant_v1
```

Evaluation reports Precision, Recall, mAP50, mAP50-95, overall and per-class
count MAE, RMSE, bias, exact-match rate, and per-image counts. Use mAP50-95 to
compare detector quality, then use count MAE and bias to judge the counting
application.

## 5. Recommended iteration loop

1. Audit annotations and correct structural errors.
2. Build and freeze one dataset version.
3. Train a baseline without changing several variables at once.
4. Inspect images with the largest absolute count errors.
5. Categorise failures as missed plants, duplicate detections, class confusion,
   occlusion, or domain shift.
6. Correct or add data for the dominant failure mode and train a new version.
7. Use a genuinely independent test set only for the final report.

## Tests

```powershell
conda activate plant-count
$env:PYTHONPATH = "$PWD\src"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```
