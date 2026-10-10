# Project handover

This document summarizes the maintained project state on `main`. Model quality,
local execution, and public deployment status must be verified separately.

## Current application

`app.py` provides an integrated Streamlit and Ultralytics YOLO application for
image upload, optional green-vegetation enhancement, inference, annotated
results, and per-class counting.

The sidebar exposes three repository-managed models:

| Application option | Checkpoint | Confidence | IoU | Input size |
| --- | --- | ---: | ---: | ---: |
| Species Detection | `models/species/yolov8s_full.pt` | 0.40 | 0.70 | 768 |
| Broadleaf Detection | `models/broadleaf/yolo26m_broadleaf.pt` | 0.40 | 0.70 | 768 |
| Crop & Weed Counting | `models/counting/yolov8n-100e.pt` | 0.25 | 0.30 | 640 |

Species Detection is selected by default. The application loads the selected
checkpoint automatically, caches model resources, and reports unresolved Git
LFS pointers. It does not accept arbitrary model uploads.

## Available workflows

- Three-model inference and counting through `app.py`.
- Crop/weed command-line inference through `count_plants.py`.
- Four-species image or folder inference through
  `scripts/predict_species_yolov8s.py`.
- Optional green-dominance enhancement in the Streamlit and crop/weed CLI
  workflows; it is disabled by default.
- Dataset conversion, annotation diagnostics, threshold audits, image
  preparation, training, and evaluation through `src/plant_counter/` and
  `scripts/`.
- Fast mocked unit tests under `tests/` that do not require model downloads,
  a GPU, full training, or full-dataset inference.

## Model storage and Git LFS

All `.pt` files are configured for Git LFS:

```gitattributes
*.pt filter=lfs diff=lfs merge=lfs -text
```

New checkouts should run:

```powershell
git lfs install
git lfs pull
```

Normal Git history contains small LFS pointers. The working tree must contain
the complete model binaries before inference. `YOLO(model_path)` only loads a
local checkpoint and does not retrieve LFS objects.

The deployed application checkpoints are:

- `models/species/yolov8s_full.pt`
- `models/broadleaf/yolo26m_broadleaf.pt`
- `models/counting/yolov8n-100e.pt`

Additional comparison and historical checkpoints are documented in
`models/README.md`. Generated training runs belong under `runs/` or `outputs/`
and must not be promoted wholesale into Git.

## Evaluation guidance

- Keep checkpoint, confidence, IoU, input size, dataset split, and dataset
  version fixed when comparing results.
- The established Crop & Weed Counting defaults are confidence `0.25`, IoU
  `0.30`, and input size `640`.
- The source annotations have known gaps. Manual audits found that some
  apparent false positives were real plants missing from the labels.
- Raising confidence solely to improve measured precision can increase
  undercounting.
- Unit tests that mock YOLO verify logic and argument forwarding; they do not
  establish real-model accuracy, GPU compatibility, or successful deployment.
- Real-model evaluation should use an independent representative test set
  whenever possible.

Broadleaf dataset construction, fine-tuning, and evaluation details are kept
under `docs/`, including `MODEL_DEVELOPMENT_REPORT.md`,
`SPARSE_MIXED_FINETUNING.md`, and `CROPANDWEED_BROADLEAF_MAPPING.md`.

## Setup and testing

The standard Windows development setup is:

```powershell
conda activate plant-count
python -m pip install -e ".[app,tools]" --no-build-isolation
python -m pip install -r requirements-dev.txt
```

Run the fast automated test suite with:

```powershell
python -m pytest -v -m "not slow and not model and not acceptance"
```

Run real-model or acceptance checks separately and record the checkpoint and
inference settings used.

## Deployment

The project uses a single Streamlit and YOLO application rather than a separate
model API service. A deployment must retrieve the real Git LFS objects before
the application starts. Repository tracking alone does not prove that a hosted
environment loaded the weights or completed inference successfully.

The historical Streamlit deployment may use a deployment-specific branch and
may not match the latest `main`. Verify its branch, logs, checkpoint sizes,
available model options, and one real inference before treating it as current.

## Known follow-up work

- Build and preserve a genuinely independent test set for final comparison.
- Expand under-represented species and target-domain examples.
- Continue resolving annotation gaps that distort precision and counting
  measurements.
- Add automated Streamlit smoke or acceptance coverage without making the
  default unit suite load real weights.
- Keep `README.md`, `models/README.md`, and this handover aligned when
  application models or defaults change.
