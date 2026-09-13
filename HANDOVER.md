# Project handover

This document describes the consolidated project state. Branch-specific notes
have been folded into the maintained structure.

## Available workflows

- Crop/weed counting through app.py and count_plants.py.
- Four-species dataset, training, and evaluation through src/plant_counter/.
- Dataset conversion, annotation diagnostics, threshold audits, and image
  preparation under scripts/.
- Automated tests under tests/.

## Curated checkpoints

- models/counting/yolov8n-50e.pt: default crop/weed model.
- models/counting/yolov8n-100e.pt: longer-trained crop/weed model.
- models/counting/yolov8s-comparison.pt: larger counting comparison.
- models/species/yolov8n-50e.pt: four-species baseline.
- models/species/yolov8n-100e.pt: preferred four-species model.

Generated runs belong in runs/ or outputs/; do not promote an entire run
directory into Git.

## Validated counting settings

- Confidence threshold: 0.25.
- NMS IoU threshold: 0.3.
- The original annotations have known gaps. Manual audits found that many
  apparent false positives were real plants missing from the labels.
- Raising confidence solely to improve raw precision can worsen real counting
  accuracy through undercounting.

The specialized scripts under scripts/ preserve the original audit and
corrected-count workflows.

## Known follow-up work

- Compare curated checkpoints on one fixed independent test split and document
  the selected production model.
- Improve separation among wild oat, brome grass, and barley grass.
- Add automated smoke coverage for the Streamlit inference path.
- Decide whether large curated checkpoints should move to Git LFS or release
  assets before the repository grows further.

## Deployment

A test deployment exists at
[crop-weed-counter-afridi.streamlit.app](https://crop-weed-counter-afridi.streamlit.app/).
Deployment functionality and model accuracy should be verified independently.
