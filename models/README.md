# Curated models

This directory contains checkpoints intentionally distributed with the project.
Generated training runs belong under runs/ or outputs/ and are ignored by Git.

## Counting models

| File | Purpose |
| --- | --- |
| counting/yolov8n-100e.pt | Default crop/weed counter |
| counting/yolov8n-50e.pt | Shorter-trained crop/weed counter |
| counting/yolov8s-comparison.pt | Larger comparison model |

## Species models

| File | Purpose |
| --- | --- |
| species/yolov8n-50e.pt | Four-species baseline |
| species/yolov8n-100e.pt | Preferred four-species checkpoint |

When adding a model, use a descriptive filename and update this table. Do not
commit entire training-run directories.
