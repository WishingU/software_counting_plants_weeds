# Windows 11 Setup Guide

This project has been configured for the current computer:

- Project directory: `E:\win11_iso\software_counting_plants_weeds`
- Conda environment: `plant-count`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Dataset configuration: `data\yolo\data.yaml`

## Activate the Environment

```powershell
cd E:\win11_iso\software_counting_plants_weeds
conda activate plant-count
```

## Run Sample Inference

```powershell
python .\count_plants.py .\data\raw_images\sample.png --device 0
```

## Launch the Web Application

```powershell
python -m streamlit run .\app.py
```

## Train from YOLOv8n

```powershell
python .\scripts\train.py --model yolov8n.pt --epochs 100 --batch 8 --device 0
```

Training results are saved to `runs\crop_weed_v1`. If GPU memory is insufficient,
change `--batch 8` to `--batch 4`.

## Continue Training from an Existing Model

```powershell
python .\scripts\train.py `
  --model .\runs\colab_50epoch\best.pt `
  --epochs 50 `
  --batch 8 `
  --device 0 `
  --name crop_weed_finetune_v1
```

## Evaluate Counting on the Independent Test Set

```powershell
python .\scripts\evaluate_counts.py `
  --weights .\runs\crop_weed_v1\weights\best.pt `
  --split test `
  --device 0
```

Run the following command to calculate standard detection metrics:

```powershell
yolo detect val `
  model=E:/win11_iso/software_counting_plants_weeds/runs/crop_weed_v1/weights/best.pt `
  data=E:/win11_iso/software_counting_plants_weeds/data/yolo/data.yaml `
  split=test `
  device=0 `
  workers=0
```

The `test` split is reserved for final evaluation and must not be used for
training or hyperparameter selection.
