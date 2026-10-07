# Windows 11 setup

## Environment

~~~powershell
cd E:\win11_iso\software_counting_plants_weeds
conda activate plant-count
python -m pip install -e ".[app,tools]" --no-build-isolation
~~~

## Sample inference

~~~powershell
python count_plants.py data\raw_images\sample.png --device 0
python count_plants.py data\raw_images\sample.png --device 0 --enhance-green
~~~

The default checkpoint is models\counting\yolov8n-50e.pt. Select another
model with --weights. Add --enhance-green to enable optional green enhancement.

## Web application

~~~powershell
python -m streamlit run app.py
~~~

Use the **Enhance green vegetation** switch in the inference sidebar to turn
green enhancement on or off. It is off by default.

## Training

~~~powershell
python -m plant_counter.train --data data\yolo\data.yaml --model yolov8n.pt --epochs 100 --batch 8 --device 0 --project outputs\runs\detect --name crop_weed_v1
~~~

If GPU memory is insufficient, reduce --batch 8 to --batch 4.

To continue from a curated checkpoint:

~~~powershell
python -m plant_counter.train --data data\yolo\data.yaml --model models\counting\yolov8n-50e.pt --epochs 50 --batch 8 --device 0 --project outputs\runs\detect --name crop_weed_finetune_v1
~~~

## Final evaluation

~~~powershell
python -m plant_counter.evaluate --model outputs\runs\detect\crop_weed_v1\weights\best.pt --data data\yolo\data.yaml --split test --device 0 --output-dir outputs\runs\evaluation\crop_weed_v1
~~~

Reserve the test split for final evaluation. Use train/validation data for
model selection and parameter tuning.
