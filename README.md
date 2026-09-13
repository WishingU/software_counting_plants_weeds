# Plant and Weed Counter

This project provides reproducible tools for crop/weed counting and four-species
plant identification with Ultralytics YOLO models. It includes a Streamlit
interface, command-line inference, dataset utilities, evaluation scripts,
curated checkpoints, and automated tests.

[中文说明](README_cn.md) · [Windows setup](README_WINDOWS.md) ·
[Annotation guide](docs/ANNOTATION_GUIDE.en.md)

## Project layout

~~~text
app.py                 Streamlit application
count_plants.py        Command-line inference
src/plant_counter/     Maintained dataset, audit, training, and evaluation code
scripts/               Data conversion, image preparation, and diagnostics
tests/                 Automated unit tests
configs/               Portable configuration examples and annotation rules
data/                  Dataset YAML files and one sample image
models/counting/       Curated crop/weed checkpoints
models/species/        Curated four-species checkpoints
runs/, outputs/        Generated local results (not committed)
notebooks/             Exploratory image-splitting notebooks
~~~

## Setup

~~~powershell
conda activate plant-count
python -m pip install -e ".[app,tools]" --no-build-isolation
~~~

For Streamlit deployment only:

~~~powershell
python -m pip install -r requirements.txt
~~~

## Run the application

~~~powershell
python -m streamlit run app.py
~~~

The application discovers checkpoints under models/ and runs/. The default is
models/counting/yolov8n-50e.pt.

Enable **Enhance green vegetation** in the inference sidebar to mildly boost
pixels that are already green-dominant. The preview shows the exact image sent
to the model. The option is off by default.

## Command-line inference

~~~powershell
python count_plants.py data/raw_images/sample.png --device 0
python count_plants.py data/raw_images/sample.png --device 0 --enhance-green
~~~

Use --weights to select another checkpoint. Use --enhance-green to enable the
same optional preprocessing available in the web application.

## Train and evaluate

Install the project first, then use the maintained package entry points:

~~~powershell
python -m plant_counter.train --data data/yolo/data.yaml --model yolov8n.pt --project outputs/runs/detect --name crop_weed_v1

python -m plant_counter.evaluate --model models/counting/yolov8n-50e.pt --data data/yolo/data.yaml --split test --output-dir outputs/runs/evaluation/crop_weed
~~~

Specialized audits and comparison tools remain under scripts/.

## Image preparation

~~~powershell
python scripts/split_images.py data/source data/split --grid 3
python scripts/split_images.py data/source data/wild_split --grid 2 --name-contains wild
python scripts/heic_to_jpeg.py data/heic data/jpeg
~~~

## Tests

~~~powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
~~~

The public Streamlit deployment is available for functional testing at
[crop-weed-counter-afridi.streamlit.app](https://crop-weed-counter-afridi.streamlit.app/).
Model accuracy should be assessed separately with the evaluation workflow.
