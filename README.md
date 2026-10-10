# Plant and Weed Counter

This project provides reproducible tools for crop/weed counting, broadleaf-weed
detection, and four-species plant identification with Ultralytics YOLO models.
It includes a Streamlit interface, command-line inference, dataset utilities,
evaluation scripts, curated checkpoints, and automated tests.

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
models/broadleaf/      Curated broadleaf-weed checkpoint
runs/, outputs/        Generated local results (not committed)
notebooks/             Exploratory image-splitting notebooks
~~~

## Quick start

### Prerequisites

- Git
- Git LFS
- Python 3.10 or later (Python 3.11 recommended)
- pip

### Clone the repository and download model weights

~~~powershell
git lfs install
git clone https://github.com/WishingU/software_counting_plants_weeds.git
cd software_counting_plants_weeds
git lfs pull
~~~

When Git LFS is installed before cloning, Git normally downloads the model
objects during `git clone`. Running `git lfs pull` explicitly ensures that all
checkpoint files are present. A small Git LFS pointer is not a usable model
file, and Ultralytics `YOLO(model_path)` does not download LFS objects.

### Install dependencies

Using an isolated Python environment is recommended. On Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[app,tools]" --no-build-isolation
~~~

Project developers who use the existing Conda environment can instead run:

~~~powershell
conda create -n plant-count python=3.11 -y
conda activate plant-count
python -m pip install -e ".[app,tools]" --no-build-isolation
~~~

For a minimal Streamlit-only installation, use the following command instead
of the editable development installation above:

~~~powershell
python -m pip install -r requirements.txt
~~~

## Run the application

~~~powershell
python -m streamlit run app.py
~~~

Open the local URL printed by Streamlit, normally
`http://localhost:8501`.

The application provides three repository-managed model options in the
sidebar:

| Model option | Checkpoint | Purpose |
| --- | --- | --- |
| Species Detection | `models/species/yolov8s_full.pt` | Identify wheat, wild oat, brome grass, and barley grass |
| Broadleaf Detection | `models/broadleaf/yolo26m_broadleaf.pt` | Detect and count broadleaf weeds |
| Crop & Weed Counting | `models/counting/yolov8n-100e.pt` | Detect and count crop and weed classes |

Species Detection is selected by default. Switching the sidebar selection
loads the corresponding repository-managed checkpoint; users do not upload
model weights. Loaded models are cached for reuse within the running
Streamlit process.

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

python -m plant_counter.evaluate --model models/counting/yolov8n-100e.pt --data data/yolo/data.yaml --split test --output-dir outputs/runs/evaluation/crop_weed
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
python -m pip install -r requirements-dev.txt
python -m pytest -v -m "not slow and not model and not acceptance"
~~~

Pytest also discovers and runs the existing `unittest.TestCase` tests. Real-model
and acceptance tests are opt-in; see `AGENTS.md` for the test levels and evidence
requirements.

The public Streamlit deployment is available for functional testing at
[crop-weed-counter-afridi.streamlit.app](https://crop-weed-counter-afridi.streamlit.app/).
Model accuracy should be assessed separately with the evaluation workflow.
