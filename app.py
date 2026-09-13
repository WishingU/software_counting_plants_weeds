"""Interactive Streamlit frontend for plant detection and segmentation models."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "runs" / "colab_50epoch" / "best.pt"
SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png"]


def discover_models() -> list[Path]:
    """Find local Ultralytics weights, preferring trained best checkpoints."""
    candidates = set(BASE_DIR.glob("*.pt"))
    for directory in (BASE_DIR / "models", BASE_DIR / "runs"):
        if directory.is_dir():
            candidates.update(directory.rglob("*.pt"))
    return sorted(
        (path.resolve() for path in candidates if path.is_file()),
        key=lambda path: (path.name != "best.pt", str(path).lower()),
    )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str, modified_ns: int) -> YOLO:
    """Cache a model until the selected weights file changes on disk."""
    del modified_ns
    return YOLO(weights_path)


st.set_page_config(page_title="Plant & Weed Analyzer", layout="wide")
st.title("Plant & Weed Analyzer")
st.write("Select a model, adjust inference parameters, and upload a field image.")

models = discover_models()
if not models:
    st.error("No .pt model weights were found in the project root, models/, or runs/.")
    st.stop()

default_resolved = DEFAULT_WEIGHTS.resolve()
default_index = models.index(default_resolved) if default_resolved in models else 0

with st.sidebar:
    st.header("Model")
    selected_weights = st.selectbox(
        "Project model",
        options=models,
        index=default_index,
        format_func=display_path,
        help="Models are discovered from the project root, models/, and runs/.",
    )
    custom_weights = st.text_input(
        "Custom model path (optional)",
        placeholder=r"E:\path\to\best.pt",
        help="When set, this path overrides the project model selected above.",
    ).strip()

    if custom_weights:
        custom_path = Path(custom_weights).expanduser()
        weights_path = custom_path if custom_path.is_absolute() else BASE_DIR / custom_path
    else:
        weights_path = selected_weights
    weights_path = weights_path.resolve()

    st.header("Inference parameters")
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider(
        "IoU threshold",
        0.05,
        0.95,
        0.30,
        0.05,
        help="Lower values suppress overlapping predictions more aggressively.",
    )
    image_size = st.select_slider(
        "Input image size",
        options=[320, 480, 640, 800, 1024, 1280],
        value=640,
    )
    max_detections = st.number_input("Maximum detections", 1, 3000, 300, 10)

    device_options = {"Auto": None, "CPU": "cpu"}
    if torch.cuda.is_available():
        device_options[f"GPU 0 - {torch.cuda.get_device_name(0)}"] = "0"
    selected_device = st.selectbox("Inference device", options=list(device_options))
    device = device_options[selected_device]

    st.divider()
    st.caption(f"Selected weights: `{display_path(weights_path)}`")

uploaded_file = st.file_uploader(
    "Upload a field image",
    type=SUPPORTED_UPLOAD_TYPES,
    help="Supported formats: JPG, JPEG, and PNG.",
)

if uploaded_file is not None:
    preview = Image.open(uploaded_file).convert("RGB")
    _, preview_column, _ = st.columns([1, 2, 1])
    with preview_column:
        st.image(preview, caption="Uploaded image", width="stretch")

    if st.button("Run analysis", type="primary", use_container_width=True):
        if not weights_path.is_file():
            st.error(f"Model weights not found: {weights_path}")
            st.stop()

        try:
            with st.spinner(f"Loading {display_path(weights_path)} and running inference..."):
                model = load_model(str(weights_path), weights_path.stat().st_mtime_ns)
                predict_args = {
                    "source": np.asarray(preview),
                    "conf": confidence,
                    "iou": iou,
                    "imgsz": image_size,
                    "max_det": int(max_detections),
                    "verbose": False,
                }
                if device is not None:
                    predict_args["device"] = device
                result = model.predict(**predict_args)[0]
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

        names = {int(index): str(name) for index, name in model.names.items()}
        counts = {name: 0 for name in names.values()}
        if result.boxes is not None:
            for class_id in result.boxes.cls.int().cpu().tolist():
                counts[names[class_id]] = counts.get(names[class_id], 0) + 1

        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        result_column, count_column = st.columns([3, 1])
        with result_column:
            st.subheader("Analysis result")
            st.image(annotated_rgb, caption="Model predictions", width="stretch")
        with count_column:
            st.subheader("Counts")
            st.metric("Total", sum(counts.values()))
            for name, count in counts.items():
                st.metric(name.replace("_", " ").title(), count)

            st.subheader("Model information")
            st.write(f"Task: `{model.task}`")
            st.write(f"Classes: `{', '.join(names.values())}`")
            st.write(f"Device: `{selected_device}`")
