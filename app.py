
"""Streamlit application for plant detection, identification, and counting."""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

from plant_counter.preprocess import enhance_green


# ============================================================
# Model configuration
# ============================================================

SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png"]

SPECIES_ORDER = [
    "wheat",
    "wild oat",
    "brome grass",
    "barley grass",
]

MODEL_OPTIONS = {
    "Species Detection": {
        "path": BASE_DIR / "models" / "species" / "yolov8s_full.pt",
        "description": "Identify wheat, wild oat, brome grass, and barley grass.",
        "conf": 0.40,
        "iou": 0.70,
        "imgsz": 768,
    },
    "Broadleaf Detection": {
        "path": BASE_DIR / "models" / "broadleaf" / "yolo26m_broadleaf.pt",
        "description": "Detect and count broadleaf weeds.",
        "conf": 0.40,
        "iou": 0.70,
        "imgsz": 768,
    },
    "Crop & Weed Counting": {
        "path": BASE_DIR / "models" / "counting" / "yolov8n-100e.pt",
        "description": "Detect and count crops and weeds.",
        "conf": 0.25,
        "iou": 0.30,
        "imgsz": 640,
    },
}


# ============================================================
# Model loading and LFS validation
# ============================================================

def validate_model_file(path: Path):
    """Check that model weights exist and are not Git LFS pointers."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {path}"
        )

    file_size = path.stat().st_size

    # An unresolved Git LFS pointer is a small text file,
    # not a usable PyTorch model.
    if file_size < 512:
        content = path.read_bytes()

        if content.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            raise RuntimeError(
                f"{path.name} is a Git LFS pointer, "
                "not the actual model weights. "
                "Make sure Git LFS files are downloaded "
                "during deployment."
            )

    if file_size == 0:
        raise RuntimeError(
            f"Model file is empty: {path}"
        )


@st.cache_resource(show_spinner=False, max_entries=2)
def load_model(model_path: str, modified_ns: int) -> YOLO:
    """Load and cache a YOLO model based on its file path and modification time."""

    del modified_ns
    return YOLO(model_path)


def get_model_names(model):
    """Return class names as a dictionary."""

    names = model.names

    if isinstance(names, dict):
        return {
            int(index): str(name)
            for index, name in names.items()
        }

    return {
        index: str(name)
        for index, name in enumerate(names)
    }


# ============================================================
# Streamlit page
# ============================================================

st.set_page_config(
    page_title="Plant & Weed Analyzer",
    layout="wide",
)

st.title("Plant & Weed Analyzer")

st.write(
    "Upload a field image to detect, identify, "
    "and count plants using trained YOLO models."
)


# ============================================================
# Sidebar: model selection and inference settings
# ============================================================

with st.sidebar:

    st.header("Model Selection")

    selected_model = st.selectbox(
        "Choose detection model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        key="selected_model",
    )

    model_config = MODEL_OPTIONS[selected_model]
    weights_path = model_config["path"]

    st.caption(model_config["description"])

    st.divider()

    st.header("Inference Parameters")

    # Each model remembers its own settings.
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=model_config["conf"],
        step=0.01,
        key=f"conf_{selected_model}",
    )

    iou = st.slider(
        "IoU threshold",
        min_value=0.05,
        max_value=0.95,
        value=model_config["iou"],
        step=0.05,
        key=f"iou_{selected_model}",
        help="Lower values suppress overlapping predictions more aggressively.",
    )

    image_size = st.select_slider(
        "Input image size",
        options=[
            320,
            480,
            640,
            768,
            800,
            1024,
            1280,
        ],
        value=model_config["imgsz"],
        key=f"imgsz_{selected_model}",
    )

    max_detections = st.number_input(
        "Maximum detections",
        min_value=1,
        max_value=3000,
        value=300,
        step=10,
    )

    enhance_green_input = st.toggle(
        "Enhance green vegetation",
        value=False,
        help=(
            "Apply mild green-dominance enhancement "
            "before inference."
        ),
    )

    device_options = {
        "Auto": None,
        "CPU": "cpu",
    }

    if torch.cuda.is_available():
        device_options[
            f"GPU 0 - {torch.cuda.get_device_name(0)}"
        ] = "0"

    selected_device = st.selectbox(
        "Inference device",
        options=list(device_options),
    )

    device = device_options[selected_device]

    st.divider()

    st.caption(f"Model file: `{weights_path.name}`")


# ============================================================
# Automatically load selected model on page entry or switch
# ============================================================

try:
    validate_model_file(weights_path)

    with st.spinner(f"Loading {selected_model}..."):
        model = load_model(
            str(weights_path.resolve()),
            weights_path.stat().st_mtime_ns,
        )

    st.sidebar.success(f"Loaded: {weights_path.name}")

except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()


# ============================================================
# Main interface
# ============================================================

st.subheader("Active Model")

st.info(
    f"**{selected_model}** | "
    f"{model_config['description']}"
)

uploaded_file = st.file_uploader(
    "Upload a field image",
    type=SUPPORTED_UPLOAD_TYPES,
    help="Supported formats: JPG, JPEG, and PNG.",
)


# ============================================================
# Image preprocessing
# ============================================================

if uploaded_file is not None:

    try:
        preview = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read image: {exc}")
        st.stop()

    original_rgb = np.asarray(preview)

    inference_rgb = (
        enhance_green(original_rgb)
        if enhance_green_input
        else original_rgb
    )

    preview_caption = (
        "Inference input (green enhancement on)"
        if enhance_green_input
        else "Inference input (original image)"
    )

    _, preview_column, _ = st.columns([1, 2, 1])

    with preview_column:
        st.image(
            inference_rgb,
            caption=preview_caption,
            width="stretch",
        )

    # ========================================================
    # Run prediction
    # ========================================================

    if st.button(
        "Run analysis",
        type="primary",
        use_container_width=True,
    ):

        try:
            with st.spinner(
                f"Running inference with {selected_model}..."
            ):

                # Convert RGB to BGR for Ultralytics.
                inference_bgr = cv2.cvtColor(
                    inference_rgb,
                    cv2.COLOR_RGB2BGR,
                )

                predict_args = {
                    "source": inference_bgr,
                    "conf": confidence,
                    "iou": iou,
                    "imgsz": image_size,
                    "max_det": int(max_detections),
                    "verbose": False,
                }

                if device is not None:
                    predict_args["device"] = device

                result = model.predict(
                    **predict_args
                )[0]

        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

        # ====================================================
        # Count detections
        # ====================================================

        names = get_model_names(model)

        counts = {
            name: 0
            for name in names.values()
        }

        if result.boxes is not None:

            class_ids = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            for class_id in class_ids:

                class_name = names[class_id]

                counts[class_name] = (
                    counts.get(class_name, 0) + 1
                )

        # ====================================================
        # Draw prediction boxes
        # ====================================================

        annotated_bgr = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB,
        )

        result_column, count_column = st.columns([3, 1])

        with result_column:

            st.subheader("Analysis Result")

            st.image(
                annotated_rgb,
                caption="Model predictions",
                width="stretch",
            )

        # ====================================================
        # Display detection counts
        # ====================================================

        with count_column:

            st.subheader("Detection Counts")

            total_count = sum(counts.values())

            st.metric(
                "Total Plants",
                total_count,
            )

            # Show species in a consistent order.
            if selected_model == "Species Detection":

                for species in SPECIES_ORDER:
                    if species in counts:
                        st.metric(
                            species.title(),
                            counts[species],
                        )

            # Display all other class counts.
            for name, count in counts.items():

                if (
                    selected_model == "Species Detection"
                    and name in SPECIES_ORDER
                ):
                    continue

                display_name = (
                    name.replace("_", " ").title()
                )

                st.metric(
                    display_name,
                    count,
                )

            # ================================================
            # Model information
            # ================================================

            st.subheader("Model Information")

            st.write(f"Model: `{weights_path.name}`")

            st.write(f"Task: `{model.task}`")

            st.write(
                f"Classes: `{', '.join(names.values())}`"
            )

            st.write(f"Device: `{selected_device}`")

            st.write(f"Confidence: `{confidence}`")

            st.write(f"IoU: `{iou}`")

            st.write(f"Image size: `{image_size}`")

            st.write(
                "Green enhancement: "
                f"`{'On' if enhance_green_input else 'Off'}`"
            )
