"""Interactive Streamlit frontend for plant detection and segmentation models."""

import hashlib
from pathlib import Path
import sys
import tempfile


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# Allow ``streamlit run app.py`` to import the local src-layout package even
# when the project has not been installed in editable mode.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

from plant_counter.preprocess import enhance_green


SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png"]

SPECIES_ORDER = [
    "wheat",
    "wild oat",
    "brome grass",
    "barley grass",
]

# Final full-dataset species-identification model.
DEFAULT_WEIGHTS_PATH = (
    BASE_DIR / "models" / "species" / "yolov8s_full.pt"
)


def materialize_uploaded_model(uploaded_file) -> Path:
    """Persist uploaded model bytes under a content-addressed temporary path."""
    payload = uploaded_file.getvalue()
    digest = hashlib.sha256(payload).hexdigest()

    model_dir = (
        Path(tempfile.gettempdir())
        / "plant_weed_analyzer_models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = model_dir / f"{digest}.pt"

    if not model_path.exists():
        model_path.write_bytes(payload)

    return model_path


@st.cache_resource(show_spinner=False)
def load_model(
    weights_path: str,
    modified_ns: int,
) -> YOLO:
    """Cache a model until the selected weights file changes on disk."""
    del modified_ns

    path = Path(weights_path)

    # Use a project-relative path for models stored inside this repository.
    # This avoids Windows path problems caused by characters such as
    # the apostrophe in the local project directory name.
    try:
        model_path = path.relative_to(BASE_DIR)
    except ValueError:
        model_path = path

    return YOLO(str(model_path))


st.set_page_config(
    page_title="Plant & Weed Analyzer",
    layout="wide",
)

st.title("Plant & Weed Analyzer")

st.write(
    "Upload a field image to detect, identify, "
    "and count wheat and weed species."
)


with st.sidebar:
    st.header("Model")

    uploaded_model = st.file_uploader(
        "Load model weights",
        type=["pt"],
        key="model_weights",
        help=(
            "Optionally choose a trusted Ultralytics .pt weights file "
            "from your computer. If no model is uploaded, the final "
            "project YOLOv8s species-identification model is used. "
            "Model files can contain executable data, so do not load "
            "untrusted files."
        ),
    )

    # Use a custom uploaded model when provided.
    # Otherwise use the final project species-identification model.
    weights_path = (
        materialize_uploaded_model(uploaded_model)
        if uploaded_model is not None
        else DEFAULT_WEIGHTS_PATH
    )

    st.header("Inference parameters")

    confidence = st.slider(
        "Confidence threshold",
        0.0,
        1.0,
        0.40,
        0.01,
    )

    iou = st.slider(
        "IoU threshold",
        0.05,
        0.95,
        0.70,
        0.05,
        help=(
            "Lower values suppress overlapping "
            "predictions more aggressively."
        ),
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
        value=768,
    )

    max_detections = st.number_input(
        "Maximum detections",
        1,
        3000,
        300,
        10,
    )

    enhance_green_input = st.toggle(
        "Enhance green vegetation",
        value=False,
        help=(
            "Apply a mild green-dominance enhancement before inference. "
            "Turn it off to use the original image."
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

    if uploaded_model is None:
        st.caption(
            "Default model: `yolov8s_full.pt`"
        )
        st.caption(
            "Upload another `.pt` file above to temporarily "
            "override the project model."
        )
    else:
        st.caption(
            f"Using custom model: `{uploaded_model.name}`"
        )
        st.caption(
            "Remove the uploaded file to return to "
            "`yolov8s_full.pt`."
        )


uploaded_file = st.file_uploader(
    "Upload a field image",
    type=SUPPORTED_UPLOAD_TYPES,
    help="Supported formats: JPG, JPEG, and PNG.",
)


if uploaded_file is not None:
    preview = Image.open(
        uploaded_file
    ).convert("RGB")

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

    _, preview_column, _ = st.columns(
        [1, 2, 1]
    )

    with preview_column:
        st.image(
            inference_rgb,
            caption=preview_caption,
            width="stretch",
        )

    if st.button(
        "Run analysis",
        type="primary",
        use_container_width=True,
    ):

        if not weights_path.is_file():
            st.error(
                f"Model weights not found: {weights_path}"
            )
            st.stop()

        model_name = (
            uploaded_model.name
            if uploaded_model is not None
            else DEFAULT_WEIGHTS_PATH.name
        )

        try:
            with st.spinner(
                f"Loading {model_name} "
                "and running inference..."
            ):
                model = load_model(
                    str(weights_path),
                    weights_path.stat().st_mtime_ns,
                )

                # Ultralytics interprets NumPy image sources as OpenCV BGR.
                # The uploaded PIL image and green enhancement pipeline use RGB,
                # so convert explicitly to avoid swapping red and blue channels.
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
            st.error(
                f"Inference failed: {exc}"
            )
            st.stop()

        names = {
            int(index): str(name)
            for index, name
            in model.names.items()
        }

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
                    counts.get(
                        class_name,
                        0,
                    )
                    + 1
                )

        annotated_bgr = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB,
        )

        result_column, count_column = (
            st.columns([3, 1])
        )

        with result_column:
            st.subheader(
                "Analysis result"
            )

            st.image(
                annotated_rgb,
                caption="Model predictions",
                width="stretch",
            )

        with count_column:
            st.subheader(
                "Species Counts"
            )

            total_count = sum(
                counts.values()
            )

            st.metric(
                "Total Plants",
                total_count,
            )

            # Show the four species in a consistent order
            # when using the species-identification model.
            for species in SPECIES_ORDER:
                if species in counts:
                    st.metric(
                        species.title(),
                        counts[species],
                    )

            # Keep compatibility with other models such as
            # older crop/weed counting or custom models.
            for name, count in counts.items():
                if name not in SPECIES_ORDER:
                    display_name = (
                        name
                        .replace(
                            "_",
                            " ",
                        )
                        .title()
                    )

                    st.metric(
                        display_name,
                        count,
                    )

            st.subheader(
                "Model information"
            )

            st.write(
                f"Model: `{model_name}`"
            )

            st.write(
                f"Task: `{model.task}`"
            )

            st.write(
                f"Classes: "
                f"`{', '.join(names.values())}`"
            )

            st.write(
                f"Device: "
                f"`{selected_device}`"
            )

            st.write(
                "Green enhancement: "
                f"`{'On' if enhance_green_input else 'Off'}`"
            )