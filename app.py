"""Web app for counting crop and weed plants in a photo.

Wraps the trained YOLOv8 model (see count_plants.py for the CLI version)
in a simple drag-and-drop web interface using Streamlit - no coding needed
to use it. Run locally with `streamlit run app.py`, or deploy to Streamlit
Community Cloud for a permanent public link.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "runs" / "colab_50epoch" / "best.pt"
CLASS_NAMES = ["crop", "weed"]
CONFIDENCE = 0.25
IOU = 0.3


@st.cache_resource
def load_model():
    if not WEIGHTS_PATH.exists():
        st.error(f"Model weights not found: {WEIGHTS_PATH}")
        st.stop()

    return YOLO(WEIGHTS_PATH)


st.title("Crop & Weed Counter")
st.write("Upload a field photo to count crop plants and weeds detected by our trained model.")

uploaded_file = st.file_uploader("Upload a field photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model = load_model()
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    results = model.predict(source=image_np, conf=CONFIDENCE, iou=IOU, verbose=False)
    result = results[0]

    counts = {name: 0 for name in CLASS_NAMES}
    for box in result.boxes:
        class_id = int(box.cls.item())
        counts[CLASS_NAMES[class_id]] += 1

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, caption="Detected plants", width="stretch")

    st.subheader("Counts")
    st.write(f"**Crop:** {counts['crop']}")
    st.write(f"**Weed:** {counts['weed']}")
    st.write(f"**Total:** {counts['crop'] + counts['weed']}")
