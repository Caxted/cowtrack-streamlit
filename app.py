import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page setup
st.set_page_config(
    page_title="Cow Disease Detection",
    layout="centered"
)

st.title("🐄COWTRACK")
st.write("Upload a cow image to identify the disease")

# Load YOLO model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Image uploader
uploaded_file = st.file_uploader(
    "Upload cow image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to numpy array
    img_array = np.array(image)

    # Run inference
    results = model(img_array)

    # Extract prediction
    if len(results[0].boxes) > 0:
        cls_id = int(results[0].boxes.cls[0])
        confidence = float(results[0].boxes.conf[0])
        disease = model.names[cls_id]

        st.success(f"🩺 Disease Detected: **{disease}**")
        st.write(f"Confidence: **{confidence:.2f}**")
    else:
        st.warning("No disease detected in the image")
