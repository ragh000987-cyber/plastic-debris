import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2

from detector import detect_plastic
from drift import predict_drift
from priority import calculate_priority
from map_utils import make_map

st.set_page_config(page_title="Marine Plastic AI", page_icon="🌊", layout="wide")

st.title("🌊 AI/ML-Based Detection and Mapping of Floating Marine Plastic Debris")
st.caption("Hackathon prototype: satellite image → detection → drift prediction → priority map")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("YOLO model path", "weights/best.pt")
    current_speed = st.slider("Ocean current speed (m/s)", 0.0, 2.0, 0.5, 0.05)
    current_direction = st.slider("Current direction (degrees)", 0, 359, 45)
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 20.0, 5.0, 0.5)
    wind_direction = st.slider("Wind direction (degrees)", 0, 359, 45)
    hours = st.slider("Prediction time (hours)", 1, 168, 24)

uploaded = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    st.subheader("1. Satellite image")
    st.image(image, use_container_width=True)

    with st.spinner("Running AI detection..."):
        result = detect_plastic(image_np, model_path)

    st.subheader("2. AI detection")

    if result["mode"] == "trained_model":
        st.success("Custom plastic-detection model used.")
    else:
        st.warning(
            "No trained plastic model was found. Demo detections are shown. "
            "Train a custom YOLO model and place best.pt in weights/ for real detection."
        )

    if result["detections"]:
        df = pd.DataFrame(result["detections"])
        st.dataframe(df, use_container_width=True)

        annotated = result["annotated_image"]
        st.image(annotated, caption="Detected debris", use_container_width=True)

        first = result["detections"][0]
        lat = first["latitude"]
        lon = first["longitude"]
        confidence = first["confidence"]
        area = first["area_percent"]

        drift = predict_drift(
            lat, lon, current_speed, current_direction,
            wind_speed, wind_direction, hours
        )

        priority = calculate_priority(
            confidence=confidence,
            area_percent=area,
            drift_distance_km=drift["distance_km"]
        )

        st.subheader("3. Drift prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current location", f"{lat:.5f}, {lon:.5f}")
        c2.metric("Predicted location", f'{drift["latitude"]:.5f}, {drift["longitude"]:.5f}')
        c3.metric("Predicted drift", f'{drift["distance_km"]:.2f} km')

        st.write(
            f'Estimated movement after **{hours} hours**: '
            f'**{drift["distance_km"]:.2f} km**.'
        )

        st.subheader("4. Cleanup priority")
        st.metric("Priority score", f'{priority["score"]:.1f}/100')
        st.info(priority["level"])

        st.subheader("5. Interactive GIS map")
        map_html = make_map(
            lat, lon,
            drift["latitude"], drift["longitude"],
            confidence, priority["score"]
        )
        st.components.v1.html(map_html, height=600, scrolling=False)

        st.subheader("6. System output")
        st.write(
            "The prototype combines satellite-based detection, geographic "
            "mapping, ocean/wind drift estimation and a priority score to "
            "support field verification and targeted cleanup."
        )
    else:
        st.error("No potential debris was detected.")

else:
    st.info("Upload a satellite image to start the pipeline.")

st.divider()
st.caption("Important: this is a hackathon prototype. Real deployment requires validated satellite labels, trained weights, calibrated oceanographic data and field verification.")
