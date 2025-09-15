import streamlit as st
import cv2
from ultralytics import YOLO

# Set page configuration
st.set_page_config(
    page_title="Object Detection Web App",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("📊 Object Detection Web App")
st.markdown("YOLOv8 with Streamlit")

# Webcam live feed section
st.title("Webcam Live Feed")
run = st.checkbox('Camera On/Off')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

model = YOLO("yolov8n.pt")

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame)

    annotated_frame = results[0].plot()

    FRAME_WINDOW.image(annotated_frame)
else:
    st.write('Stopped')
