import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Set page configuration
st.set_page_config(
    page_title="Object Detection Web App",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("📊 Object Detection Web App")
st.markdown("YOLOv8 with Streamlit")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Video transformer class for object detection
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO detection
        results = self.model(img)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        return annotated_frame

# WebRTC configuration for better connectivity
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Webcam live feed section
st.title("Live Object Detection")
st.markdown("Click 'START' to begin using your camera for real-time object detection.")

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    video_transformer_factory=ObjectDetectionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Instructions
st.markdown("""
### Instructions:
1. Click the **START** button above
2. Allow camera permissions when prompted by your browser
3. The app will show real-time object detection on your camera feed
4. Click **STOP** to end the session

**Note:** This app requires camera permissions to work properly.
""")
