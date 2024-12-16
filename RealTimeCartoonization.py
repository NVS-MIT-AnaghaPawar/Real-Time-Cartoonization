import cv2
import streamlit as st
import numpy as np

# Streamlit app title and description
st.title("Real-Time Cartoon Filter")
st.write("A cartoon filter that emulates Snapchat's style. Click 'Start Video' to begin and 'Stop Video' to end.")

# Define Start and Stop buttons
start_button = st.button("Start Video")
stop_button = st.button("Stop Video")

# Initialize session state variables
if "run" not in st.session_state:
    st.session_state.run = False

# Update session state based on button clicks
if start_button:
    st.session_state.run = True
if stop_button:
    st.session_state.run = False

# Cartoonization parameters
BILATERAL_FILTER_VALUE = 5  # Reduced for better speed
COLOR_QUANTIZATION_LEVEL = 8  # Reduced for faster processing

def apply_bilateral_filter(frame):
    """Smooths the image while preserving edges using bilateral filtering."""
    return cv2.bilateralFilter(frame, BILATERAL_FILTER_VALUE, 75, 75)

def color_quantization(frame, k=COLOR_QUANTIZATION_LEVEL):
    """Applies color quantization to reduce the color palette of the image."""
    data = np.float32(frame).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    quantized = palette[labels.flatten()].reshape(frame.shape)
    return quantized.astype(np.uint8)

def detect_edges_stylized(gray_frame):
    """Detects edges using a stylized filter approach."""
    # Using adaptive thresholding for more pronounced, clean edges
    edges = cv2.adaptiveThreshold(
        gray_frame, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )
    return edges

def cartoonize_frame(frame):
    """Main cartoonization pipeline."""
    # Resize for faster processing
    small_frame = cv2.resize(frame, (320, 240))  # Adjust resolution for better performance

    # Apply bilateral filter to smooth the image
    filtered = apply_bilateral_filter(small_frame)

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    edges = detect_edges_stylized(gray)

    # Apply color quantization
    quantized = color_quantization(filtered)

    # Convert edges to 3-channel format for blending
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend the quantized image with the edge mask
    cartoon = cv2.bitwise_and(quantized, edges_colored)

    # Resize back to the original frame size
    cartoon = cv2.resize(cartoon, (frame.shape[1], frame.shape[0]))
    return cartoon

# Open video capture if Start button is clicked
cap = cv2.VideoCapture(0)

if st.session_state.run:
    stframe = st.empty()  # Placeholder for video frames
    
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam.")
            break

        # Apply cartoon effect
        cartoon_frame = cartoonize_frame(frame)

        # Display the cartoonized video feed
        stframe.image(cartoon_frame, channels="BGR")

# Release video capture when Stop button is clicked
if not st.session_state.run and cap.isOpened():
    cap.release()
    st.write("Video stopped.")