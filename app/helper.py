import streamlit as st
import cv2
from pathlib import Path
import settings


def process_image(file_path, image, conf):
    """
    Process an image based on the selected task and file path.

    Args:
    - file_path (Path): Path to the task-specific file.
    - image (PIL Image): Image to be processed.
    - conf (float): Confidence threshold for processing.

    Returns:
    None
    """
    # Example processing logic based on task type
    if "Bounding Box" in str(file_path):
        # Add your bounding box detection logic here
        st.write("Processing with Bounding Box model...")
        # Assume output is saved at settings.OUTPUT_IMAGE_PATH
    elif "Driving Guide Comment" in str(file_path):
        # Add your driving guide comment logic here
        st.write("Processing with Driving Guide Comment model...")
        # Assume output is saved at settings.OUTPUT_IMAGE_PATH
    else:
        st.error("Invalid task selected")


def process_video(file_path, video, conf):
    """
    Process a video based on the selected task and file path.

    Args:
    - file_path (Path): Path to the task-specific file.
    - video (UploadedFile): Video to be processed.
    - conf (float): Confidence threshold for processing.

    Returns:
    None
    """
    st.write("Processing video...")
    # Example: Save the uploaded video
    with open(settings.UPLOADED_VIDEO_PATH, 'wb') as out_file:
        out_file.write(video.read())

    # Example processing logic
    if "Bounding Box" in str(file_path):
        # Add your bounding box detection logic here
        st.write("Processing video with Bounding Box model...")
    elif "Driving Guide Comment" in str(file_path):
        # Add your driving guide comment logic here
        st.write("Processing video with Driving Guide Comment model...")
    else:
        st.error("Invalid task selected")

    # Example: Save the processed video
    # Assuming the processed video is saved at settings.OUTPUT_VIDEO_PATH
    st.write(f"Video saved at: {settings.OUTPUT_VIDEO_PATH}")