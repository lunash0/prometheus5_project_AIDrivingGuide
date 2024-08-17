# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="AI Driving Guide Simulation",
    page_icon="ð",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("AI Driving Guide Simulation")

# Sidebar
st.sidebar.header("Task Config")

# Task Options
task_type = st.sidebar.radio(
    "Select Task", ['Bounding Box', 'Driving Guide Comment'])

# Confidence level
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# File paths based on task selection
if task_type == 'Bounding Box':
    file_path = Path(settings.BOUNDING_BOX_FILE)
elif task_type == 'Driving Guide Comment':
    file_path = Path(settings.DRIVING_GUIDE_FILE)

# Sidebar
st.sidebar.header("Source Config")
source_radio = st.sidebar.radio(
    "Select Source", [settings.IMAGE, settings.VIDEO])

source_file = None

# If image is selected
if source_radio == settings.IMAGE:
    source_file = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_file is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_file)
                st.image(source_file, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_file is None:
            default_output_image_path = str(settings.DEFAULT_OUTPUT_IMAGE)
            st.image(default_output_image_path, caption='Output Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Process Image'):
                helper.process_image(file_path, uploaded_image, confidence)
                processed_image = PIL.Image.open(settings.OUTPUT_IMAGE_PATH)
                st.image(processed_image, caption='Processed Image',
                         use_column_width=True)

# If video is selected
elif source_radio == settings.VIDEO:
    source_file = st.sidebar.file_uploader(
        "Choose a video...", type=("mp4", "avi", "mov"))

    if st.sidebar.button('Process Video'):
        if source_file is not None:
            helper.process_video(file_path, source_file, confidence)
            st.video(settings.OUTPUT_VIDEO_PATH)
        else:
            st.error("Please upload a video file.")

else:
    st.error("Please select a valid source type!")