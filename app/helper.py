import streamlit as st
import cv2
from pathlib import Path
import settings
from main import main as all_main 

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


# def process_video(video, score_threshold, type='Driving Guide Comment'):
#     """
#     Process a video based on the selected task and file path.

#     Args:
#     - file_path (Path): Path to the task-specific file.
#     - video (UploadedFile): Video to be processed.
#     - conf (float): Confidence threshold for processing.
#     - type : 'Driving Guide Comment' or 'Bounding Box'
#     Returns:
#     None
#     """
#     st.write("Processing video...")
#     # Example: Save the uploaded video
#     with open(settings.UPLOADED_VIDEO_PATH, 'wb') as out_file:
#         out_file.write(video.read())

#     # Example processing logic
#     if type == "Bounding Box":
#         st.write("Processing video with Bounding Box model...")

#     elif type == "Driving Guid Comment":
#         st.write("Processing video with Driving Guide Comment model...")
#         print(f'[info] uploaded video path : {settings.UPLOADED_VIDEO_PATH}')
#         print(f'[info] output video path : {settings.OUTPUT_VIDEO_PATH}')
#         all_main(settings.CFG_DIR, settings.UPLOADED_VIDEO_PATH, settings.OUTPUT_VIDEO_PATH, score_threshold)
#     else:
#         st.error("Invalid task selected")

#     # Example: Save the processed video
#     # Assuming the processed video is saved at settings.OUTPUT_VIDEO_PATH
#     st.write(f"Video saved at: {settings.OUTPUT_VIDEO_PATH}")


def process_video(video, score_threshold, type='Driving Guide Comment'):
    st.write("Processing video...")
    with open(settings.UPLOADED_VIDEO_PATH, 'wb') as out_file:
        out_file.write(video.read())

    # Check if the uploaded video was saved correctly
    if not settings.UPLOADED_VIDEO_PATH.exists():
        st.error(f"Uploaded video not found at: {settings.UPLOADED_VIDEO_PATH}")

    # Call the processing function (assuming all_main is defined elsewhere)
    all_main(settings.CFG_DIR, settings.OUTPUT_VIDEO_PATH, settings.UPLOADED_VIDEO_PATH, None)

    # Check if the processed video was saved
    if not settings.OUTPUT_VIDEO_PATH.exists():
        st.error(f"Processed video not found at: {settings.OUTPUT_VIDEO_PATH}")

    st.write(f"Video saved at: {settings.OUTPUT_VIDEO_PATH}")

    