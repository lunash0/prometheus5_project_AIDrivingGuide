from pathlib import Path
import PIL
import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import settings
import helper 
import streamlit as st

st.set_page_config(
    page_title="AI Driving Guide Simulation",
    page_icon="ð",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("AI Driving Guide Simulation")
st.sidebar.header("Task Config")

task_type = st.sidebar.radio(
    "Select Task", ['Bounding Box', 'Driving Guide Comment'])

# Confidence level
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 1, 100, 15)) / 100

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
            print(f'source_file: {source_file}')
            helper.process_video(source_file, confidence)
            video_path = settings.OUTPUT_VIDEO_PATH.as_posix()  # Convert Path to string
            st.video(video_path)
            # st.video(settings.OUTPUT_VIDEO_PATH)
        else:
            st.error("Please upload a video file.")

else:
    st.error("Please select a valid source type!")

#TODO(Yoojin) Solve Error
""" 
[INFO] Saved video to /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/app/videos/processed_video.mp4
2024-08-17 14:15:10.975 Uncaught app exception
Traceback (most recent call last):
  File "/home/yoojinoh/.miniconda3/envs/deeplearning/lib/python3.8/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 85, in exec_func_with_error_handling
    result = func()
  File "/home/yoojinoh/.miniconda3/envs/deeplearning/lib/python3.8/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 576, in code_to_exec
    exec(code, module.__dict__)
  File "/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/app/app.py", line 83, in <module>
    st.video(settings.OUTPUT_VIDEO_PATH)
  File "/home/yoojinoh/.miniconda3/envs/deeplearning/lib/python3.8/site-packages/streamlit/runtime/metrics_util.py", line 408, in wrapped_func
    result = non_optional_func(*args, **kwargs)
  File "/home/yoojinoh/.miniconda3/envs/deeplearning/lib/python3.8/site-packages/streamlit/elements/media.py", line 341, in video
    marshall_video(
  File "/home/yoojinoh/.miniconda3/envs/deeplearning/lib/python3.8/site-packages/streamlit/elements/media.py", line 531, in marshall_video
    _marshall_av_media(coordinates, proto, data, mimetype)
  File "/home/yoojinoh/.miniconda3/envs/deeplearning/lib/python3.8/site-packages/streamlit/elements/media.py", line 435, in _marshall_av_media
    raise RuntimeError("Invalid binary data format: %s" % type(data))
RuntimeError: Invalid binary data format: <class 'pathlib.PosixPath'>
"""