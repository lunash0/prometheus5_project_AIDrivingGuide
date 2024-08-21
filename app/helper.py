import streamlit as st
import settings
import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from play import main as all_main 
import subprocess

def process_image(image, ped_score_threshold, tl_score_threshold, type="📝 Driving Guide Comment", output_path = settings.OUTPUT_IMAGE_PATH_1):
    with open(settings.UPLOADED_IMAGE_PATH, 'wb') as img_file:
        img_file.write(image.getbuffer())

    if type == "📦 Bounding Box":
        all_main('all', settings.CFG_DIR, output_path, None, settings.UPLOADED_IMAGE_PATH, ped_score_threshold, tl_score_threshold)
    elif type == "📝 Driving Guide Comment":
        all_main('message', settings.CFG_DIR, output_path, None, settings.UPLOADED_IMAGE_PATH, ped_score_threshold, tl_score_threshold)
    else:
        st.error("⚠️ Invalid task selected")
        return

def process_video(video, ped_score_threshold, tl_score_threshold, type='📝 Driving Guide Comment', output_path=settings.OUTPUT_VIDEO_PATH_1):
    avi_output_path = output_path.with_suffix('.avi')

    if type == "📦 Bounding Box":
        all_main('all', settings.CFG_DIR, avi_output_path, settings.UPLOADED_VIDEO_PATH, None, ped_score_threshold, tl_score_threshold)
    elif type == "📝 Driving Guide Comment":
        all_main('message', settings.CFG_DIR, avi_output_path, settings.UPLOADED_VIDEO_PATH, None, ped_score_threshold, tl_score_threshold)

    if not avi_output_path.exists():
        st.error(f"⚠️ Processed AVI video not found at: {avi_output_path}")
        return
    
    convert_avi_to_mp4(avi_output_path, output_path)

    # Return the path to the processed video, now in MP4 format
    return output_path

def convert_avi_to_mp4(avi_path, mp4_path):
    command = f"ffmpeg -y -i {avi_path} -vcodec libx264 {mp4_path}"
    subprocess.run(command, shell=True)

    

