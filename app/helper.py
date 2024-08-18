import streamlit as st
import cv2
from pathlib import Path
import settings
from play import main as all_main 
import moviepy.editor as mp 
import PIL 

def process_image(image, score_threshold, type="📝 Driving Guide Comment", output_path = settings.OUTPUT_IMAGE_PATH):
    with open(settings.UPLOADED_IMAGE_PATH, 'wb') as img_file:
        img_file.write(image.getbuffer())

    if type == "📦 Bounding Box":
        all_main('all', settings.CFG_DIR, output_path, None, settings.UPLOADED_IMAGE_PATH, score_threshold)
    elif type == "📝 Driving Guide Comment":
        all_main('message', settings.CFG_DIR, output_path, None, settings.UPLOADED_IMAGE_PATH, score_threshold)
    # elif type == "⚔️ Show All":
    #     all_main('all', settings.CFG_DIR, settings.OUTPUT_IMAGE_PATH, None, settings.UPLOADED_IMAGE_PATH, score_threshold)
    #     all_main('message', settings.CFG_DIR, settings.OUTPUT_IMAGE_PATH_2, None, settings.UPLOADED_IMAGE_PATH, score_threshold)
    else:
        st.error("⚠️ Invalid task selected")
        return

def process_video(video, score_threshold, type='📝 Driving Guide Comment', output_path=settings.OUTPUT_VIDEO_PATH_1):
    if type == "📦 Bounding Box":
        all_main('all', settings.CFG_DIR, output_path, settings.UPLOADED_VIDEO_PATH, None, score_threshold)
    elif type == "📝 Driving Guide Comment":
        all_main('message', settings.CFG_DIR, output_path, settings.UPLOADED_VIDEO_PATH, None, score_threshold)
    # elif type == "⚔️ Show All":
    #     process_video(video, score_threshold, type="📦 Bounding Box", output_path=settings.OUTPUT_VIDEO_PATH_1)
    #     process_video(video, score_threshold, type= "📝 Driving Guide Comment", output_path=settings.OUTPUT_VIDEO_PATH_2)

    if not output_path.exists():
        st.error(f"⚠️ Processed video not found at: {output_path}")
        return

    # Convert the processed video to MP4 format and overwrite the original file
    clip = mp.VideoFileClip(str(output_path))
    converted_output_path = output_path.with_suffix(".mp4")
    clip.write_videofile(str(converted_output_path), codec="libx264")

    if not converted_output_path.exists():
        st.error(f"⚠️ Converted video not found at: {converted_output_path}")
        return
    
    return converted_output_path  

