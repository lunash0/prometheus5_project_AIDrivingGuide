import streamlit as st
from pathlib import Path
import settings
import helper
import PIL 

def page_home():
    st.title("🚗 AI Driving Guide Simulation")
    st.sidebar.header("🔧 Task Config")
    task_type = st.sidebar.radio(
        "Select Task", 
        ['📦 Bounding Box', '📝 Driving Guide Comment', '⚔️ Show All']
    )
    ped_score_threshold = float(st.sidebar.slider(
        "Pedestrian Model Score Threshold", 25, 100, 10, \
            help="This slider allows you to set the confidence threshold for the pedestrian model.")) / 100

    tl_score_threshold = float(st.sidebar.slider(
        "Traffic Lights Model Score Threshold", 25, 100, 10, \
            help="This slider allows you to set the confidence threshold for the traffic lights model.")) / 100
    
    file_path = Path(settings.TASK_FILE) 
        
    st.sidebar.header("🔍 Source Config")
    source_radio = st.sidebar.radio("Select Source", [settings.DEFAULT, settings.IMAGE, settings.VIDEO])

    source_file = None

    # 1) HOME
    if source_radio == settings.DEFAULT:
        st.subheader('Select Your Source to Start the Simulation !')
        st.markdown(
            """
            <style>
            .gif-container {
                display: flex;
                justify-content: center;
            }
            .gif-container img {
                width: 800px;
                height: auto; 
            }
            </style>
            <div class="gif-container">
                <img src="https://github.com/user-attachments/assets/819a1b4a-fb48-4f2a-abf6-78eb7cf4bc88">
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        st.markdown("### How to Use the Simulation 📋")
        st.write("""
        1. **Select Task**: Choose the task you want to simulate.
        2. **Set Score Threshold**: Adjust the confidence threshold for the model.
        3. **Choose Source**: Upload an image(.png) or video(.mp4) to run the simulation.
        4. **Process**: Click the process button to start the simulation.
        """)

        st.write("[🚩View on our poster](%s)" % settings.POSTER_URL)
        st.write("[🖥️Visit our Github Page](%s)" % settings.GITHUB)
        st.write("[🔥Check out Prometheus Page](%s)" % settings.PROMETHEUS_URL)

        st.markdown("### Need Help? 🆘")
        st.write("If you have any questions or need support, please reach out to us at [HERE](dianaoh1021@gmail.com).")

    # 2) Image 
    elif source_radio == settings.IMAGE:
        source_file = st.sidebar.file_uploader("📷 Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        if task_type == "⚔️ Show All":     
            if source_file is None:
                st.image(str(settings.DEFAULT_IMAGE), caption="Default Image", use_column_width=False,  width=600)
            else:
                st.image(source_file, caption="Uploaded Image", use_column_width=True)

            col1, col2 = st.columns(2)
            
            if st.sidebar.button('🔄 Process Image', key='process_image_button'):
                if source_file is None:
                    st.error("⚠️ Please upload an image file before processing.")
                with col1:
                    helper.process_image(source_file, ped_score_threshold, tl_score_threshold, "📝 Driving Guide Comment", settings.OUTPUT_IMAGE_PATH_1)
                    processed_image = PIL.Image.open(settings.OUTPUT_IMAGE_PATH_1)
                    st.image(processed_image, caption='Comments View', use_column_width=True) 
                with col2:
                    helper.process_image(source_file, ped_score_threshold, tl_score_threshold, "📦 Bounding Box", settings.OUTPUT_IMAGE_PATH_2)
                    processed_image = PIL.Image.open(settings.OUTPUT_IMAGE_PATH_2)
                    st.image(processed_image, caption='Total View', use_column_width=True) 

                st.caption(f'Selected Score threshold (Pedestrian, Traffic Lights): {ped_score_threshold, tl_score_threshold}')
                st.caption(f'📒 NOTICE : Re-run to apply updated threshold !')
            else:
                with col1:
                    if source_file is None:
                        st.image(str(settings.DEFAULT_OUTPUT_IMAGE_2), caption="Default Comments View", use_column_width=True)
                with col2:
                    if source_file is None:
                        st.image(str(settings.DEFAULT_OUTPUT_IMAGE_1), caption="Default Total Image", use_column_width=True)
                        
        elif task_type == "📝 Driving Guide Comment" or "📦 Bounding Box":
            col1, col2 = st.columns(2)
            with col1:
                if source_file is None:
                    st.image(str(settings.DEFAULT_IMAGE), caption="Default Image", use_column_width=True)
                else:
                    st.image(source_file, caption="Uploaded Image", use_column_width=True)
            with col2:
                if source_file is None:
                    if task_type == "📝 Driving Guide Comment":
                        default_path = settings.DEFAULT_OUTPUT_IMAGE_2
                    else:
                        default_path = settings.DEFAULT_OUTPUT_IMAGE_1  
                    st.image(str(default_path), caption="Default View", use_column_width=True)

                else:
                    if st.sidebar.button('🔄 Process Image', key="process_image_one_button"):
                        if source_file is None:
                            st.error("⚠️ Please upload an image file before processing.")
                        helper.process_image(source_file, ped_score_threshold, tl_score_threshold, task_type)
                        processed_image = PIL.Image.open(settings.OUTPUT_IMAGE_PATH_1)
                        st.image(processed_image, caption='Output View', use_column_width=True)    
            st.caption(f'Selected Score threshold (Pedestrian, Traffic Lights): {ped_score_threshold, tl_score_threshold}')
            st.caption(f'Re-run to apply updated threshold !')
        else:
            st.error(f"⚠️ Inavailable Selection")

    # 3) Video 
    elif source_radio == settings.VIDEO:
        source_file = st.sidebar.file_uploader("🎥 Choose a video...", type=("mp4", "avi", "mov"))

        if task_type == "⚔️ Show All":     
            col1, col2 = st.columns(2)
            if source_file is None:
                st.info("Select Your video to start inference.")
            if st.sidebar.button('🔄 Process Video', key='process_video_button'):
                if source_file is None:
                    st.error("⚠️ Please upload an image file before processing.")
                if source_file is not None:
                    # Save the uploaded video to a file
                    with open(settings.UPLOADED_VIDEO_PATH, 'wb') as out_file:
                        out_file.write(source_file.read())
                        
                    # Process the first video (Driving Guide Comment)
                    with col1:
                        video_path_1 = helper.process_video(source_file, ped_score_threshold, tl_score_threshold, "📝 Driving Guide Comment", settings.OUTPUT_VIDEO_PATH_1)
                        st.video(str(video_path_1))
                        st.caption("📝 Driving Guide Comment")
                    
                    # Process the second video (Bounding Box)
                    with col2:
                        video_path_2 = helper.process_video(source_file, ped_score_threshold, tl_score_threshold, "📦 Bounding Box", settings.OUTPUT_VIDEO_PATH_2)
                        st.video(str(video_path_2))
                        st.caption("📦 Bounding Box Comment")

                    st.caption(f'Selected Score threshold (Pedestrian, Traffic Lights): {ped_score_threshold, tl_score_threshold}')
                    st.caption(f'📒 NOTICE : Re-run to apply updated threshold !')
            
        elif task_type == "📝 Driving Guide Comment" or "📦 Bounding Box":
            col1, col2 = st.columns(2)
            if source_file is None:
                st.info("Select Your video to start inference.")
            else: 
                with open(settings.UPLOADED_VIDEO_PATH, 'wb') as out_file:
                    out_file.write(source_file.read()) #getbuffer
                with col1:
                    st.video(str(settings.UPLOADED_VIDEO_PATH), autoplay=True)
                    st.caption(f"Original Video")
                with col2:
                    if st.sidebar.button('🔄 Process Video', key="process_video_one_button"):
                        if source_file is None:
                            st.error("⚠️ Please upload an image file before processing.")
                        video_path = helper.process_video(source_file, ped_score_threshold, tl_score_threshold, task_type)
                        st.video(str(video_path))
                        st.caption(f"{task_type}")

                st.caption(f'Selected Score threshold (Pedestrian, Traffic Lights): {ped_score_threshold, tl_score_threshold}')
                st.caption(f'🔹Re-run to apply updated threshold !')
        else:
            st.error(f"⚠️ Inavailable Selection")