#!/bin/bash

cd "$(dirname "$0")" || { echo "Faild to move directory"; exit 1; }

cd .. || { echo "Faild to move to high directory"; exit 1; }

# Inference video (Only bounding boxes)
python no_message_all.py \
    --CFG_DIR /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml \
    --OUTPUT_DIR /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/results/test_video2_outputs_new.mp4 \
    --video_path /home/yoojinoh/Others/PR/data/videos/test_video2.mp4 &

# Inference image 
python no_message_all.py \
    --CFG_DIR /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml \
    --OUTPUT_DIR /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/results/test_image1_output.jpg \
    --image_path /home/yoojinoh/Others/PR/data/videos/test_image1.jpg &

# Inference video (REAL)
python message_video.py \
    --CFG_DIR /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml \
    --OUTPUT_DIR /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/results/test_video2_outputs_new.mp4 \
    --video_path /home/yoojinoh/Others/PR/data/videos/test_video2.mp4 &
wait

echo "All jobs are done."