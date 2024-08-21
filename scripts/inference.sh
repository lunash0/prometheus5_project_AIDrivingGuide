#!/bin/bash
# This bash file is for Standalone Execution of the model using main.py

echo "Inference on video"
python inference.py \
    --task_type all \
    --CFG_DIR configs/model.yaml \
    --OUTPUT_DIR test_video/kaggle_clip_all.mp4 \
    --video videos/kaggle_clip.mp4 \
    --ped_score_threshold 0.25 \
    --tl_score_threshold 0.4 &

python inference.py \
    --task_type message \
    --CFG_DIR configs/model.yaml \
    --OUTPUT_DIR test_video/kaggle_clip_message.mp4 \
    --video videos/kaggle_clip.mp4 \
    --ped_score_threshold 0.25 \
    --tl_score_threshold 0.4 

wait
echo "All jobs are done."