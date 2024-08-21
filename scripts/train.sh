#!/bin/bash 
# 📍NOTICE: You can find detailed explanation to finetune each model through corresponding README.md.

echo "Starting Pedestrian Detection model training..."
cd ./models/Pedestrian_Detection/
python train.py \
    --mode train \
    --config_file configs/${MODEL_YAML_FILE} \
    --OUTPUT_DIR ${OUTPUT_DIR}

echo "Starting Lane Detection model training..."
cd ../Lane_Detection/
python train.py \
    --dataset ${DATASET_DIR}  \
    --pretrained ./model_weight/existing/best_model.pth

echo "Starting Traffic Lights Detection model training..."
cd ../TrafficLights_Detection
python train.py  # Modify config.py for your configurations

echo "All training processes have been started." # Modify the directories of each model's paths
