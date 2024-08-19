# Real-time Driving Guide AI with Object Detection and Instance Segmentation

This repository contains the codebase for the **AI Driving Guide project**, which provides driver guidance through *Object Detection* and *Instance Segmentation*. The project offers a user-friendly and customizable interface designed to detect and track pedestrians, traffic signs, and lanes in real-time video streams from various sources. For demonstration purposes, we use [Streamlit](https://streamlit.io/), a widely-used Python framework for developing interactive web applications.

## Overview
This README file provides a general introduction to the project. For detailed information about each detection task, please refer to the following README files linked :
- [Traffic Lights Detection README.md](https://github.com/lunash0/prometheus5_project_AIDrivingGuide/tree/feat/traffic_lights_detection/TrafficLights-Detection)
- [Pedestrian Detection README.md](https://github.com/lunash0/prometheus5_project_AIDrivingGuide/tree/feat/pedestrian_detection/Pedestrian-Detection)
- [Road Lane Detection README.md](https://github.com/lunash0/prometheus5_project_AIDrivingGuide/tree/feat/lane_detection/Lane-Detection)

<p align ="center">
  <img src="https://github.com/user-attachments/assets/f7b7a5a6-f9f1-429c-8fc4-1caa9da09e3c" alt="video_demo" width="500">
</p>


<center> 🛠 Tech Stack 🛠  
<br></br> 

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)

</center>




## Getting Started
### Requirements
```
pip install streamlit
```
### WebApp Demo on Streamlit 
**NOTE** : Before you run the following command, please modify the model path in `configs/model.yaml`. 
```
streamlit run app/app.py
```
### 🏠 Home page
![home_demo](./assets/home_demo.png)

### Select and Upload your souce ➡️ Click Process Button and Wait ! 
![image_bbox_demo](./assets/image_bbox_demo.png)
![image_bbox_input_demo](./assets/image_bbox_input_demo.png)
## Training
For fine-tuning the backbone, follow the bash file in `scripts/train.sh` :
```
cd ./models/Pedestrian_Detection/
python train.py \
    --mode train \
    --config_file configs/noHue_0.50.5_large_re_3.yaml \
    --OUTPUT_DIR ${OUTPUT_DIR}

cd ../Lane_Detection/
python train.py \
    --dataset ${DATASET_DIR}  \
    --pretrained ./model_weight/existing/best_model.pth

cd ../TrafficLights_Detection
python train.py  # Modify config.py for your configurations
```
You can also download each finetuned model url from here:
- [Pedestrian Detection](https://drive.google.com/file/d/10v2MYGYEH9h2a7KQS9HI1jGNG9QWBRC6/view?usp=drive_link)
- [TrafficLights Detection](https://drive.google.com/file/d/1yA3YCBp68J29G6osIzDu17igv3G3M2pM/view?usp=drive_link)
- [Lane Detection](https://drive.google.com/file/d/1ahltlZjJl-hdBRxf58jfbwFygS6bqIFB/view?usp=drive_link)

## Results
#### Total View
<div align="center">
  <img src="https://github.com/user-attachments/assets/1c7b86a3-597e-4048-85eb-6937bc1e0ced" alt="combined_video" style="width: 100%;">
</div>

#### Left : Comments Only | Right : Total View
<div align="center">
  <img src="https://github.com/user-attachments/assets/7a59aa64-ab7c-4b67-89f7-8eaa8149043a" alt="output_combined" style="width: 100%;">
</div>


## Directory Structure

```
prometheus5_project_AIDrivingGuide/
│
├── README.md         
├── play.py           
├── __init__.py      
│
├── engine/           
│   ├── models.py    
│   ├── utils.py       
│   └── __init__.py   
│
├── models/           
│   ├── TrafficLights_Detection/
│   ├── Pedestrian_Detection/
│   └── Lane_Detection/
│
├── scripts/           
│   ├── train.sh   
│   └── inference.sh   
│
├── configs/           
│   └── model.yaml     
│
├── assets/           
│   ├── image_bbox_input_demo.png
│   ├── home_demo.png
│   └── ... 
│
└── app/              
    ├── app.py      
    ├── settings.py   
    └── assets/       
        ├── videos/
        └── images/
```
  <br><br><br><br><br><br>

# TO-DO
- [x] Merge Pedestrian-Detection
- [x] Merge Traffic-Lights-Detection
    - [x] Fix Detecting Color of Light Issue
- [x] Merge Lane-Detection
    - [x] Develop messeage printing algorithms (for non RGB cases, considering score using threshold, etc.)
- [x] Add Image Output version
    - [x] Fix image/video input coexistence issue
- [x] Connect Streamlit
- [] Improve infernce time
  - [] Change Lane detection (Merging) Algorithm
- [] Refactor (Hard coded, comments, path, stremlit statistics page)

