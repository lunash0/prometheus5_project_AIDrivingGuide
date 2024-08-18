# AI Driving Guide project

This repository contains the codebase for the **AI Driving Guide project**, which provides driver guidance through *Object Detection* and *Instance Segmentation*. The project offers a user-friendly and customizable interface designed to detect and track pedestrians, traffic signs, and lanes in real-time video streams from various sources. For demonstration purposes, we use [Streamlit](https://streamlit.io/), a widely-used Python framework for developing interactive web applications.

## Overview
This README file provides a general introduction to the project. For detailed information about each detection task, please refer to the following README files linked :
- [Traffic Lights Detection README.md](./TrafficLights-Detection/README.md)
- [Pedestrian Detection README.md](./Pedestrian-Detection/README.md)
- [Road Lane Detection README.md](./Lane-Detection/README.md)
- [Road Sign Detection README.md](./RoadSign-Detection/README.md)

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
### Usage
```
streamlit run app/app.py
```
![home_demo](./assets/home_demo.png)
![image_bbox_demo](./assets/image_bbox_demo.png)
![image_bbox_input_demo](./assets/image_bbox_input_demo.png)

## 2. Detection Tasks
### 2.1. Traffic Lights Detection
Using AI hub's traffic lights dataset, we trained the retinanet_resnet50_fpn_v2 model provided by torchvision. Not only can it distinguish red/green/yellow lights, but it can also distinguish information about left turns and right turns.

<br>

### 2.2. Pedestrian Detection

<br>

### 2.3. Road Lane Detection

<br>

  <br><br><br>


## Inference Results
 여기에 인퍼런스 영상 원본 캡쳐화면이랑 거기에 4가지 모델 다 적용시킨 아웃풋 영상 캡쳐화면 넣어서 간단하게 보여주기


## Acknowledgements


## Directory Structure

```
blank
```
  <br><br><br><br><br><br>







> **About External Resources**   
> 
> 프로젝트에 포함된 외부 코드나 리소스 정보(각각의 출처 및 배포 라이선스)


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
