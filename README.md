# AI Driving Guide project

<br>

> This README file provides general introduction about the project. For detailed information about each detection task, please refer to the README files below.  
>   
> - [Traffic Lights Detection README.md](./TrafficLights-Detection/README.md)
> - [Pedestrian Detection README.md](./Pedestrian-Detection/README.md)
> - [Road Lane Detection README.md](./Lane-Detection/README.md)
> - [Road Sign Detection README.md](./RoadSign-Detection/README.md)
<center> 🛠 Tech Stack 🛠

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)

</center>

<br>

-----
<p align ="center">
  <img src="https://github.com/user-attachments/assets/f7b7a5a6-f9f1-429c-8fc4-1caa9da09e3c" alt="video_demo" width="500">
</p>

<br><br>
## 1. Brief? Simple? Introduction <- 무슨 단어 쓰는게 더 자연스러운가 
- Goal
  - Provide driver guidance through vehicle driving environment object recognition  
  <br>
- Detailed description
  - Performs four types of detection: traffic lights, pedestrians, lanes, and traffic signs.  
  - Using detection models trained on road driving images, create two types of videos: video giving driving guide comments, video showing bounding boxes on detection objects
  - Use streamlit for demonstration. (Users can select the desired simulation video type and adjust model confidence directly.)

  <br><br><br>

## 2. Detection Tasks
### 2.1. Traffic Lights Detection
Using AI hub's traffic lights dataset, we trained the retinanet_resnet50_fpn_v2 model provided by torchvision. Not only can it distinguish red/green/yellow lights, but it can also distinguish information about left turns and right turns.
(내용 간단히만 더 보충하고 정리할 예정~,~)

<br>

### 2.2. Pedestrian Detection

<br>

### 2.3. Road Lane Detection

<br>

### 2.4. Road Sign Detection

  <br><br><br>

## 3. Getting Started

  <br><br><br>

## 4. Inference Results
 여기에 인퍼런스 영상 원본 캡쳐화면이랑 거기에 4가지 모델 다 적용시킨 아웃풋 영상 캡쳐화면 넣어서 간단하게 보여주기


## 5. Directory Structure
깃헙 레포 최종 정리되고나면 이부분 채워넣기

  <br><br><br><br><br><br>







> **About External Resources**   
> 
> 프로젝트에 포함된 외부 코드나 리소스 정보(각각의 출처 및 배포 라이선스)

> **Members? Contributers?** <- 어떤 단어가 좋을까용  
> 
> Seunghyeon Moon, 본인 이름들 추가해주세요 스펠링틀릴까봐...

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
  - [] Parallize
  - [] Change Lane detection (Merging) Algorithm
- [] Refactor (Hard coded, comments, path, stremlit statistics page)
