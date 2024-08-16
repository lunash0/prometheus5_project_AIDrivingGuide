import torch
import cv2
import numpy as np
from engine.models import *
from engine.utils import load_yaml, default_argument_parser 
from tqdm import tqdm 
from Pedestrian_Detection.ped_inference import detect_ped_frame as detect_ped_frame
from TrafficLights_Detection.tl_inference import detect_tl_frame as detect_tl_frame
from Lane_Detection.lane_inference import detect_lane_frame , load_lane_model

def draw_boxes_on_frame(frame, ped_info, tl_info, lane_info):
    ped_rects, ped_texts, ped_warning_texts = ped_info 
    tl_rectangles, tl_texts, tl_messages = tl_info 
    for rect, text in zip(ped_rects, ped_texts):
    #    cv2.rectangle(frame, rect[0][0], rect[0][1], rect[1], 2)
        cv2.putText(frame, text[0], text[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, text[2], 2)
    for warning_text in ped_warning_texts:
        cv2.putText(frame, warning_text[0], warning_text[1], cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

    for rect, text in zip(tl_rectangles, tl_texts):
    #    cv2.rectangle(frame, rect[0][0], rect[0][1], rect[1], 3)
        cv2.putText(frame, text[0], text[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, text[2], 2, lineType=cv2.LINE_AA)
    
    if len(tl_messages) > 0:
        message = max(tl_messages)
        if message == 'STOP':
            color = (0, 0, 255)
        elif message == 'WAIT':
            color = (0, 152, 255)
        elif message == 'GO':
            color = (0, 248, 211)
        cv2.putText(frame, message, (35, 85), cv2.FONT_HERSHEY_TRIPLEX, 3, color, 6, lineType=cv2.LINE_AA)
        
    cv2.drawContours(frame, lane_info[0], lane_info[1], lane_info[2], thickness=3) 

    return frame 

def detect_video(pedestrian_model, traffic_light_model, lane_model, input_path, output_path, score_thr, iou_thr, conf_thr, warning_dst, device):
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened() == False:
        print('Error while trying to read video. Please check path again')

    codec = cv2.VideoWriter_fourcc(*'XVID')
    video_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(output_path, codec, video_fps, video_size)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'[INFO] Total number of frames: {frame_cnt}')

    with tqdm(total=frame_cnt, desc="Processing Frames") as pbar:
        while True:
            hasFrame, img_frame = cap.read()
            if not hasFrame:
                print(f'Processed all frames')
                break

            ped_rects, ped_texts, ped_warning_texts = detect_ped_frame(pedestrian_model, img_frame, score_thr, iou_thr, conf_thr, warning_dst, device)
            ped_info = (ped_rects, ped_texts, ped_warning_texts)

            tl_rectangles, tl_texts, tl_messages = detect_tl_frame(traffic_light_model, img_frame, device, score_thr)
            tl_info = (tl_rectangles, tl_texts, tl_messages)

            lane_info = detect_lane_frame(lane_model, img_frame, device)
            processed_frame = draw_boxes_on_frame(img_frame, ped_info, tl_info, lane_info)

            video_writer.write(processed_frame)

            pbar.update(1)

    video_writer.release()
    cap.release()
    print(f'[INFO] Saved video to {output_path}')

def main(CFG_DIR, OUTPUT_DIR, video_path, image_path): 
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") 
    cfg = load_yaml(CFG_DIR)

    print('[INFO] Loading models...')
    pedestrian_model = load_ped_model(cfg['pedestrian']['model_path'], cfg['pedestrian']['num_classes'], device)
    traffic_light_model = load_tl_model(cfg['traffic_light']['model_path'], cfg['traffic_light']['num_classes'], device)
    lane_model = load_lane_model(cfg['lane']['model_path'], cfg['lane']['model_type'], device)
    print('[INFO] Model loaded successfully')

    if video_path is not None:
        print(f'[INFO] Processing video: {video_path}')
        detect_video(pedestrian_model, traffic_light_model, lane_model, video_path, OUTPUT_DIR, cfg['score_threshold'], cfg['pedestrian']['iou_threshold'], cfg['pedestrian']['confidence_threshold'], cfg['pedestrian']['warning_distance'], device)
    if image_path is not None:
        print(f'[INFO] Processing image: {image_path}')

if __name__ == "__main__":
    # args = default_argument_parser()
    video_path=None 
    image_path=None

    # CFG_DIR = args.CFG_DIR
    # OUTPUT_DIR = args.OUTPUT_DIR
    
    # if args.video is not None:
    #     video_path = args.video 
    # if args.image is not None:
    #     image_path = args.image 

    CFG_DIR = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml'
    OUTPUT_DIR = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/results/test_video2_output.mp4'
    video_path = '/home/yoojinoh/Others/PR/data/videos/test_video2.mp4'
    main(CFG_DIR, OUTPUT_DIR, video_path, image_path)

"""
# Example Usage
    # CFG_DIR = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml'
    # OUTPUT_DIR  = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/results/test.mp4'
    # video_path = '/home/yoojinoh/Others/PR/data/videos/test_video2.mp4'
    # image_path = None 

python main.py -c /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml \
    -o /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/results/test.mp4 -v /home/yoojinoh/Others/PR/data/videos/test_video2.mp4 

"""