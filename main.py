import torch
import cv2
import numpy as np
from engine.models import *
from engine.utils import load_yaml, default_argument_parser 
from tqdm import tqdm 
from Pedestrian_Detection.test import detect_ped_frame as detect_ped_frame
from TrafficLights_Detection.tf_inference import detect_tl_frame as detect_tl_frame

def draw_boxes_on_frame(frame, ped_info, tl_info):
    ped_rects, ped_texts, ped_warning_texts = ped_info 
    tl_rectangles, tl_texts = tl_info 
    for rect, text in zip(ped_rects, ped_texts):
        """
        rect[0] = [(x1, y1), (x2, y2)] 
        rect[1] = color
        text[0] = f"{label_text} Score: {score:.2f} Dist: {distance}px
        """
        cv2.rectangle(frame, rect[0][0], rect[0][1], rect[1], 2)
        cv2.putText(frame, text[0], text[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, text[2], 2)
    for warning_text in ped_warning_texts:
        cv2.putText(frame, warning_text[0], warning_text[1], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    for rect, text in zip(tl_rectangles, tl_texts):
        """
        rect[0] = [(x1, y1), (x2, y2)]
        rect[1] = color[::-1]
        """
        cv2.rectangle(frame, rect[0][0], rect[0][1], rect[1], 3)
        cv2.putText(frame, text[0], text[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, text[2], 2, lineType=cv2.LINE_AA)
    
    return frame 

def detect_video(pedestrian_model, traffic_light_model, input_path, output_path, score_thr, iou_thr, conf_thr, warning_dst, device):
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
            # Draw Pedestrian Detection on Frame
            ped_rects, ped_texts, ped_warning_texts = detect_ped_frame(pedestrian_model, img_frame, device, score_thr, iou_thr, conf_thr, warning_dst)

            # Draw Traffic Light Detection on Frame
            tl_rectangles, tl_texts = detect_tl_frame(traffic_light_model, img_frame, device, score_thr)

            processed_frame = draw_boxes_on_frame(img_frame, (ped_rects, ped_texts, ped_warning_texts), (tl_rectangles, tl_texts))

            video_writer.write(processed_frame)

            pbar.update(1)

    video_writer.release()
    cap.release()
    print(f'[INFO] Saved video to {output_path}')

def main(CFG_DIR, OUTPUT_DIR, video_path, image_path):
    cfg = load_yaml(CFG_DIR)
    device = torch.device(f'cuda:{cfg["device"]}' if torch.cuda.is_available() else 'cpu')

    print('[INFO] Loading models...')
    pedestrian_model = load_ped_model(cfg['pedestrian']['model_path'], cfg['pedestrian']['num_classes'], device)
    traffic_light_model = load_tl_model(cfg['traffic_light']['model_path'], cfg['traffic_light']['num_classes'], device)
    # lane_model = load_lane_model(cfg['lane']['model_path'], cfg['lane']['model_type'], device)
    print('[INFO] Model loaded successfully')

    if video_path is not None:
        print(f'[INFO] Processing video: {video_path}')
        detect_video(pedestrian_model, traffic_light_model, video_path, OUTPUT_DIR, cfg['score_threshold'], cfg['pedestrian']['iou_threshold'], cfg['pedestrian']['confidence_threshold'], cfg['pedestrian']['warning_distance'], device)
    if image_path is not None:
        print(f'[INFO] Processing image: {image_path}')

    print(f'[INFO] Save results to {OUTPUT_DIR}')

if __name__ == "__main__":
    args = default_argument_parser()
    video_path=None 
    image_path=None

    CFG_DIR = args.CFG_DIR
    OUTPUT_DIR = args.OUTPUT_DIR
    
    if args.video is not None:
        video_path = args.video 
    if args.image is not None:
        image_path = args.image 

    main(CFG_DIR, OUTPUT_DIR, video_path, image_path)

"""
# Example Usage
    # CFG_DIR = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml'
    # OUTPUT_DIR  = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/results/test.mp4'
    # video_path = '/home/yoojinoh/Others/PR/data/videos/test_video2.mp4'
    # image_path = None 

python main.py -c /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml \
    -o /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/results/test.mp4 -v /home/yoojinoh/Others/PR/data/videos/test_video1.mp4 
"""