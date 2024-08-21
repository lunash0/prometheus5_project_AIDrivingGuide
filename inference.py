"""
main.py

This script is intended for Standalone Execution.
It runs the model directly without requiring a web interface.

"""
import torch
import cv2
from engine.utils import load_yaml, default_argument_parser, setup 
from tqdm import tqdm 
from models.Pedestrian_Detection.ped_inference import detect_ped_frame, load_ped_model
from models.TrafficLights_Detection.tl_inference import detect_tl_frame, message_rule, load_tl_model
from models.Lane_Detection.lane_inference import detect_lane_frame , load_lane_model
from pathlib import Path
import os 

def draw_boxes_on_frame(frame, ped_info, tl_info, lane_info, task_type='all', debug=False):
    """
    type = ['all', 'message']
        - all : bounding boxes and comments
        - message : comments only 
    """
    ped_rects, ped_texts, ped_warning_texts = ped_info 
    tl_rectangles, tl_texts, tl_messages, prev_tl_messages = tl_info 
    
    # 1) Pedestrian 
    if task_type=='all':
        for rect, text in zip(ped_rects, ped_texts):
            cv2.rectangle(frame, rect[0][0], rect[0][1], rect[1], 2)
            cv2.putText(frame, text[0], text[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, text[2], 2)
        
    for warning_text in ped_warning_texts:
        cv2.putText(frame, warning_text[0], warning_text[1], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # 2) Traffic Lights 
    if task_type=="all":
        for rect, text in zip(tl_rectangles, tl_texts):
            cv2.rectangle(frame, rect[0][0], rect[0][1], rect[1], 3)
            cv2.putText(frame, text[0], text[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, text[2], 2, lineType=cv2.FILLED)
    
    # sanity check
    if debug is True:
        print(f'[DEBUG] tl_messages = {tl_messages}')
    
    if len(tl_messages) > 1:
        message, color, prev_tl_messages = message_rule(tl_messages, prev_tl_messages)
        prev_tl_messages.append(message)

        cv2.putText(frame, message, (35, 85), cv2.FONT_HERSHEY_TRIPLEX, 3, color, 4, lineType=cv2.LINE_AA)

    elif len(tl_messages) == 1:
        message, color, prev_tl_messages = message_rule(tl_messages, prev_tl_messages)
        if message == 'STOP' or message == 'PREPARE TO STOP':
            prev_tl_messages.append(message)
            cv2.putText(frame, message, (35, 85), cv2.FONT_HERSHEY_TRIPLEX, 3, color, 4, lineType=cv2.LINE_AA)
    
    # 3) Lane
    cv2.drawContours(frame, lane_info[0], lane_info[1], (0, 255, 255), thickness=3) 

    return frame, prev_tl_messages

def detect_image(task_type, pedestrian_model, traffic_light_model, lane_model, 
                 input_path, output_path, \
                 ped_score_thr, tl_score_thr, \
                 iou_thr, conf_thr, warning_dst, device, \
                 lane_merge_thr=50, resize_width=1280, resize_height=720):
    
    img_frame = cv2.imread(input_path)
    img_frame = cv2.resize(img_frame, (resize_width, resize_height))

    prev_tl_messages = ['NONE', 'NONE']  

    ped_rects, ped_texts, ped_warning_texts = detect_ped_frame(pedestrian_model, img_frame, ped_score_thr, iou_thr, conf_thr, warning_dst, device)
    ped_info = (ped_rects, ped_texts, ped_warning_texts)

    tl_rectangles, tl_texts, tl_messages = detect_tl_frame(traffic_light_model, img_frame, device, tl_score_thr)
    tl_info = (tl_rectangles, tl_texts, tl_messages, prev_tl_messages)

    lane_info = detect_lane_frame(lane_model, img_frame, device, threshold=lane_merge_thr)
    processed_frame, prev_tl_messages = draw_boxes_on_frame(img_frame, ped_info, tl_info, lane_info, task_type)

    cv2.imwrite(output_path, processed_frame)
    print(f'[INFO] Saved image to {output_path}')

def detect_video(task_type, pedestrian_model, traffic_light_model, lane_model, 
                 input_path, output_path, \
                 ped_score_thr, tl_score_thr, \
                 iou_thr, conf_thr, warning_dst, device, 
                 lane_merge_thr=50, resize_width=1280, resize_height=720):
    """
    Save video as avi file
    """
    cap = cv2.VideoCapture(input_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) 

    codec = cv2.VideoWriter_fourcc(*'XVID')
    avi_output_path = output_path.with_suffix(".avi") 
    video_writer = cv2.VideoWriter(str(avi_output_path), codec, video_fps, (resize_width, resize_height))
    
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_tl_messages = ['NONE', 'NONE'] 

    print(f'[INFO] Total number of frames: {frame_cnt}')

    with tqdm(total=frame_cnt, desc="[INFO] Processing Frames") as pbar:
        while True:
            hasFrame, img_frame = cap.read()
            if not hasFrame:
                print(f'Processed all frames')
                break
            img_frame = cv2.resize(img_frame, (resize_width, resize_height))
            
            ped_rects, ped_texts, ped_warning_texts = detect_ped_frame(pedestrian_model, img_frame, ped_score_thr, iou_thr, conf_thr, warning_dst, device)
            ped_info = (ped_rects, ped_texts, ped_warning_texts)

            tl_rectangles, tl_texts, tl_messages  = detect_tl_frame(traffic_light_model, img_frame, device, tl_score_thr)
            tl_info = (tl_rectangles, tl_texts, tl_messages, prev_tl_messages)

            lane_info = detect_lane_frame(lane_model, img_frame, device, threshold=lane_merge_thr)
            processed_frame, prev_tl_messages = draw_boxes_on_frame(img_frame, ped_info, tl_info, lane_info, task_type)

            video_writer.write(processed_frame) 
            pbar.update(1)

    video_writer.release()
    cap.release()
    print(f'[INFO] Saved video to {avi_output_path}')

def detect_video2mp4(task_type, pedestrian_model, traffic_light_model, lane_model, 
                     input_path, output_path, 
                     ped_score_thr, tl_score_thr, \
                     iou_thr, conf_thr, warning_dst, device, 
                     lane_merge_thr=50, resize_width=1280, resize_height=720):
    """
    Saves detected video as mp4 file
    """
    cap = cv2.VideoCapture(input_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) 

    codec = cv2.VideoWriter_fourcc(*'mp4v') 
    mp4_output_path = str(Path(output_path).with_suffix(".mp4"))  
    video_writer = cv2.VideoWriter(mp4_output_path, codec, video_fps, (resize_width, resize_height))
    
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_tl_messages = ['NONE', 'NONE'] 

    print(f'[INFO] Total number of frames: {frame_cnt}')

    with tqdm(total=frame_cnt, desc="[INFO] Processing Frames") as pbar:
        while True:
            hasFrame, img_frame = cap.read()
            if not hasFrame:
                print(f'Processed all frames')
                break
            img_frame = cv2.resize(img_frame, (resize_width, resize_height))
            
            ped_rects, ped_texts, ped_warning_texts = detect_ped_frame(pedestrian_model, img_frame, ped_score_thr, iou_thr, conf_thr, warning_dst, device)
            ped_info = (ped_rects, ped_texts, ped_warning_texts)

            tl_rectangles, tl_texts, tl_messages  = detect_tl_frame(traffic_light_model, img_frame, device, tl_score_thr)
            tl_info = (tl_rectangles, tl_texts, tl_messages, prev_tl_messages)

            lane_info = detect_lane_frame(lane_model, img_frame, device, threshold=lane_merge_thr)
            processed_frame, prev_tl_messages = draw_boxes_on_frame(img_frame, ped_info, tl_info, lane_info, task_type)

            video_writer.write(processed_frame) 
            pbar.update(1)

    video_writer.release()
    cap.release()
    print(f'[INFO] Saved video to {mp4_output_path}')

def main(task_type, CFG_DIR, OUTPUT_DIR, video_path, image_path, ped_score_threshold, tf_score_threshold): 
    check_input_path = video_path if video_path is not None else image_path

    if not os.path.exists(check_input_path):
        print(f'[ERROR] Input file not found at: {check_input_path}. Exiting...')
        exit(1)

    cfg = load_yaml(CFG_DIR)
    device_num = cfg['device']
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    print(f'[INFO] Device is on {device_num}')

    print('[INFO] Loading models...')
    pedestrian_model = load_ped_model(cfg['pedestrian']['model_path'], cfg['pedestrian']['num_classes'], device)
    traffic_light_model = load_tl_model(cfg['traffic_light']['model_path'], cfg['traffic_light']['num_classes'], device)
    lane_model = load_lane_model(cfg['lane']['model_path'], cfg['lane']['model_type'], device)
    print('[INFO] Models loaded successfully')

    print(f'[INFO] Task type : {task_type} | Pedestrian Score threshold : {ped_score_threshold} | Traffic Lights Score threshold : {tf_score_threshold}')

    if video_path is not None:
        print(f'[INFO] Processing video: {video_path}')
        detect_video2mp4(task_type, pedestrian_model, traffic_light_model, lane_model, 
                         video_path, OUTPUT_DIR, 
                         ped_score_threshold, tf_score_threshold,
                         cfg['pedestrian']['iou_threshold'], 
                         cfg['pedestrian']['confidence_threshold'], 
                         cfg['pedestrian']['warning_distance'], 
                         device,
                         lane_merge_thr=cfg['lane']['lane_merge_thr'])
    
    if image_path is not None:
        print(f'[INFO] Processing image: {image_path}')        
        detect_video2mp4(task_type, pedestrian_model, traffic_light_model, lane_model, 
                         image_path, 
                         OUTPUT_DIR, 
                         ped_score_threshold, tf_score_threshold,
                         cfg['pedestrian']['iou_threshold'], 
                         cfg['pedestrian']['confidence_threshold'], 
                         cfg['pedestrian']['warning_distance'], device)

if __name__ == "__main__":
    args = default_argument_parser()
    task_type, cfg_dir, output_dir, video_path, image_path, ped_score_threshold, tf_score_threshold = setup(args)
    main(task_type, cfg_dir, output_dir, video_path, image_path, ped_score_threshold, tf_score_threshold)