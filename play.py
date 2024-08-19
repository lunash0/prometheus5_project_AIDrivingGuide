import torch
import cv2
from engine.utils import load_yaml, default_argument_parser 
from tqdm import tqdm 
from models.Pedestrian_Detection.ped_inference import detect_ped_frame, load_ped_model
from models.TrafficLights_Detection.tl_inference import detect_tl_frame, message_rule, load_tl_model
from models.Lane_Detection.lane_inference import detect_lane_frame , load_lane_model
from pathlib import Path

def draw_boxes_on_frame(frame, ped_info, tl_info, lane_info, task_type='all'):
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
        cv2.putText(frame, warning_text[0], warning_text[1], cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    
    # 2) Traffic Lights 
    if task_type=="all":
        for rect, text in zip(tl_rectangles, tl_texts):
            cv2.rectangle(frame, rect[0][0], rect[0][1], rect[1], 3)
            cv2.putText(frame, text[0], text[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, text[2], 2, lineType=cv2.LINE_AA)
    
    # sanity check
    # print(f'[DEBUG] tl_messages = {tl_messages}')
    
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

def detect_image(task_type, pedestrian_model, traffic_light_model, lane_model, input_path, output_path, \
                 score_thr, iou_thr, conf_thr, warning_dst, device, \
                 resize_width=1280, resize_height=720):
    
    img_frame = cv2.imread(input_path)
    img_frame = cv2.resize(img_frame, (resize_width, resize_height))

    prev_tl_messages = ['STOP', 'STOP']  # Initialize

    ped_rects, ped_texts, ped_warning_texts = detect_ped_frame(pedestrian_model, img_frame, score_thr, iou_thr, conf_thr, warning_dst, device)
    ped_info = (ped_rects, ped_texts, ped_warning_texts)

    tl_rectangles, tl_texts, tl_messages = detect_tl_frame(traffic_light_model, img_frame, device, score_thr)
    tl_info = (tl_rectangles, tl_texts, tl_messages, prev_tl_messages)

    lane_info = detect_lane_frame(lane_model, img_frame, device, start_fraction=1/2)
    processed_frame, prev_tl_messages = draw_boxes_on_frame(img_frame, ped_info, tl_info, lane_info, task_type)

    cv2.imwrite(output_path, processed_frame)
    print(f'[INFO] Saved image to {output_path}')

def detect_video(task_type, pedestrian_model, traffic_light_model, lane_model, input_path, output_path, 
                 score_thr, iou_thr, conf_thr, warning_dst, device, 
                 resize_width=1280, resize_height=720):
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

    with tqdm(total=frame_cnt, desc="Processing Frames") as pbar:
        while True:
            hasFrame, img_frame = cap.read()
            if not hasFrame:
                print(f'Processed all frames')
                break
            img_frame = cv2.resize(img_frame, (resize_width, resize_height))
            
            ped_rects, ped_texts, ped_warning_texts = detect_ped_frame(pedestrian_model, img_frame, score_thr, iou_thr, conf_thr, warning_dst, device)
            ped_info = (ped_rects, ped_texts, ped_warning_texts)

            tl_rectangles, tl_texts, tl_messages  = detect_tl_frame(traffic_light_model, img_frame, device, score_thr)
            tl_info = (tl_rectangles, tl_texts, tl_messages, prev_tl_messages)

            lane_info = detect_lane_frame(lane_model, img_frame, device)
            processed_frame, prev_tl_messages = draw_boxes_on_frame(img_frame, ped_info, tl_info, lane_info, task_type)

            video_writer.write(processed_frame) 
            pbar.update(1)

    video_writer.release()
    cap.release()
    print(f'[INFO] Saved video to {avi_output_path}')

def detect_video2mp4(task_type, pedestrian_model, traffic_light_model, lane_model, input_path, output_path, 
                 score_thr, iou_thr, conf_thr, warning_dst, device, 
                 resize_width=1280, resize_height=720):
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

    with tqdm(total=frame_cnt, desc="Processing Frames") as pbar:
        while True:
            hasFrame, img_frame = cap.read()
            if not hasFrame:
                print(f'Processed all frames')
                break
            img_frame = cv2.resize(img_frame, (resize_width, resize_height))
            
            ped_rects, ped_texts, ped_warning_texts = detect_ped_frame(pedestrian_model, img_frame, score_thr, iou_thr, conf_thr, warning_dst, device)
            ped_info = (ped_rects, ped_texts, ped_warning_texts)

            tl_rectangles, tl_texts, tl_messages  = detect_tl_frame(traffic_light_model, img_frame, device, score_thr)
            tl_info = (tl_rectangles, tl_texts, tl_messages, prev_tl_messages)

            lane_info = detect_lane_frame(lane_model, img_frame, device)
            processed_frame, prev_tl_messages = draw_boxes_on_frame(img_frame, ped_info, tl_info, lane_info, task_type)

            video_writer.write(processed_frame) 
            pbar.update(1)

    video_writer.release()
    cap.release()
    print(f'[INFO] Saved video to {mp4_output_path}')

def main(task_type, CFG_DIR, OUTPUT_DIR, video_path, image_path, score_threshold): 
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") 
    cfg = load_yaml(CFG_DIR)

    print('[INFO] Loading models...')
    pedestrian_model = load_ped_model(cfg['pedestrian']['model_path'], cfg['pedestrian']['num_classes'], device)
    traffic_light_model = load_tl_model(cfg['traffic_light']['model_path'], cfg['traffic_light']['num_classes'], device)
    lane_model = load_lane_model(cfg['lane']['model_path'], cfg['lane']['model_type'], device)
    print('[INFO] Models loaded successfully')

    if video_path is not None:
        print(f'[INFO] Processing video: {video_path}')
        detect_video(task_type, pedestrian_model, traffic_light_model, lane_model, video_path, OUTPUT_DIR, score_threshold, cfg['pedestrian']['iou_threshold'], cfg['pedestrian']['confidence_threshold'], cfg['pedestrian']['warning_distance'], device)
    
    if image_path is not None:
        print(f'[INFO] Processing image: {image_path}')        
        detect_image(task_type, pedestrian_model, traffic_light_model, lane_model, image_path, OUTPUT_DIR, score_threshold, cfg['pedestrian']['iou_threshold'], cfg['pedestrian']['confidence_threshold'], cfg['pedestrian']['warning_distance'], device)

if  __name__ == "__main__":
    main(task_type="all", CFG_DIR='/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml', \
         OUTPUT_DIR='/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/app/videos/test_video4_clip_all_output.mp4', \
             video_path='/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/app/videos/clipped4.mp4', \
                image_path=None, 
                score_threshold=0.25)
    
    main(task_type="message", CFG_DIR='/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml', \
         OUTPUT_DIR='/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/app/videos/test_video4_clip_message_output.mp4', \
             video_path='/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/app/videos/clipped4.mp4', \
                image_path=None, 
                score_threshold=0.25)