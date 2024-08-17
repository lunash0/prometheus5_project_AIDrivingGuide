import torch
import cv2
from torchvision.transforms import functional as F
from tqdm import tqdm
from Pedestrian_Detection.utils import * 
from Pedestrian_Detection.engine import default_argument_parser, setup
from Pedestrian_Detection.model import load_model
import warnings
warnings.filterwarnings('ignore')

def process_frame(frame, model, device, iou_thresh=0.3, confidence_threshold=0.4):
    image = F.to_tensor(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)[0]

    output = apply_nms(output, iou_thresh)
    output = filter_boxes_by_score(output, confidence_threshold)

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    return boxes, labels, scores

def draw_boxes_on_frame(frame, boxes, labels, scores, score_threshold, warning_distance):
    frame_height = frame.shape[0]

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = map(int, box)
            label_text = 'Person' if label == 0 else 'Object'
            color = (255, 255, 0) if label_text == 'Person' else (0, 255, 255)

            distance = frame_height - y2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_text} Score: {score:.2f} Distance: {distance}px", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if distance < warning_distance:
                warning_text = "Collision Warning!"
                cv2.putText(frame, warning_text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    return frame

def detect_frame(model, img_frame, device, score_threshold, iou_threshold, confidence_threshold, warning_distance=167):
    boxes, labels, scores = process_frame(img_frame, model, device, iou_thresh=iou_threshold, confidence_threshold=confidence_threshold)
    img_frame = draw_boxes_on_frame(img_frame, boxes, labels, scores, score_threshold, warning_distance)
    return img_frame

def detect_video(model, input_path, output_path, device, score_thr, iou_thr, conf_thr, wartning_dst):
    cap = cv2.VideoCapture(input_path)
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

            img_frame = detect_frame(model, img_frame, device, score_thr, iou_thr, conf_thr, wartning_dst)
            video_writer.write(img_frame)

            pbar.update(1)

    video_writer.release()
    cap.release()
    print(f'[INFO] Saved video to {output_path}')


def draw_boxes_on_frame(frame, boxes, labels, scores, previous_heights, threshold=0.5, roi_points=None, object_width=50,
                        focal_length=1000, collision_height_threshold=500): #before: collision_height_threshold=300
    # if roi_points is not None:
        # cv2.polylines(frame, [roi_points], True, (0, 200, 0), 2)

    line_y1 = 600
    line_gap = 10
    line_ys = [line_y1 + i * line_gap for i in range(12)]
    line_colors = [(255, 0, 0) for _ in range(12)]
    crossed_lines = []
    # warning_text = ""

    current_heights = []  # Heights for current frame

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            box_height = y2 - y1  # Calculate the height of the bounding box
            current_heights.append(box_height)  # Track current heights

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label} Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            if roi_points is not None and cv2.pointPolygonTest(roi_points, (centroid_x, centroid_y), False) > 0:
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                for i, line_y in enumerate(line_ys):
                    if y2 >= line_y:
                        line_colors[i] = (0, 0, 255)
                        crossed_lines.append(i + 1)

                # Calculate distance
                box_width = x2 - x1
                if box_width > 0:
                    distance = (object_width * focal_length) / box_width
                    cv2.putText(frame, f"Dist: {distance:.2f} cm", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 2)

            # Collision avoidance
            if box_height > collision_height_threshold:
                # Determine warning severity
                height_excess = box_height - collision_height_threshold

                # Determine if the box is on the left, right, or near the center
                frame_center_x = frame.shape[1] // 2
                center_tolerance = frame.shape[1] // 10  # Define a tolerance range for "near center"

                if abs((x1 + x2) // 2 - frame_center_x) <= center_tolerance:
                    direction = "IN FRONT"
                elif x1 < frame_center_x:
                    direction = "ON THE LEFT"
                else:
                    direction = "ON THE RIGHT"

                # Set warning message based on height excess and direction
                if height_excess < 150:  # Adjust the threshold to define "Caution"
                    warning_msg = f"CAUTION: Potential Collision {direction}"
                else:  # Anything above this will be classified as "Warning"
                    warning_msg = f"WARNING: Approaching Collision {direction}"

                # Draw the warning text on the frame
                cv2.putText(frame, warning_msg, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            cv2.LINE_AA)

    return frame, current_heights  # Return updated heights for next frame


def main(args):
    cfg_dir, output_dir, model_path, video_name = setup(args)

    if len(video_name) == 0:
        video_name = os.path.basename(model_path).split('.pth')[0] + "_video"

    cfg = load_yaml(cfg_dir)['test']
    output_video_path = os.path.join(output_dir, video_name + '.mp4') 
    device = torch.device(f'cuda:{cfg["device"]}' if torch.cuda.is_available() else 'cpu')

    print('[INFO] Load model...')
    model = load_model(model_path, cfg['num_classes'], device) # eval mode
    print('[INFO] Model loaded successfully')
    print(f'[INFO] Processing video: {cfg["input_video_path"]}')
    print(f'[INFO] Thresholds: Score: {cfg["score_threshold"]}, IoU: {cfg["iou_threshold"]}, Confidence: {cfg["confidence_threshold"]}')
    detect_video(model, cfg['input_video_path'], output_video_path, device, cfg['score_threshold'], cfg['iou_threshold'], cfg['confidence_threshold'], cfg['warning_distance'])

if __name__ == "__main__":
    args = default_argument_parser()
    print("Command Line Args:", args)
    main(args) 
    

    