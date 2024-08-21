import torch
from torchvision.transforms import functional as F
from models.Pedestrian_Detection.utils import * 
import warnings
warnings.filterwarnings('ignore')
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, RetinaNetClassificationHead

"""1. Pedestrian Detection Model"""
def build_ped_model(num_classes):
    model = retinanet_resnet50_fpn(pretrained=True)
    
    in_features = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_features, num_anchors, num_classes)
    
    return model

def load_ped_model(checkpoint_path, num_classes, device):
    model = build_ped_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model

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

def detect_ped_frame_simple(model, frame, score_thr, iou_thr, conf_thr, warning_distance, device):
    boxes, labels, scores = process_frame(frame, model, device, iou_thresh=iou_thr, confidence_threshold=conf_thr)
    frame_height = frame.shape[0]
    
    rectangles = []
    texts = []
    warning_texts = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_thr:
            x1, y1, x2, y2 = map(int, box)
            label_text = 'Person' if label == 0 else 'Object'
            color = (255, 255, 0) if label_text == 'Person' else (0, 255, 255)

            distance = frame_height - y2
            rectangles.append([[(x1, y1), (x2, y2)], color])
            texts.append([f"{label_text} Score: {score:.2f} Dist: {distance}px", (x1, y1 - 10),color])

            if distance < warning_distance:
                warning_text = "Collision Warning!"
            else:
                warning_text = ""
            warning_texts.append([warning_text, (x1, y2 + 30)])
    return rectangles, texts, warning_texts

def detect_ped_frame(model, frame, score_thr, iou_thr, conf_thr, warning_distance, device, 
                     roi_points=None, object_width=50, focal_length=1000, 
                     collision_height_threshold=300):
    
    boxes, labels, scores = process_frame(frame, model, device, iou_thresh=iou_thr, confidence_threshold=conf_thr)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    rectangles = []
    texts = []
    warning_texts = []
    
    line_y1 = 600
    line_gap = 10
    line_ys = [line_y1 + i * line_gap for i in range(12)]
    line_colors = [(255, 0, 0) for _ in range(12)]
    crossed_lines = []

    current_heights = []  # Heights for current frame
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_thr:
            x1, y1, x2, y2 = map(int, box)
            box_height = y2 - y1  # Calculate the height of the bounding box
            current_heights.append(box_height)  # Track current heights

            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)
            
            label_text = 'Person' if label == 0 else 'Object'
            color = (255, 255, 0) if label_text == 'Person' else (0, 255, 255)

            # Draw the bounding box and label
            rectangles.append([[(x1, y1), (x2, y2)], color])
            texts.append([f"{label_text} Score: {score:.2f}", (x1, y1 - 10), color])

            if roi_points is not None and cv2.pointPolygonTest(roi_points, (centroid_x, centroid_y), False) > 0:
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                for i, line_y in enumerate(line_ys):
                    if y2 >= line_y:
                        line_colors[i] = (0, 0, 255)
                        crossed_lines.append(i + 1)

                # Calculate distance
                box_width = x2 - x1
                if box_width > 0:
                    distance_cm = (object_width * focal_length) / box_width
                    texts.append([f"Dist: {distance_cm:.2f} cm", (x1, y1 - 30), (0, 255, 255)])

            # Collision warning based on height threshold
            if box_height > collision_height_threshold:
                height_excess = box_height - collision_height_threshold

                # Determine object position relative to frame center
                frame_center_x = frame_width // 2
                center_tolerance = frame_width // 10  # Tolerance range for "near center"

                if abs(centroid_x - frame_center_x) <= center_tolerance:
                    direction = "IN FRONT"
                elif x1 < frame_center_x:
                    direction = "ON THE LEFT"
                else:
                    direction = "ON THE RIGHT"

                # Set warning message based on height excess and direction
                if height_excess < 150:  # Define "Caution" threshold
                    warning_msg = f"CAUTION: Potential Collision {direction}"
                else:  # Anything above this is classified as "Warning"
                    warning_msg = f"WARNING: Approaching Collision {direction}"

                # Add warning message
                warning_texts.append([warning_msg, (x1, y1 - 50)])

            # Previous warning based on distance from bottom of frame
            # distance_px = frame_height - y2
            # if distance_px < warning_distance:
            #     warning_texts.append(["Collision Warning!", (x1, y2 + 30)])
    
    return rectangles, texts, warning_texts
