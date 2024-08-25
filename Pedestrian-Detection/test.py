import torch
import cv2
import numpy as np
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, RetinaNetClassificationHead
from torchvision.transforms import functional as F
from utils import filter_boxes_by_score, apply_nms
import warnings

warnings.filterwarnings('ignore')


# Load the model
def build_ped_model(num_classes):
    model = retinanet_resnet50_fpn(pretrained=True)

    in_features = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_features, num_anchors, num_classes)

    return model


def load_ped_model(model_path, num_classes, device):
    model = build_ped_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


# Process each frame
def process_frame(frame, model, device, iou_thresh=0.3, confidence_threshold=0.4):
    image = F.to_tensor(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)[0]

    # Apply NMS and confidence threshold filtering
    output = apply_nms(output, iou_thresh)
    output = filter_boxes_by_score(output, confidence_threshold)

    boxes = output.get("boxes", torch.tensor([])).cpu().numpy()
    labels = output.get("labels", torch.tensor([])).cpu().numpy()
    scores = output.get("scores", torch.tensor([])).cpu().numpy()

    return boxes, labels, scores


def detect_ped_frame(model, frame, score_thr, iou_thr, conf_thr, device,
                     roi_points=None, object_width=50, focal_length=1000,
                     collision_height_threshold=400):
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

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_thr:
            x1, y1, x2, y2 = map(int, box)
            box_height = y2 - y1  # Calculate the height of the bounding box

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
            distance_px = frame_height - y2
            if distance_px < 100:
                warning_texts.append(["CAREFUL", (x1, y2 + 30)])

    return rectangles, texts, warning_texts

def draw_boxes_on_frame(frame, rectangles, texts, warning_texts):
    for rect, color in rectangles:
        cv2.rectangle(frame, rect[0], rect[1], color, 2)
    for text, position, color in texts:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for text, position in warning_texts:
        if text:
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

# Main function to run the detection and processing pipeline
def main(video_path, model_path, device='cpu'):
    model = load_ped_model(model_path, num_classes=2, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
    output_path = '/Users/vitaminmin/Downloads/outputVideo.mp4'  # Output path(outputVideo를 저장할 경로)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    roi_points = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32)  # Example ROI

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rectangles, texts, warning_texts = detect_ped_frame(model, frame, score_thr=0.4, iou_thr=0.3, conf_thr=0.4,
                                                            device=device, roi_points=roi_points,
                                                            collision_height_threshold=100)

        frame = draw_boxes_on_frame(frame, rectangles, texts, warning_texts)

        # Write the processed frame to the video file
        out.write(frame)

        # Display the frame
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = '/Users/vitaminmin/Downloads/test_video2.mp4'  # Path to the input video(test_video2가 저장된 경로)
    model_path = '/Users/vitaminmin/Downloads/best_retinanet_e10s0.89l0.80.pth'  # Path to the model checkpoint(체크포인트 모델이 저장된 경로)
    main(video_path, model_path)