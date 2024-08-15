import torch
import cv2
import numpy as np
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.ops import nms


# Load the RetinaNet model from a checkpoint
def load_retinanet_model(checkpoint_path, device, num_classes=2):
    model = retinanet_resnet50_fpn(pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


# Filter boxes by confidence score
def filter_boxes_by_score(output, threshold):
    if len(output['scores']) == 0:
        return output
    keep = output['scores'] > threshold
    filtered_output = {k: v[keep] for k, v in output.items()}
    return filtered_output


# Apply Non-Maximum Suppression (NMS)
def apply_nms(orig_prediction, iou_thresh=0.3):
    if len(orig_prediction['boxes']) == 0:
        return orig_prediction
    keep = nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = {k: v[keep] for k, v in orig_prediction.items()}
    return final_prediction


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


# Main function to run the detection and processing pipeline
def main(video_path, checkpoint_path, device='cpu'):
    model = load_retinanet_model(checkpoint_path, device)

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
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    roi_points = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32)  # Example ROI
    previous_heights = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, labels, scores = process_frame(frame, model, device, iou_thresh=0.4, confidence_threshold=0.4)
        frame, previous_heights = draw_boxes_on_frame(frame, boxes, labels, scores, previous_heights, threshold=0.4,
                                                      roi_points=roi_points, collision_height_threshold=100)

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
    output_path = '/Users/vitaminmin/Downloads/output_video.mp4' #output_video를 저장할 경로
    video_path = '/Users/vitaminmin/Downloads/test_video2.mp4' #test_video2가 저장된 경로
    checkpoint_path = '/Users/vitaminmin/Downloads/best_retinanet_e10s0.89l0.80.pth' #체크포인트 모델이 저장된 경로
    main(video_path, checkpoint_path)