import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib

from model import create_model
from torchvision import transforms as transforms
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--weights',
    default='original_outputs/best_model.pth',
    help='path to the model weights'
)
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='test_video2.mp4'
)
parser.add_argument(
    '--imgsz',
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
args = parser.parse_args()

os.makedirs('original_inference_outputs/videos', exist_ok=True)

# RGB format.
COLORS = [
    [0, 0, 0],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [255, 255, 255]
]

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load(args.weights, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Define the detection threshold.
detection_threshold = 0.2

cap = cv2.VideoCapture(args.input)

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# Get the frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = str(pathlib.Path(args.input)).split(os.path.sep)[-1].split('.')[0]
print(save_name)
# Define codec and create VideoWriter object .
out = cv2.VideoWriter(f"original_inference_outputs/videos/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))

frame_count = 0  # To count total frames.
total_fps = 0  # To get the final frames per second.


def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)


# Read until end of video.
while (cap.isOpened()):
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        if args.imgsz is not None:
            image = cv2.resize(image, (args.imgsz, args.imgsz))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply transforms
        image_input = infer_transforms(image)
        image_input = torch.unsqueeze(image_input, 0)
        # Get the start time.
        start_time = time.time()
        # Predictions
        with torch.no_grad():
            outputs = model(image_input.to(DEVICE))
        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Total FPS till current frame.
        total_fps += fps
        frame_count += 1

        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # Filter out boxes according to `detection_threshold`.
            boxes = boxes[scores >= args.threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # Get all the predicited class names.
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # Draw the bounding boxes and write the class name on top of it.
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]
                # Recale boxes.
                xmin = int((box[0] / image.shape[1]) * frame.shape[1])
                ymin = int((box[1] / image.shape[0]) * frame.shape[0])
                xmax = int((box[2] / image.shape[1]) * frame.shape[1])
                ymax = int((box[3] / image.shape[0]) * frame.shape[0])
                cv2.rectangle(frame,
                              (xmin, ymin),
                              (xmax, ymax),
                              color[::-1],
                              3)
                cv2.putText(frame,
                            class_name,
                            (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color[::-1],
                            2,
                            lineType=cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.0f} FPS",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, lineType=cv2.LINE_AA)

        cv2.imshow('image', frame)
        out.write(frame)
        # Press `q` to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
