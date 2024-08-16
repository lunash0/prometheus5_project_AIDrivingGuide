import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib
from torchvision import transforms as transforms
from torchvision.transforms import functional as F

CLASSES = [
    'green', 'red', 'yellow', 'red and green arrow', 'red and yellow', 'green and green arrow', 'green and yellow',
    'yellow and green arrow', 'green arrow and green arrow', 'red cross', 'green arrow(down)', 'green arrow', 'etc'
]

MAP_CLASSES_COLORS = {
     0: 3,
     1: 1,
     2: 2, 
     3: 4,
     4: 4, 
     5: 4, 
     6: 4, 
     7: 4, 
     8: 4,
     9: 1,
     10: 3,
     11: 3,
     12: 0
}
          
COLORS = [
    [0, 0, 0], # etc
    [255, 0, 0], # red
    [255, 255, 0], # yellow
    [0, 255, 0], # green
    [255, 255, 255] # Mixed
]

def process_frame(frame, model, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_input = F.to_tensor(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_input.to(device))
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    return image_input, outputs[0]

def detect_tl_frame(model, frame, device, score_threshold=0.25):
    image, outputs = process_frame(frame, model, device)
    image = image

    rectangles = []
    texts = []
    messages = []
    
    if len(outputs['boxes']) != 0:
            boxes = outputs['boxes'].data.numpy()
            scores = outputs['scores'].data.numpy()
            # Filter out boxes according to `detection_threshold`.
            boxes = boxes[scores >= score_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # Get all the predicited class names
            pred_classes_colors = [MAP_CLASSES_COLORS[i] for i in outputs['labels'].cpu().numpy()]
            pred_classes = [CLASSES[i] for i in outputs['labels'].cpu().numpy()]

            # Draw the bounding boxes and write the class name on top of it.
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color_idx = pred_classes_colors[j]
                color = COLORS[color_idx]

                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])

                rectangles.append([[(xmin, ymin), (xmax, ymax)], color[::-1]])
                texts.append([class_name, (xmin, ymin - 5), color[::-1]])

                if color_idx in [1, 2, 3]:
                    if color_idx == 1:
                        message = 'STOP'
                    elif color_idx == 2:
                        message = 'WAIT'
                    elif color_idx == 3:
                        message = 'GO'
                    messages.append(message)       

    return rectangles, texts, messages
