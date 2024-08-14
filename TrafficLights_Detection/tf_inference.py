import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib
from torchvision import transforms as transforms

CLASSES = [
    'green', 'red', 'yellow', 'red and green arrow', 'red and yellow', 'green and green arrow', 'green and yellow',
    'yellow and green arrow', 'green arrow and green arrow', 'red cross', 'green arrow(down)', 'green arrow', 'etc'
]

MAP_CLASSES = {
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
MAP_CLASSES_NAMES = ['etc', 'red', 'yellow', 'green', 'Mixed']
from torchvision.transforms import functional as F

# def infer_transforms(image):
#     # Define the torchvision image transforms.
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),
#     ])
#     return transform(image)

def process_frame(frame, model, device):
    # image_input = infer_transforms(frame)
    
    # image_input = torch.unsqueeze(image_input, 0)
    image_input = F.to_tensor(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_input.to(device))
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    return image_input, outputs[0]

def detect_tl_frame(model, frame, device, score_threshold=0.25):
    image, outputs = process_frame(frame, model, device)
    rectangles = []
    texts = []
    if len(outputs['boxes']) != 0:
            boxes = outputs['boxes'].data.numpy()
            scores = outputs['scores'].data.numpy()
            # Filter out boxes according to `detection_threshold`.
            boxes = boxes[scores >= score_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # Get all the predicited class names
            pred_classes = [MAP_CLASSES[i] for i in outputs['labels'].cpu().numpy()]

            # Draw the bounding boxes and write the class name on top of it.
            for j, box in enumerate(draw_boxes):
                class_name = MAP_CLASSES_NAMES[pred_classes[j]]
                color = COLORS[pred_classes[j]]
                # Recale boxes.
                # xmin = int((box[0] / image.shape[1]) * frame.shape[1])
                # ymin = int((box[1] / image.shape[0]) * frame.shape[0])
                # xmax = int((box[2] / image.shape[1]) * frame.shape[1])
                # ymax = int((box[3] / image.shape[0]) * frame.shape[0])
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])
                # cv2.rectangle(frame,
                #               (xmin, ymin),
                #               (xmax, ymax),
                #               color[::-1],
                #               3)
                rectangles.append([[(xmin, ymin), (xmax, ymax)], color[::-1]])
                texts.append([class_name, (xmin, ymin - 5), color[::-1]])
                # cv2.putText(frame,
                #             class_name,
                #             (xmin, ymin - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.8,
                #             color[::-1],
                #             2,
                #             lineType=cv2.LINE_AA)
    return rectangles, texts
