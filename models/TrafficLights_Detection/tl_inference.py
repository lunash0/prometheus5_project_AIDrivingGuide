import numpy as np
import cv2
import torch
from torchvision import transforms as transforms
from torchvision.transforms import functional as F
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import torchvision
from functools import partial

"""2. Traffic Light Detection Model"""
def build_tl_model(num_classes=91):
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights='DEFAULT'
    )
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    return model

def load_tl_model(checkpoint_path, num_classes, device):
    model = build_tl_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model 

CLASSES = [
    'go', 'stop', 'wait', 'red and green arrow', 'stop/wait', 'green and green arrow', 'go/wait',
    'wait/go', 'go arrow', 'stop cross', 'go arrow(down)', 'go arrow', 'etc'
]

MAP_CLASSES_COLORS = { # maps to stop if the class includes red for the driving safety
     0: 3, # green = go
     1: 1, # red = stop
     2: 2, # yellow = wait
     3: 4, # red and green arrow
     4: 1, # red and yellow = stop/wait
     5: 3, # green and green arror
     6: 2, # green and yellow = go/wait
     7: 2, # yellow and green arrow = wait/go
     8: 3, # green arrow and green arrow = go arrow
     9: 1, # red cross = stop cross
     10: 3, # green arrow(down) = go arrow(down)
     11: 3, # green arrow = go arrow
     12: 0 # etc
}
          
COLORS = [
    [0, 0, 0], # etc 0
    [255, 0, 0], # red(stop) 1
    [255, 255, 0], # yellow(wait) 2
    [0, 255, 0], # green(go) 3
    [255, 255, 255] # Mixed 4
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
                        message = 'PROCEED WITH CAUTION'
                    messages.append(message)       

    return rectangles, texts, messages

def message_rule(messages, prev_tl_messages, debug=False):
    prev_tl_message = prev_tl_messages[-1]
    message = max(messages)
    if message == 'STOP':
        color = (0, 0, 255)
    elif message in ['WAIT', 'PREPARE TO PROCEED', 'PREPARE WITH CAUTION']:
        if prev_tl_message == 'STOP' and prev_tl_messages[-2] in ['STOP', 'PREPARE TO PROCEED']:
            message = 'PREPARE TO PROCEED'
        elif prev_tl_message == 'PROCEED' and prev_tl_messages[-2] in ['STOP', 'PREPARE TO STOP']:
            message = 'PREPARE TO STOP'
        color = (0, 152, 255)
    elif message == 'PROCEED WITH CAUTION':
        color = (0, 248, 211)
    
    # sanity check
    if debug:
        print(f'[DEBUG] Add message {message}')

    prev_tl_messages.append(message) 
    return message, color, prev_tl_messages 