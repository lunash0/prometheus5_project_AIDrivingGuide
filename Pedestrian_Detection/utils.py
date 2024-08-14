import torch 
import numpy as np
import json
import cv2 
import matplotlib.pyplot as plt
import re 
import os 
from torchvision.ops import nms 
import yaml 
import albumentations as A
from albumentations.pytorch import ToTensorV2

def collate_fn(batch):
    """
    Combines a list of samples into a batch.
    """
    return tuple(zip(*batch))

def load_yaml(file_path: str) -> dict:
    """
    Loads a YAML configuration file.
    """
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_wh(image_size):
    """
    Retrieves width and height from the image size tuple.

    Args:
        image_size (tuple): Tuple containing width and height.
    Returns:
        tuple: Width and height of the image.
    """
    w = image_size[0]
    h = image_size[1]
    return w, h  # width, height

def train_transform(cfg_dir):
    """
    Defines the transformation pipeline for training images.
    """
    WIDTH, HEIGHT = get_wh(load_yaml(cfg_dir)['train']['image_size'])
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(height=HEIGHT, width=WIDTH),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def valid_transform(cfg_dir):
    """
    Defines the transformation pipeline for validation images.
    """
    WIDTH, HEIGHT = get_wh(load_yaml(cfg_dir)['train']['image_size'])
    return A.Compose([
        A.Resize(height=HEIGHT, width=WIDTH),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def box_denormalize(x1, y1, x2, y2, width, height, cfg_dir):
    """
    Converts normalized bounding box coordinates to original image coordinates.

    Args:
        x1 (float): Normalized x1 coordinate.
        y1 (float): Normalized y1 coordinate.
        x2 (float): Normalized x2 coordinate.
        y2 (float): Normalized y2 coordinate.
        width (int): Width of the original image.
        height (int): Height of the original image.
        cfg_dir (str): Path to the configuration file.
    Returns:
        tuple: Bounding box coordinates in the original image scale.
    """
    WIDTH, HEIGHT = get_wh(load_yaml(cfg_dir)['train']['image_size'])
    x1 = (x1 / WIDTH) * width
    y1 = (y1 / HEIGHT) * height
    x2 = (x2 / WIDTH) * width
    y2 = (y2 / HEIGHT) * height
    return x1.item(), y1.item(), x2.item(), y2.item()

def normalize_bbox(bboxes, width, height):
    """
    Normalizes bounding box coordinates to the range [0, 1].

    Args:
        bboxes (list of lists): List of bounding boxes with coordinates.
        width (int): Width of the image.
        height (int): Height of the image.
    Returns:
        list of lists: Normalized bounding boxes.
    """
    return [[xmin / width, ymin / height, xmax / width, ymax / height] for xmin, ymin, xmax, ymax in bboxes]

def calculate_IoU(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (tuple): Coordinates of the first bounding box (x1, y1, x2, y2).
        box2 (tuple): Coordinates of the second bounding box (x1, y1, x2, y2).
    Returns:
        float: IoU score between the two bounding boxes.
    """
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def apply_nms(orig_prediction, iou_thresh=0.3):
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

    Args:
        orig_prediction (dict): Dictionary containing bounding boxes and scores.
        iou_thresh (float): IoU threshold for NMS.
    Returns:
        dict: Filtered bounding boxes after applying NMS.
    """
    keep = nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = {k: v[keep] for k, v in orig_prediction.items()}
    return final_prediction

def filter_boxes_by_score(output, threshold):
    """
    Filters bounding boxes based on a score threshold.

    Args:
        output (dict): Dictionary containing bounding boxes and scores.
        threshold (float): Minimum score threshold for filtering.
    Returns:
        dict: Filtered bounding boxes based on score threshold.
    """
    keep = output['scores'] > threshold
    filtered_output = {k: v[keep] for k, v in output.items()}
    return filtered_output

def draw_boxes_on_image_val(image_path, pred_boxes, gt_boxes, pred_labels, gt_labels, save_path):
    """
    Draws predicted and ground truth bounding boxes on an image and saves the result.

    Args:
        image_path (str): Path to the input image.
        pred_boxes (list of lists): List of predicted bounding boxes.
        gt_boxes (list of lists): List of ground truth bounding boxes.
        pred_labels (list of str): List of predicted labels.
        gt_labels (list of str): List of ground truth labels.
        save_path (str): Path to save the output image.
    """
    image = cv2.imread(image_path)
    
    for box, label in zip(pred_boxes, pred_labels):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Pred: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'GT: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imwrite(save_path, image)
    print(f'Saved image with bounding boxes to {save_path}')

def draw_boxes_on_image(image, boxes, labels, save_path):
    """
    Draws bounding boxes and labels on an image and saves the result.

    Args:
        image (numpy array): Image to draw boxes on.
        boxes (list of lists): List of bounding boxes.
        labels (list of str): List of labels.
        save_path (str): Path to save the output image.
    """
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(save_path, image)

def visualize_image(image_tensor):
    """
    Visualizes an image tensor by converting it to a displayable format.

    """
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    plt.imshow(image)
    plt.show()
