import os
import argparse
import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
def load_test_data(img, transform):
    img = Image.fromarray(img).convert("RGB")  # 이미지를 RGB로 변환합니다.
    img = transform(img)
    return img

def process_frame(model, frame, transform, device):
    tensor_img = load_test_data(frame, transform).to(v)
    tensor_img = torch.unsqueeze(tensor_img, dim=0)
    outputs = model(tensor_img)
    binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255
    return binary_pred

def detect_frame(model, frame, ):
    binary_pred = process_frame(model, frame, data_transform).astype(np.uint8)
    return binary_pred 
