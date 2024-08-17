import torch
from models.Lane_Detection.dataloader.transformers import Rescale
from models.Lane_Detection.model.lanenet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from scipy.spatial import distance as dist

"""3. Lane Detection Model"""

def build_lane_model(device, model_type='ENet'):
    model = LaneNet(DEVICE=device, arch=model_type)
    return model
def load_lane_model(checkpoint_path, model_type, device):
    model = build_lane_model(device, model_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_test_data(img, transform):
    img = Image.fromarray(img).convert("RGB")
    img = transform(img)
    return img


def merge_close_contours(contours, threshold=50):
    merged_contours = []
    used = [False] * len(contours)

    for i, c1 in enumerate(contours):
        if used[i]:
            continue
        merged_contour = c1
        used[i] = True

        for j, c2 in enumerate(contours):
            if i == j or used[j]:
                continue

            d = dist.cdist(
                merged_contour.reshape(-1, 2), c2.reshape(-1, 2), "euclidean"
            )
            if np.any(d < threshold):
                merged_contour = np.vstack((merged_contour, c2))
                used[j] = True

        merged_contours.append(merged_contour)

    return merged_contours


def calculate_weighted_score(contour, image_width):
    length = cv2.arcLength(contour, closed=True)

    contour_center_x = np.mean(contour[:, 0, 0])

    distance_from_center = abs(contour_center_x - image_width / 2)

    weighted_score = length / (distance_from_center + 1)

    return weighted_score

def overlay_binary_mask(image, binary_mask, alpha=0.01, color=(255, 0, 0)):
    color_mask = np.zeros_like(image)
    color_mask[binary_mask > 0] = color
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return [1-alpha, color_mask, alpha]

def draw_lane_contours(image, binary_pred, color=(0, 255, 255), start_fraction=2/3,resize_width=1080):
    contours, _ = cv2.findContours(binary_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_contours = merge_close_contours(contours, threshold=1)  # threshold

    weighted_contours = sorted(
        merged_contours,
        key=lambda c: calculate_weighted_score(c, resize_width),
        reverse=True,
    )[:5]
    return [weighted_contours, -1, color]

def process_frame(model, frame, transform, device, start_fraction):
    input_img = frame # cv2.resize(frame, (resize_width, resize_height)) 
    tensor_img = load_test_data(input_img, transform).to(device)
    tensor_img = torch.unsqueeze(tensor_img, dim=0)
    outputs = model(tensor_img)
    binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255
    binary_pred = binary_pred.astype(np.uint8)
    overlay_img_info = draw_lane_contours(input_img, binary_pred, start_fraction) #Modified(Yoojin)
    return overlay_img_info

# def draw_lane_contours(image, binary_mask, color=(0, 255, 255), start_fraction=2/3):
#     height = image.shape[0]
#     partial_mask = np.zeros_like(binary_mask)
#     start_height = int(height * start_fraction)
#     partial_mask[start_height:, :] = binary_mask[start_height:, :]
    
#     contours, _ = cv2.findContours(partial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.drawContours(image, contours, -1, color, thickness=3)
#     return [contours, -1, color]

# def process_frame(model, frame, transform, device, start_fraction):
#     input_img = frame # cv2.resize(frame, (resize_width, resize_height)) 
#     tensor_img = load_test_data(input_img, transform).to(device)
#     tensor_img = torch.unsqueeze(tensor_img, dim=0)
#     outputs = model(tensor_img)
#     binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255
#     binary_pred = binary_pred.astype(np.uint8)
#     # overlay_img_info = overlay_binary_mask(input_img, binary_pred, alpha=0.5, color=(0, 0, 255))
#     overlay_img_info = draw_lane_contours(input_img, binary_pred, start_fraction) #Modified(Yoojin)
#     return overlay_img_info

def detect_lane_frame(model, frame, device, start_fraction=2/3):
    lane_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    return process_frame(model, frame, lane_transform, device, start_fraction)
