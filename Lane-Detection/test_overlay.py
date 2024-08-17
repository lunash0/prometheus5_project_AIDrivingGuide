import os
import argparse
import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from scipy.spatial import distance as dist

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img, transform):
    img = Image.fromarray(img).convert("RGB")
    img = transform(img)
    return img


def overlay_binary_mask(image, binary_mask, alpha=0.5, color=(255, 0, 0)):
    color_mask = np.zeros_like(image)
    color_mask[binary_mask > 0] = color
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay


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


def process_frame(model, frame, transform, resize_width, resize_height):
    input_img = cv2.resize(frame, (resize_width, resize_height))
    tensor_img = load_test_data(input_img, transform).to(DEVICE)
    tensor_img = torch.unsqueeze(tensor_img, dim=0)
    outputs = model(tensor_img)
    binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255
    binary_pred_resized = cv2.resize(binary_pred, (resize_width, resize_height)).astype(
        np.uint8
    )

    contours, _ = cv2.findContours(
        binary_pred_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    merged_contours = merge_close_contours(contours, threshold=1)  # threshold

    weighted_contours = sorted(
        merged_contours,
        key=lambda c: calculate_weighted_score(c, resize_width),
        reverse=True,
    )[:5]
    # contours count

    top5_mask = np.zeros_like(binary_pred_resized)
    cv2.drawContours(top5_mask, weighted_contours, -1, 255, thickness=cv2.FILLED)

    overlay_img = overlay_binary_mask(
        input_img, top5_mask, alpha=0.5, color=(0, 0, 255)
    )
    return overlay_img


def test():
    if not os.path.exists("test_output"):
        os.mkdir("test_output")
    args = parse_args()

    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose(
        [
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    if hasattr(args, "video") and args.video:
        cap = cv2.VideoCapture(args.video)

        fps = cap.get(cv2.CAP_PROP_FPS)
        out = None
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            "test_output/result.avi", fourcc, fps, (resize_width, resize_height), True
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            overlay_frame = process_frame(
                model, frame, data_transform, resize_width, resize_height
            )
            out.write(overlay_frame)

        cap.release()
        out.release()

        from IPython.display import Video

        return Video("test_output/result.avi")

    elif hasattr(args, "img") and args.img:
        img_path = args.img
        if not os.path.exists(img_path):
            print(f"Image path {img_path} does not exist.")
            return

        input_img = cv2.imread(img_path)
        if input_img is None:
            print(f"Failed to load image {img_path}.")
            return

        overlay_img = process_frame(
            model, input_img, data_transform, resize_width, resize_height
        )

        cv2.imwrite(os.path.join("test_output", "overlay_output.jpg"), overlay_img)


def parse_args():
    parser = argparse.ArgumentParser(description="LaneNet Test")
    parser.add_argument("--img", type=str, help="Path to the input image")
    parser.add_argument("--video", type=str, help="Path to the input video")
    parser.add_argument("--height", type=int, default=720, help="Resize height")
    parser.add_argument("--width", type=int, default=1280, help="Resize width")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument(
        "--model_type", type=str, required=True, help="Type of the model architecture"
    )
    return parser.parse_args()


if __name__ == "__main__":
    test()
