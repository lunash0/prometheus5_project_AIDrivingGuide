import os
import argparse
import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data(img, transform):
    img = Image.fromarray(img).convert("RGB")  # 이미지를 RGB로 변환합니다.
    img = transform(img)
    return img

def process_frame(model, frame, transform, resize_width, resize_height):
    input_img = cv2.resize(frame, (resize_width, resize_height))
    tensor_img = load_test_data(input_img, transform).to(DEVICE)
    tensor_img = torch.unsqueeze(tensor_img, dim=0)
    outputs = model(tensor_img)
    binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255
    return binary_pred

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

    if hasattr(args, 'video') and args.video:  # 비디오 파일을 입력으로 받을 때
        cap = cv2.VideoCapture(args.video)
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 비디오의 FPS 가져오기
        out = None
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test_output/result.avi', fourcc, fps, (resize_width, resize_height), False)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            binary_pred = process_frame(model, frame, data_transform, resize_width, resize_height)
            binary_pred_resized = cv2.resize(binary_pred, (resize_width, resize_height)).astype(np.uint8)
            out.write(binary_pred_resized)

        cap.release()
        out.release()
        
        # Display the result video in Colab
        from IPython.display import Video
        return Video('test_output/result.avi')

    elif hasattr(args, 'img') and args.img:  # 이미지 파일을 입력으로 받을 때
        img_path = args.img
        if not os.path.exists(img_path):
            print(f"Image path {img_path} does not exist.")
            return

        input_img = cv2.imread(img_path)
        if input_img is None:
            print(f"Failed to load image {img_path}.")
            return

        dummy_input = load_test_data(input_img, data_transform).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        outputs = model(dummy_input)

        input_img = cv2.resize(input_img, (resize_width, resize_height))

        instance_pred = torch.squeeze(outputs["instance_seg_logits"].detach().to("cpu")).numpy() * 255
        binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255

        cv2.imwrite(os.path.join("test_output", "input.jpg"), input_img)
        cv2.imwrite(
            os.path.join("test_output", "instance_output.jpg"),
            instance_pred.transpose((1, 2, 0)),
        )
        cv2.imwrite(os.path.join("test_output", "binary_output.jpg"), binary_pred.astype(np.uint8))

def parse_args():
    parser = argparse.ArgumentParser(description="LaneNet Test")
    parser.add_argument("--img", type=str, help="Path to the input image")
    parser.add_argument("--video", type=str, help="Path to the input video")
    parser.add_argument("--height", type=int, default=720, help="Resize height")
    parser.add_argument("--width", type=int, default=1280, help="Resize width")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("—model_type", type=str, required=True, help="Type of the model architecture")
    return parser.parse_args()

if __name__ == "__main__":
    test()