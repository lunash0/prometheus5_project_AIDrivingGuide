import os
import albumentations as A
import cv2
import numpy as np
import torch
# import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont  # using plt -> using PIL

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES

# plt.style.use('ggplot')


# This class keeps track of the training and validation loss values
# and helps to get the average for each epoch as well.
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation mAP @0.5:0.95 IoU higher than the previous highest, then save the
    model state.
    """
    def __init__(
        self, best_valid_map=float(0)
    ):
        self.best_valid_map = best_valid_map
        
    def __call__(
        self, 
        model, 
        current_valid_map, 
        epoch, 
        OUT_DIR,
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, f"{OUT_DIR}/best_model.pth")


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# Define the training tranforms.
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# Define the validation transforms.
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

# ## original show_tranformed_iamge()
# def show_tranformed_image(train_loader):
#     """
#     This function shows the transformed images from the `train_loader`.
#     Helps to check whether the tranformed images along with the corresponding
#     labels are correct or not.
#     Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
#     """
#     if len(train_loader) > 0:
#         for i in range(1):
#             images, targets = next(iter(train_loader))
#             images = list(image.to(DEVICE) for image in images)
#             targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#             boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
#             labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
#             sample = images[i].permute(1, 2, 0).cpu().numpy()
#             sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
#             for box_num, box in enumerate(boxes):
#                 cv2.rectangle(sample,
#                             (box[0], box[1]),
#                             (box[2], box[3]),
#                             (0, 0, 255), 2)
#                 cv2.putText(sample, CLASSES[labels[box_num]],
#                             (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
#                             1.0, (0, 0, 255), 2)
#             cv2.imshow('Transformed image', sample)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
def show_tranformed_image(train_loader):
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = (sample * 255).astype(np.uint8)
            sample = Image.fromarray(sample)
            draw = ImageDraw.Draw(sample)
            font = ImageFont.load_default()

            for box_num, box in enumerate(boxes):
                draw.rectangle(
                    [(box[0], box[1]), (box[2], box[3])],
                    outline="red", width=2
                )
                draw.text(
                    (box[0], box[1]-10),
                    CLASSES[labels[box_num]],
                    fill="red",
                    font=font
                )
            sample.show()


def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    filename = f'outputs/model_epoch_{epoch + 1}.pth'
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, filename)


# original save_loss_plot()
# def save_loss_plot(
#     OUT_DIR,
#     train_loss_list,
#     x_label='iterations',
#     y_label='train loss',
#     save_name='train_loss'
# ):
#     """
#     Function to save both train loss graph.
#
#     :param OUT_DIR: Path to save the graphs.
#     :param train_loss_list: List containing the training loss values.
#     """
#     figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
#     train_ax = figure_1.add_subplot()
#     train_ax.plot(train_loss_list, color='tab:blue')
#     train_ax.set_xlabel(x_label)
#     train_ax.set_ylabel(y_label)
#     figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
#     print('SAVING PLOTS COMPLETE...')
# def save_loss_plot(
#     OUT_DIR,
#     train_loss_list,
#     x_label='iterations',
#     y_label='train loss',
#     save_name='train_loss'
# ):
#     from PIL import Image, ImageDraw
#
#     width, height = 800, 600
#     image = Image.new("RGB", (width, height), "white")
#     draw = ImageDraw.Draw(image)
#
#     max_loss = max(train_loss_list)
#     min_loss = min(train_loss_list)
#     normalized_loss = [
#         height - ((loss - min_loss) / (max_loss - min_loss) * height)
#         for loss in train_loss_list
#     ]
#
#     step = width / len(train_loss_list)
#     for i in range(1, len(normalized_loss)):
#         draw.line(
#             [(step * (i - 1), normalized_loss[i - 1]),
#              (step * i, normalized_loss[i])],
#             fill="blue", width=2
#         )
#
#     draw.text((10, 10), x_label, fill="black")
#     draw.text((10, height - 20), y_label, fill="black")
#
#     image.save(f"{OUT_DIR}/{save_name}.png")
#     print('SAVING PLOTS COMPLETE...')


def save_loss_plot(
        out_dir,
        train_loss_list,
        x_label='iterations',
        y_label='train loss',
        save_name='train_loss'
):
    # Normalize the loss values to be between 0 and 1 for plotting.
    min_loss = min(train_loss_list)
    max_loss = max(train_loss_list)

    # Handle the case where all loss values are the same to avoid division by zero.
    if max_loss - min_loss == 0:
        normalized_loss = [0.5] * len(train_loss_list)
    else:
        normalized_loss = [
            (loss - min_loss) / (max_loss - min_loss)
            for loss in train_loss_list
        ]

    width, height = 800, 600
    plot_img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(plot_img)

    step = width / len(normalized_loss)
    for i in range(1, len(normalized_loss)):
        x1 = int((i - 1) * step)
        y1 = height - int(normalized_loss[i - 1] * height)
        x2 = int(i * step)
        y2 = height - int(normalized_loss[i] * height)
        draw.line((x1, y1, x2, y2), fill="blue", width=2)

    draw.text((10, 10), x_label, fill="black")
    draw.text((10, height - 20), y_label, fill="black")

    plot_img.save(os.path.join(out_dir, f"{save_name}.png"))
    print('SAVING PLOTS COMPLETE...')


# original save_mAP()
# def save_mAP(OUT_DIR, map_05, map):
#     """
#     Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.
#     :param OUT_DIR: Path to save the graphs.
#     :param map_05: List containing mAP values at 0.5 IoU.
#     :param map: List containing mAP values at 0.5:0.95 IoU.
#     """
#     figure = plt.figure(figsize=(10, 7), num=1, clear=True)
#     ax = figure.add_subplot()
#     ax.plot(
#         map_05, color='tab:orange', linestyle='-',
#         label='mAP@0.5'
#     )
#     ax.plot(
#         map, color='tab:red', linestyle='-',
#         label='mAP@0.5:0.95'
#     )
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('mAP')
#     ax.legend()
#     figure.savefig(f"{OUT_DIR}/map.png")
def save_mAP(OUT_DIR, map_05, map):
    from PIL import Image, ImageDraw

    width, height = 800, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    max_map = max(map + map_05)
    min_map = min(map + map_05)
    normalized_map = [
        height - ((val - min_map) / (max_map - min_map) * height)
        for val in map
    ]
    normalized_map_05 = [
        height - ((val - min_map) / (max_map - min_map) * height)
        for val in map_05
    ]

    step = width / len(map)
    for i in range(1, len(normalized_map)):
        draw.line(
            [(step * (i - 1), normalized_map[i - 1]),
             (step * i, normalized_map[i])],
            fill="red", width=2
        )
        draw.line(
            [(step * (i - 1), normalized_map_05[i - 1]),
             (step * i, normalized_map_05[i])],
            fill="orange", width=2
        )

    draw.text((10, 10), 'Epochs', fill="black")
    draw.text((10, height - 20), 'mAP', fill="black")
    draw.text((10, 30), 'mAP@0.5', fill="orange")
    draw.text((10, 50), 'mAP@0.5:0.95', fill="red")

    image.save(f"{OUT_DIR}/map.png")
    print('SAVING PLOTS COMPLETE...')
