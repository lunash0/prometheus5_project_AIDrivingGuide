from config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_EPOCHS, 
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS,
    TRAIN_IMG,
    TRAIN_ANNOT,
    VALID_IMG,
    VALID_ANNOT,
    CLASSES,
    RESIZE_TO,
    BATCH_SIZE
)
from model import create_model
from custom_utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP
)
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR

import torch
# import matplotlib.pyplot as plt
import time
import os
import numpy as np
import random

import wandb

# plt.style.use('ggplot')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

wandb.init(project="traffic_lights_detection", entity="camorineon")  ###################################################


# Function for running training iterations.
def train(train_data_loader, model):
    print('Training')
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(
        train_data_loader, 
        total=len(train_data_loader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        wandb.log({"train_loss": loss_value})

    return loss_value


# Function for running validation iterations.
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    # Initialize tqdm progress bar.
    prog_bar = tqdm(
        valid_data_loader, 
        total=len(valid_data_loader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for k in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[k]['boxes'].detach()
            true_dict['labels'] = targets[k]['labels'].detach()
            preds_dict['boxes'] = outputs[k]['boxes'].detach()
            preds_dict['scores'] = outputs[k]['scores'].detach()
            preds_dict['labels'] = outputs[k]['labels'].detach()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric.reset()
    metric.update(preds, target)
    metric_summary = metric.compute()

    wandb.log({
        "mAP@0.50:0.95": metric_summary['map'],
        "mAP@0.50": metric_summary['map_50']
    })

    return metric_summary


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    train_dataset = create_train_dataset(
        TRAIN_IMG, TRAIN_ANNOT, CLASSES, RESIZE_TO,
    )
    valid_dataset = create_valid_dataset(
        VALID_IMG, VALID_ANNOT, CLASSES, RESIZE_TO
    )
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=50, gamma=0.1, verbose=True
    )

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Whether to show transformed images from data loader or not.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)

    # To save best model.
    save_best_model = SaveBestModel()

    metric = MeanAveragePrecision()

    # Training loop.
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # Reset the training loss histories for the current epoch.
        train_loss_hist.reset()

        # Start timer and carry out training and validation.
        start = time.time()
        train_loss = train(train_loader, model)
        metric_summary = validate(valid_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
        print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss_hist.value,
            "mAP@0.50:0.95": metric_summary['map'],
            "mAP@0.50": metric_summary['map_50'],
            "epoch_duration_minutes": (end - start) / 60
        })

        # # save the best model till now.
        # save_best_model(
        #     model, float(metric_summary['map']), epoch, 'outputs'
        # )
        # # Save the current epoch model.
        # save_model(epoch, model, optimizer)

        # Save the model every 10 epochs and at the last epoch.
        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_model(epoch, model, optimizer)

        # Save the best model till now.
        save_best_model(
            model, float(metric_summary['map']), epoch, 'outputs'
        )

        # Save loss plot.
        save_loss_plot(OUT_DIR, train_loss_list)

        # Save mAP plot.
        save_mAP(OUT_DIR, map_50_list, map_list)
        scheduler.step()
