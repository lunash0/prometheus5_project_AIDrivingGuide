from model import build_model
from utils import * 
from engine import get_log, default_argument_parser, setup 
from tqdm.auto import tqdm 
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
import torch
import time
from torchvision.ops import box_iou 
import wandb 
from torch import nn as nn 

def train(device, model, train_loader, optimizer, scheduler):
    train_losses = []
    running_loss = .0
    progress_bar = tqdm(train_loader, total=len(train_loader))

    for i, data in enumerate(progress_bar):
        images, targets, _ = data

        images = [img.to(device) for img in images] 
                
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  

        optimizer.step()

        loss_value = losses.item()
        running_loss += loss_value
        train_losses.append(loss_value)
                
        progress_bar.set_description(desc=f"Training Loss: {loss_value:.4f}")

    avg_loss = running_loss / len(train_loader)

    if scheduler is not None:
        scheduler.step(avg_loss)

    return train_losses, avg_loss

def valid(device, model, valid_loader, iou_thresh=0.4, confidence_threshold=0.4, cfg_dir=''):
    progress_bar = tqdm(valid_loader, total=len(valid_loader))

    iou_scores = []

    for i, data in enumerate(progress_bar):
        images, targets, width, height, image_filename = data
        images = [img.to(device) for img in images]
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)

        # Apply NMS and Confidence Score Threshold
        outputs = [apply_nms(output, iou_thresh) for output in outputs]
        outputs = [filter_boxes_by_score(output, confidence_threshold) for output in outputs]

        for idx, output in enumerate(outputs):
            if len(output['boxes']) > 0: # pass if it predicted as blank
                pred_boxes = []
                pred_boxes_norm = []
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                gt_boxes = targets[idx]['boxes'].cpu() 
                gt_labels = targets[idx]['labels'].cpu() 
                
                for box, label, score in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = box
                        x1_norm, y1_norm, x2_norm, y2_norm = box_denormalize(x1, y1, x2, y2, width[idx], height[idx], cfg_dir)
                        pred_boxes.append([x1, y1, x2, y2])
                        pred_boxes_norm.append([x1_norm, y1_norm, x2_norm, y2_norm])

                pred_boxes = torch.tensor(pred_boxes)
                ious = box_iou(pred_boxes, gt_boxes)

                if ious.numel() > 0:
                    mean_iou = ious.max(dim=1)[0].mean().item()
                else:
                    mean_iou = 0.0

                iou_scores.append(mean_iou)

                # For sanity check
                def find_root(root, find):
                  import os
                  for dir_path, _, filenames in (os.walk(root)):
                    for filename in filenames:
                      if find in filename:
                        full_path = os.path.join(dir_path, filename)
                        return full_path
                      
                progress_bar.set_description(desc=f"Validation IoU: {mean_iou:.4f}")

    final_avg_iou = sum(iou_scores) / len(iou_scores) if len(iou_scores) > 0 else 0.0
    print(f"> Final Average IoU : {final_avg_iou} = {sum(iou_scores)} / {len(iou_scores)}")
    return iou_scores, final_avg_iou


def main(args):
      cfg_dir, output_dir, run_name = setup(args)

      cfg = load_yaml(cfg_dir)['train']
      device = torch.device(f'cuda:{cfg["device"]}' if torch.cuda.is_available() else 'cpu')

      if len(run_name) == 0:
          run_name = os.path.basename(cfg_dir).split('.yaml')[0]

      get_log(run_name, load_yaml(cfg_dir)['log']) # Initialize wandb

      best_valid_score = -1 
      batch_size = cfg['batch_size']
      epochs = cfg['epochs']
      learning_rate = cfg['learning_rate']
      scheduler = cfg['scheduler']
      use_earlystop = cfg['early_stop']
      opt = cfg['optimizer']

      output_dir = os.path.join(output_dir, cfg_dir.split('/')[-1].split('.yaml')[0])
      os.makedirs(output_dir, exist_ok=True)
          
      print('Start training for Pedestrian Detection ...')
      print(f'Using device: {device} | Batch size: {batch_size} | Epochs: {epochs} | Learning rate: {learning_rate} | Optimizer: {opt} | Scheduler: {scheduler} | Early Stop: {use_earlystop}')

      train_dataset = create_train_dataset(cfg_dir)
      valid_dataset = create_valid_dataset(cfg_dir)
      train_loader = create_train_loader(train_dataset, batch_size)
      valid_loader = create_valid_loader(valid_dataset, batch_size)

      print(f"# of training samples : {len(train_dataset)}")
      print(f'# of validation samples : {len(valid_dataset)}')   

      model = build_model(cfg['num_classes']).to(device)    
      wandb.watch(model) 
      
      params = [p for p in model.parameters() if p.requires_grad]

      if opt == 'Adam':
          optimizer = torch.optim.Adam(params, lr=learning_rate)
      elif opt == 'SGD':
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)
      else:
          optimizer = None 

      if scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
      elif scheduler == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
      else:
        lr_scheduler = None
      
      if use_earlystop:
          early_stop_patience = cfg['early_stop_patience']
          early_stop_counter = 0

      for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}...")
            start = time.time()
            
            model.train()
            train_loss, train_tot_loss = train(device, model, train_loader, optimizer, lr_scheduler)
            model.eval()
            iou_scores, final_avg_iou = valid(device, model, valid_loader, cfg['iou_threshold'], cfg['confidence_threshold'], cfg_dir)
            
            print(f"Epoch [{epoch+1}] train loss: {train_tot_loss:.3f}")   
            print(f"Epoch [{epoch+1}] validation IoU score: {final_avg_iou:.3f}")   
            
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch + 1}")

            wandb.log({
                "train_total_loss": train_tot_loss,
                "valid_total_avg_iou": final_avg_iou,
            })

            if best_valid_score < final_avg_iou and final_avg_iou > 0.6:
                  best_valid_score = final_avg_iou 

                  print(f"\nBest validation score(IoU): {final_avg_iou}")
                  print(f"\nSaving best model for epoch: {epoch+1}\n")
                  save_model = f'best_{cfg["model_name"]}_e{epoch + 1}s{final_avg_iou:.2f}l{train_tot_loss:.2f}.pth'
                  checkpoint_path = os.path.join(output_dir, save_model)

                  torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)

                  wandb.save(checkpoint_path)
                  early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                  print("[INFO] Early stopping triggered!")
                  exit(1)
            time.sleep(2)
      wandb.finish()

if __name__ == "__main__":
    args = default_argument_parser()
    print("Command Line Args:", args)
    main(args)
