import numpy as np
import cv2
import os
import json
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Dataset
from utils import collate_fn, train_transform, valid_transform, normalize_bbox
from collections import defaultdict
from tqdm import tqdm 
import warnings
warnings.filterwarnings(action='ignore')

class PedestrianDataset(Dataset):
    def __init__(self, root: str, train: bool, split: str = "train", transforms=None, image_size=[640, 360]):
        super().__init__()
        
        self.root = root
        self.train = train
        self.transforms = transforms
        self.image_size = image_size 
        
        annot_path = os.path.join(root, f'{split.lower()}_annotations.json') 

        with open(annot_path) as f:
            raw_annots = json.load(f)["annotations"]

        with open(annot_path) as f:
            images_info = json.load(f)["images"]

        img_annots = defaultdict(list)
        for ann in raw_annots:
            img_annots[ann['image_id']].append(ann)
        
        def collect_image_paths(root):
          image_paths = []
          image_list = []
          
          # Traverse the root directory and its subdirectories
          for dirpath, _, filenames in (os.walk(root)):
              for filename in filenames:
                  if filename.endswith(".jpg") or filename.endswith(".png"):
                      full_path = os.path.join(dirpath, filename)
                      image_paths.append(full_path)
                      image_list.append(filename)
          
          return image_paths, image_list
        
        root = os.path.join(root, 'data') # Hard Coded
        image_paths1, image_list1 = collect_image_paths(root)
        
        image_list1_set = set(image_list1)
        filtered_image_list = []
        images_info_dict = {}

        for image in images_info:
            if image['id'] in img_annots.keys():
                filtered_image_list.append(image['file_name'])
                images_info_dict[image['file_name']] = image['id']

        image_paths = [] 

        for fname in tqdm(filtered_image_list):
          if fname in image_list1_set:
              idx = image_list1.index(fname)
              image_paths.append(image_paths1[idx])

        print(f'img_annots len: {len(img_annots)}')
        print(f'image_paths len: {len(image_paths)}')

        assert len(img_annots) == len(image_paths), "Number of images and labels does not match"

        self.local_image_list = filtered_image_list
        self.local_images = image_paths
        self.local_annotations = img_annots
        self.images_info_dict = images_info_dict

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        image_path = self.local_images[idx]
        image_filename = os.path.basename(image_path)
        anns = self.local_annotations[self.images_info_dict[image_filename]]

        image = cv2.imread(image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        WIDTH, HEIGHT = self.image_size[0], self.image_size[1]
        image_resized = cv2.resize(image, (WIDTH, HEIGHT)) 

        image_resized /= 255.0
        image_resized = image_resized.copy() 

        width, height = image.shape[1], image.shape[0] # original shape of image

        boxes = []
        labels = []
        boxes_or = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            x1 = int(x1)
            y1 = int(y1)
            x2 = x1 + int(w)
            y2 = y1 + int(h)

            label = ann['category_id']
            labels.append(label)

            x1_r = (x1 / width) * WIDTH
            y1_r = (y1 / height) * HEIGHT
            x2_r = (x2 / width) * WIDTH
            y2_r = (y2 / height) * HEIGHT
            boxes_or.append([x1, y1, x2, y2])
            boxes.append([x1_r, y1_r, x2_r, y2_r])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = anns[0]['image_id']

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels # -> so label_fields are also labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['image_id'] = torch.tensor([image_id])
        boxes = normalize_bbox(target['boxes'], width, height)

        if self.train:
            if self.transforms is not None:
                transformed = self.transforms(image=image_resized, bboxes=boxes, labels=labels)
                _image = transformed['image'] 
                _bboxes = transformed['bboxes']

                valid_boxes = []
                for box in _bboxes:
                    xmin, ymin, xmax, ymax = box
                    if xmax > xmin and ymax > ymin:
                        valid_boxes.append([xmin, ymin, xmax, ymax])
                image = torch.Tensor(_image)
                target['bboxes'] = torch.Tensor(valid_boxes)                

            return image, target, image_filename

        else:
            if self.transforms is not None:
                transformed = self.transforms(image=image_resized, bboxes=boxes, labels=labels)
                _image = transformed['image'] 
                _bboxes = transformed['bboxes']

                valid_boxes = []
                for box in _bboxes:
                    xmin, ymin, xmax, ymax = box
                    if xmax > xmin and ymax > ymin:
                        valid_boxes.append([xmin, ymin, xmax, ymax])
                image = torch.Tensor(_image)
                target['bboxes'] = torch.Tensor(valid_boxes) 

            return image, target, width, height, image_filename        

def create_train_dataset(cfg_dir):
    train_dataset = PedestrianDataset(root='/data/tmp/', train=True, split="train", transforms=train_transform(cfg_dir))
    return train_dataset

def create_valid_dataset(cfg_dir):
    val_dataset = PedestrianDataset(root='/data/tmp/', train= False, split="val", transforms=valid_transform(cfg_dir))
    return val_dataset

def create_train_loader(train_dataset, batch_size):
    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size,
                              shuffle= True,
                              collate_fn= collate_fn)
    return train_loader
def create_valid_loader(val_dataset, batch_size):
    valid_loader = DataLoader(val_dataset,
                              batch_size = batch_size,
                              shuffle= False,collate_fn= collate_fn)
    return valid_loader
