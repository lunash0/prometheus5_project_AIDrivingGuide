import torch


BATCH_SIZE = 8  # Increase / decrease according to GPU memeory.
RESIZE_TO = 640  # Resize the image for training and transforms.
NUM_EPOCHS = 75  # Number of epochs to train for.
NUM_WORKERS = 4  # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_IMG = '/media/hail09/HDD/S2TLD_720x1280/normal_2/JPEGImages'
TRAIN_ANNOT = '/media/hail09/HDD/S2TLD_720x1280/normal_2/Annotations'
# Validation images and XML files directory.
VALID_IMG = '/media/hail09/HDD/S2TLD_720x1280/normal_1/JPEGImages'
VALID_ANNOT = '/media/hail09/HDD/S2TLD_720x1280/normal_1/Annotations'

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'red', 'yellow', 'green', 'off'
]

NUM_CLASSES = len(CLASSES)

# Whether to visuaylize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'
