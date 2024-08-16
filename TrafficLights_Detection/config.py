import torch


BATCH_SIZE = 8  # Increase / decrease according to GPU memeory.
RESIZE_TO = 640  # Resize the image for training and transforms.
NUM_EPOCHS = 10  # Number of epochs to train for.
NUM_WORKERS = 4  # Number of parallel workers for data loading.

DEVICE = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')

# Training images and JSON files directory.
TRAIN_IMG = '/media/hail09/HDD/AIhub_costomed_dataset/training/images'
TRAIN_ANNOT = '/media/hail09/HDD/AIhub_costomed_dataset/training/xml_labels'
# Validation images and JSON files directory.
VALID_IMG = '/media/hail09/HDD/AIhub_costomed_dataset/validation/images'
VALID_ANNOT = '/media/hail09/HDD/AIhub_costomed_dataset/validation/xml_labels'

CLASSES = [
    'green', 'red', 'yellow', 'red and green arrow', 'red and yellow', 'green and green arrow', 'green and yellow',
    'yellow and green arrow', 'green arrow and green arrow', 'red cross', 'green arrow(down)', 'green arrow', 'etc'
]

NUM_CLASSES = len(CLASSES)

# Whether to visuaylize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'

# RGB format.
COLORS = [
    [0, 0, 0],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [255, 255, 255]
]
