from pathlib import Path
import sys
from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file (all/app/)
ROOT = FILE.parent 
print('ROOT is', ROOT) # /home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/app

# Add the root path (all/app/) to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Sources
IMAGE = 'Image'
VIDEO = 'Video'

# Files for Tasks
BOUNDING_BOX_FILE = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/main.py'
DRIVING_GUIDE_FILE = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/main.py'

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'raw/default_image_1.png'
DEFAULT_OUTPUT_IMAGE = IMAGES_DIR / 'default_output_image_1_all.png'
OUTPUT_IMAGE_PATH = IMAGES_DIR / 'processed_image.png'

# Videos config
VIDEO_DIR = ROOT / 'videos'
UPLOADED_VIDEO_PATH = VIDEO_DIR / 'uploaded_video.mp4'
OUTPUT_VIDEO_PATH = VIDEO_DIR / 'processed_video.mp4'

CFG_DIR = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml'
# # Get the absolute path of the current file
# FILE = Path(__file__).resolve()
# # Get the parent directory of the current file
# ROOT = FILE.parent
# # Add the root path to the sys.path list if it is not already there
# if ROOT not in sys.path:
#     sys.path.append(str(ROOT))
# # Get the relative path of the root directory with respect to the current working directory
# ROOT = ROOT.relative_to(Path.cwd())

# # Sources
# IMAGE = 'Image'
# VIDEO = 'Video'

# # Files for Tasks
# FILES_DIR = ROOT / 'files'
# BOUNDING_BOX_FILE = FILES_DIR / 'bounding_box_model.yaml'
# DRIVING_GUIDE_FILE = FILES_DIR / 'driving_guide_model.yaml'

# # Images config
# IMAGES_DIR = ROOT / 'images'
# DEFAULT_IMAGE = IMAGES_DIR / 'default_image.jpg'
# DEFAULT_OUTPUT_IMAGE = IMAGES_DIR / 'default_output_image.jpg'
# OUTPUT_IMAGE_PATH = IMAGES_DIR / 'processed_image.jpg'

# # Videos config
# VIDEO_DIR = ROOT / 'videos'
# UPLOADED_VIDEO_PATH = VIDEO_DIR / 'uploaded_video.mp4'
# OUTPUT_VIDEO_PATH = VIDEO_DIR / 'processed_video.mp4'