from pathlib import Path
import sys
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent 
# print('ROOT is', ROOT) # prometheus5_project_AIDrivingGuide/app

if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
DEFAULT = 'Home'

# Files for Tasks
TASK_FILE = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/main.py'

# Images config
IMAGES_DIR = ROOT / 'images'
UPLOADED_IMAGE_PATH = IMAGES_DIR / 'uploaded_image.png'
OUTPUT_IMAGE_PATH_1 = IMAGES_DIR / 'processed_image.png'
OUTPUT_IMAGE_PATH_2 = IMAGES_DIR / 'processed_image_2.png' # bbox version for '⚔️ Show All' case
DEFAULT_IMAGE = IMAGES_DIR / 'default_image_1.png'
DEFAULT_OUTPUT_IMAGE_1 = IMAGES_DIR / 'default_image_1_all.png' # all version
DEFAULT_OUTPUT_IMAGE_2 = IMAGES_DIR / 'default_image_1_message.png' # comments version

# Videos config
VIDEO_DIR = ROOT / 'videos'
UPLOADED_VIDEO_PATH = VIDEO_DIR / 'uploaded_video.mp4'
OUTPUT_VIDEO_PATH_1 = VIDEO_DIR / 'processed_video.mp4'
OUTPUT_VIDEO_PATH_2 = VIDEO_DIR / 'processed_video_2.mp4'
OUTPUT_VIDEO_PATH_1_AVI = VIDEO_DIR / 'processed_video.avi'  # Added for AVI output
OUTPUT_VIDEO_PATH_2_AVI = VIDEO_DIR / 'processed_video_2.avi'  # Added for AVI output

# Model config
CFG_DIR = '/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/configs/model.yaml'

# Description config
POSTER_URL = 'https://drive.google.com/file/d/1-fiCVhsU0hYbtdG4lsOg-xQNXFEaZ3hj/view?usp=sharing'
GITHUB = 'https://github.com/lunash0/prometheus5_project_AIDrivingGuide'
PROMETHEUS_URL = 'https://prometheus-ai.net/'
FEEDBACK_FILE_PATH = Path("./assets/feedback.json")