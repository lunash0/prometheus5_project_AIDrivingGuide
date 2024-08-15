from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'

# Files for Tasks
FILES_DIR = ROOT / 'files'
BOUNDING_BOX_FILE = FILES_DIR / 'bounding_box_model.yaml'
DRIVING_GUIDE_FILE = FILES_DIR / 'driving_guide_model.yaml'

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'default_image.jpg'
DEFAULT_OUTPUT_IMAGE = IMAGES_DIR / 'default_output_image.jpg'
OUTPUT_IMAGE_PATH = IMAGES_DIR / 'processed_image.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
UPLOADED_VIDEO_PATH = VIDEO_DIR / 'uploaded_video.mp4'
OUTPUT_VIDEO_PATH = VIDEO_DIR / 'processed_video.mp4'
