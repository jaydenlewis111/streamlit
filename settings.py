from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent.absolute()  # Changed to absolute

# Add the root path to the sys.path list if it is not already there
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# ML Model config
DETECTION_MODEL = ROOT / 'yolov8n.pt'
SEGMENTATION_MODEL = ROOT / 'yolov8n-seg.pt'