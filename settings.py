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

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# ML Model config
DETECTION_MODEL = ROOT / 'yolov8n.pt'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

SEGMENTATION_MODEL = ROOT / 'yolov8n-seg.pt'