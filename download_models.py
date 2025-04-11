import torch
from torch import serialization
from ultralytics import YOLO
import os
from pathlib import Path

def download_models():
    # Add safe globals for model loading
    serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
    
    # Get current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Define model paths
    detection_model_path = current_dir / 'yolov8n.pt'
    segmentation_model_path = current_dir / 'yolov8n-seg.pt'
    
    # Download detection model
    if not detection_model_path.exists():
        print("Downloading detection model...")
        model = YOLO('yolov8n.pt')
        model.save(detection_model_path)
        print("Detection model downloaded successfully!")
    
    # Download segmentation model
    if not segmentation_model_path.exists():
        print("Downloading segmentation model...")
        model = YOLO('yolov8n-seg.pt')
        model.save(segmentation_model_path)
        print("Segmentation model downloaded successfully!")

if __name__ == "__main__":
    download_models()