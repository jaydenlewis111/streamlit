from ultralytics import YOLO
import streamlit as st
import torch
from pathlib import Path

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    """
    try:
        if not isinstance(model_path, (str, Path)):
            raise ValueError("model_path must be a string or Path object")
            
        model_path = str(model_path)  # Convert Path to string
        
        # Download the model if it doesn't exist locally
        if not Path(model_path).exists():
            st.info(f"Downloading model {Path(model_path).name}...")
            try:
                model = YOLO(model_path)  # This will auto-download from ultralytics
                st.success("Model downloaded successfully!")
                return model
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                return None
        
        # Load existing model
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading local model: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in load_model: {str(e)}")
        return None

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

