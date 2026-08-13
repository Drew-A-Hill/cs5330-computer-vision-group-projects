"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    configs.py

    Holds all of the tunable settings and shared constants.
"""
# Fraction of the screen size to display the window in.
WINDOW_RATIO = 0.95

# YOLO class ids we detect and their names.
CLASS_IDS = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Minimum box height considered. Too smaller is far and too large is too close.
MIN_BOX_HEIGHT_PX = 20
MAX_BOX_HEIGHT_PX = 200

# Used to identify stationary objects
STATIONARY_DISPLACEMENT_PX = 12

# Only samples an objects color every nth frame.
SAMPLE_EVERY = 5

# Classes we detect color for.
ATTRIBUTE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# YOLO classes refined into labels by the ResNet classifier.
REFINE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# ImageNet class index with finer label used by the ResNet classifier.
TARGET_BY_INDEX = {
    407: "ambulance",
    555: "fire truck",
    569: "truck",
    734: "police car",
    656: "van",
    675: "truck",
    717: "pickup truck",
    609: "suv",
    436: "suv",
    468: "car",
    511: "car",
    627: "car",
    817: "car",
    654: "bus",
    779: "bus",
    874: "bus",
    864: "truck",
    867: "truck",
}

# Overlap above which a pedestrian is treated as a rider of bike or motorcycle.
RIDER_OVERLAP = 0.4

# Setting for handling the video stream.
MAX_CONSECUTIVE_FAILS = 20
RETRY_SLEEP_S = 0.2
CONNECTION_TIMEOUT_S = 10.0

# Maps the direction object is heading relative to the direction the camera is facing.
DIRECTIONS = {
        "N": ["W", "E"], 
        "NE": ["NW", "SE"], 
        "E": ["N", "S"], 
        "SE": ["NE", "SW"], 
        "S": ["E", "W"], 
        "SW": ["SE", "NW"], 
        "W": ["S", "N"], 
        "NW": ["SW", "NE"]
        }
# The columns written to the csv.
FIELDS = [
    "timestamp",
    "street",
    "cross_street",
    "direction",
    "class",
    "color"
    ]