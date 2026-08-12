"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

configs.py

"""
CLASS_IDS = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck",
    16: "dog",
}

# Minimum bbox height to be considered.
MIN_BOX_HEIGHT_PX = 20
MAX_BOX_HEIGHT_PX = 200

# Minimum total centroid displacement. Used to identify stationary objects
STATIONARY_WINDOW = 15
STATIONARY_DISPLACEMENT_PX = 12

# Distance from the frame edge at which a crossing object is counted.
EDGE_MARGIN_PX = 40

SAMPLE_EVERY = 5
ATTRIBUTE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Coarse YOLO classes refined into finer types by the ResNet classifier.
REFINE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# Overlap (of the smaller box) above which a pedestrian is treated as a rider.
RIDER_OVERLAP = 0.4

# Capture retry and connection check.
MAX_CONSECUTIVE_FAILS = 20
RETRY_SLEEP_S = 0.2
CONNECTION_TIMEOUT_S = 10.0

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

FIELDS = [
    "timestamp",
    "street",
    "cross_street",
    "direction",
    "class",
    "color"
    ]