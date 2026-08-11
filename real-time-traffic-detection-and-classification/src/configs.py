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
MIN_BOX_HEIGHT_PX = 40
MAX_BOX_HEIGHT_PX = 200

# Minimum total centroid displacement. Used to identify stationary objects
STATIONARY_WINDOW = 15
STATIONARY_DISPLACEMENT_PX = 12

# Distance from the frame edge at which a crossing object is counted.
EDGE_MARGIN_PX = 40

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