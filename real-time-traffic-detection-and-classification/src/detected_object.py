"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    detected_object.py

    This is a container that holds all of the information about a detected
    object in the frame.
"""
class DetectedObject:
    """
        Stores the data for a single detection.
    """
    def __init__(self, track_id, cls_id, class_name, x1, y1, x2, y2, box_h, cx, cy):
        """
            track_id: int - The trackers id for this object.
            cls_id: int - The YOLO class id number.
            class_name: str - The class name.
            x1: int - Left edge of the cropped box.
            y1: int - Top edge of the cropped box.
            x2: int - Right edge of the cropped box.
            y2: int - Bottom edge of the cropped box.
            box_h: int - Height of the cropped box in pixels.
            cx: int - X of the box center.
            cy: int - Y of the box center.
        """
        self.track_id = track_id
        self.cls_id = cls_id
        self.class_name = class_name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.box_h = box_h
        self.cx = cx
        self.cy = cy
