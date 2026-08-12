"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

detected_object.py

"""
class DetectedObject:
    """


    """
    def __init__(self, track_id, cls_id, class_name, x1, y1, x2, y2, box_h, cx, cy):
        """


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
