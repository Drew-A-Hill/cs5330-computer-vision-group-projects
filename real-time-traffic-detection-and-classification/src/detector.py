"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    detector.py

    This file is used as a wrapper for the YOLOv8 detector and ByteTrack  
    tracker. It runs detection and tracking on a frame and returns a list of 
    DetectedObjects.
"""
import configs
from ultralytics import YOLO
from detected_object import DetectedObject

class Detector:
    """
        Runs YOLO detection with tracking on each frame.
    """
    def __init__(self, weights):
        """
            weights: str - Path to the YOLO model weights file.
        """
        # Loads the YOLO model.
        self.model = YOLO(weights)

    def track(self, frame):
        """
            Detects and tracts objects in a single frame.

            frame: cv2.Mat - This is the frame captured.

            returns a list of DetectedObjects found in the frame.
        """
        # Runs YOLO with ByteTrack tracking, only on the classes we care about.
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=.35,
            classes=list(configs.CLASS_IDS.keys()),
            verbose=False,
        )[0]

        detections = []

        # Only build detections if YOLO found boxes with track ids this frame.
        if results.boxes is not None and results.boxes.id is not None:

            # Gets the boxes, track ids, and class ids as numpy arrays.
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Builds ta DetectedObject for each detection.
            for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                x1, y1, x2, y2 = box
                box_h = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                class_name = configs.CLASS_IDS.get(cls_id, "unknown")
                detections.append(
                    DetectedObject(track_id, cls_id, class_name, x1, y1, x2, y2, box_h, cx, cy)
                )

        # Drops pedestrians that are riding bikes or motorcycles before returning.
        return self.drop_riders(detections)

    def overlap_check(self, a, b):
        """
            Measures how much two boxes overlap.

            a: DetectedObject - The first object.
            b: DetectedObject - The second object.

            returns a float for percentage of overlap. 0.0 is no overlap
            1.0 is full overlap.
        """
        # Finds the overlapping rectangle.
        left = max(a.x1, b.x1)
        top = max(a.y1, b.y1)
        right = min(a.x2, b.x2)
        bottom = min(a.y2, b.y2)
         
        # Returns if no overlap.
        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        return intersection / min(area_a, area_b)

    def drop_riders(self, detections):
        """
            Removes a pedestrian detecton when it overlaps with a bicycle or 
            a motorcucle. This is so that a person riding a bike or motorcycle 
            is not counted for both motorcycle or bike and pedestrian.

            detections: list of DetectedObject - All detections this frame.

            returns list of DetectedObject with rider pedestrians removed.
        """
        vehicles = [d for d in detections if d.cls_id in (1, 3)]
        kept = []

        for d in detections:
            # Skips pedestrians that overlap with bycicle or motorcycle.
            if d.cls_id == 0 and any(self.overlap_check(d, v) >= configs.RIDER_OVERLAP for v in vehicles):
                continue

            kept.append(d)

        return kept
