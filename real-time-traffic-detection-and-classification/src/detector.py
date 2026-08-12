"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

detector.py

"""
import configs

from ultralytics import YOLO

from detected_object import DetectedObject

class Detector:
    """


    """
    def __init__(self, weights="weights/yolov8n.pt"):
        """


        """
        self.model = YOLO(weights)

    def track(self, frame):
        """


        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=.35,
            classes=list(configs.CLASS_IDS.keys()),
            verbose=False,
        )[0]

        detections = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                x1, y1, x2, y2 = box
                box_h = y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                class_name = configs.CLASS_IDS.get(cls_id, "unknown")
                detections.append(
                    DetectedObject(track_id, cls_id, class_name, x1, y1, x2, y2, box_h, cx, cy)
                )

        return self.drop_riders(detections)

    def overlap_check(self, a, b):
        """


        """
        left = max(a.x1, b.x1)
        top = max(a.y1, b.y1)
        right = min(a.x2, b.x2)
        bottom = min(a.y2, b.y2)
         
        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        return intersection / min(area_a, area_b)

    def drop_riders(self, detections):
        """


        """
        vehicles = [d for d in detections if d.cls_id in (1, 3)]
        kept = []
        for d in detections:
            if d.cls_id == 0 and any(self.overlap_check(d, v) >= configs.RIDER_OVERLAP for v in vehicles):
                continue
            kept.append(d)
        return kept
