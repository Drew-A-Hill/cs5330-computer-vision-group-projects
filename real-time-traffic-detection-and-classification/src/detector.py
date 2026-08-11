import configs

from ultralytics import YOLO

class Detection:
    def __init__(self, track_id, cls_id, class_name, x1, y1, x2, y2, box_h, cx, cy):
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

class Detector:
    def __init__(self, weights="yolov8n.pt"):
        self.model = YOLO(weights)

    def track(self, frame):
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
                    Detection(track_id, cls_id, class_name, x1, y1, x2, y2, box_h, cx, cy)
                )

        return detections
