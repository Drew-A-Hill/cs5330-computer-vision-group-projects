"""
Real-Time Street Traffic Detection and Classification
CS5330 - Drew Hill & Abhiram Bandi

Pulls an MJPEG stream, runs YOLOv8 detection + ByteTrack tracking,
filters out distant/stationary/wrong-direction objects, counts
objects crossing a virtual line, and logs timestamped counts by class.

Usage:
    python traffic_counter.py --source http://<phone-ip>:8080/video
    python traffic_counter.py --source 0                # webcam fallback for testing
    python traffic_counter.py --source path/to/test.mp4 # recorded clip for testing
"""

import argparse
import csv
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG - tune these for your camera placement / stream
# ---------------------------------------------------------------------------

# COCO class ids we care about (see model.names for the full list).
# If you fine-tune a custom model with an added electric_scooter class,
# add its id here and to CLASS_NAMES.
TARGET_CLASS_IDS = {
    0: "pedestrian",   # COCO 'person'
    1: "bicycle",
    2: "car",
    3: "motorcycle",   # electric scooters get bucketed here unless you fine-tune
    5: "bus",
    7: "truck",
}

CONF_THRESHOLD = 0.35

# Distance filter: minimum bbox height (pixels) to be considered "close enough".
# Tune this by eye against your stream resolution -- objects far down the
# street will have short boxes and get filtered out here.
MIN_BOX_HEIGHT_PX = 40

# Motion filter: minimum total centroid displacement (pixels) over the
# trailing window below which a track is considered stationary (parked
# car, reflection, etc.) and dropped.
STATIONARY_WINDOW = 15          # frames of history to keep per track
STATIONARY_DISPLACEMENT_PX = 12

# Direction filter: only count tracks moving in this direction along the
# horizontal axis. "left_to_right" or "right_to_left".
TARGET_DIRECTION = "left_to_right"

# Counting line: a horizontal fraction of frame width (0-1) or an absolute
# vertical fraction -- pick whichever axis matches your camera orientation.
# Default here is a vertical line at 50% of frame width, i.e. objects are
# counted when their centroid crosses this x-coordinate.
COUNT_LINE_FRACTION = 0.5

LOG_PATH = "traffic_log.csv"

# ---------------------------------------------------------------------------


class TrackState:
    """Per-track history used for the motion / direction / counting filters."""

    def __init__(self):
        self.centroids = deque(maxlen=STATIONARY_WINDOW)
        self.counted = False

    def update(self, cx, cy):
        self.centroids.append((cx, cy))

    def displacement(self):
        if len(self.centroids) < 2:
            return 0.0
        x0, y0 = self.centroids[0]
        x1, y1 = self.centroids[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def direction(self):
        if len(self.centroids) < 2:
            return None
        x0, _ = self.centroids[0]
        x1, _ = self.centroids[-1]
        return "left_to_right" if x1 > x0 else "right_to_left"

    def crossed_line(self, line_x):
        """True if the track's centroid history straddles line_x this frame."""
        if len(self.centroids) < 2:
            return False
        (x_prev, _), (x_curr, _) = self.centroids[-2], self.centroids[-1]
        return (x_prev < line_x <= x_curr) or (x_prev > line_x >= x_curr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="MJPEG stream URL, video file, or webcam index")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Path to YOLO weights (swap in a fine-tuned .pt if you train one)")
    parser.add_argument("--display", action="store_true", default=True,
                         help="Show live annotated window")
    args = parser.parse_args()

    model = YOLO(args.model)

    # Persistent state across frames.
    tracks = defaultdict(TrackState)
    counts = defaultdict(int)

    # CSV log: one row per counted object.
    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["timestamp", "track_id", "class"])

    # ultralytics .track() handles reading the stream frame-by-frame when
    # given a stream source, but for MJPEG http streams it's more reliable
    # to pull frames yourself with cv2.VideoCapture and call .track() per
    # frame with persist=True so IDs carry over.
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open stream: {args.source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    line_x = int(frame_w * COUNT_LINE_FRACTION)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream read failed, retrying...")
            time.sleep(0.5)
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            classes=list(TARGET_CLASS_IDS.keys()),
            verbose=False,
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                x1, y1, x2, y2 = box
                box_h = y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                class_name = TARGET_CLASS_IDS.get(cls_id, "unknown")

                # --- distance filter ---
                if box_h < MIN_BOX_HEIGHT_PX:
                    continue

                state = tracks[track_id]
                state.update(cx, cy)

                # --- stationary filter ---
                if state.displacement() < STATIONARY_DISPLACEMENT_PX:
                    color = (0, 0, 255)  # red = filtered out (stationary/too far)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    continue

                # --- direction filter ---
                if state.direction() != TARGET_DIRECTION:
                    color = (0, 165, 255)  # orange = wrong direction
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    continue

                # --- counting line ---
                if not state.counted and state.crossed_line(line_x):
                    state.counted = True
                    counts[class_name] += 1
                    writer.writerow([datetime.now().isoformat(), track_id, class_name])
                    log_file.flush()

                color = (0, 255, 0)  # green = valid, tracked, (maybe) counted
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{class_name} #{track_id}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- overlay: counting line + running totals ---
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 0), 2)
        y_off = 20
        for cname, c in counts.items():
            cv2.putText(frame, f"{cname}: {c}", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_off += 22

        if args.display:
            cv2.imshow("Traffic Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print("Final counts:", dict(counts))


if __name__ == "__main__":
    main()