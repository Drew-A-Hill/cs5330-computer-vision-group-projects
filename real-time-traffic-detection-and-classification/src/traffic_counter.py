"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

Pulls an MJPEG stream, runs YOLOv8 detection + ByteTrack tracking,
filters out distant/stationary/wrong-direction objects, counts
objects crossing a virtual line, and logs timestamped counts by class.

Usage:
    python traffic_counter.py --source <phone-ip> 
"""

import argparse
import csv
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
from ultralytics import YOLO

from tracker import Tracker

# COCO class ids we care about (see model.names for the full list).
# If you fine-tune a custom model with an added electric_scooter class,
# add its id here and to CLASS_NAMES.
TARGET_CLASS_IDS = {
    0: "pedestrian", 
    1: "bicycle",
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck",
    16: "dog",
}

CONF_THRESHOLD = 0.35

# Distance filter: minimum bbox height (pixels) to be considered "close enough".
# Tune this by eye against your stream resolution -- objects far down the
# street will have short boxes and get filtered out here.
MIN_BOX_HEIGHT_PX = 40

# Motion filter: minimum total centroid displacement (pixels) over the
# trailing window below which a track is considered stationary (parked
# car, reflection, etc.) and dropped.
# frames of history to keep per track
STATIONARY_WINDOW = 15         

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--street", required=True, help="Name of street")
    parser.add_argument("--direction", required=True, help="Direction camera is facing, N, NE, NW, S, SE, SW ")
    parser.add_argument("--cross", required=False, help="Name of cross street if any")
    args = parser.parse_args()

    log_file_name = args.street
    source = "http://" + args.source + ":8080/video"

    log_file_path = args.street
    if args.cross:
        log_file_path += "_" + args.cross
    log_file_path += "_" + args.direction + ".csv"

    model = YOLO("yolov8n.pt")

    # Persistent state across frames.
    tracks = defaultdict(Tracker)
    counts = defaultdict(int)

    # CSV log: one row per counted object.
    log_file = open(log_file_path, "a", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["timestamp", "track_id", "class"])

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

                # stationary filter 
                if state.displacement() < STATIONARY_DISPLACEMENT_PX:

                    # Color is red if object is filtered out ie stationary or too far.
                    color = (0, 0, 255)  
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    continue

                # direction filter
                if state.direction() != TARGET_DIRECTION:
                    color = (0, 165, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    continue

                # counting line
                if not state.counted and state.crossed_line(line_x):
                    state.counted = True
                    counts[class_name] += 1
                    writer.writerow([datetime.now().isoformat(), track_id, class_name])
                    log_file.flush()

                color = (0, 255, 0)  
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # overlay: counting line + running totals
        y_off = 20
        for cname, c in counts.items():
            cv2.putText(frame, f"{cname}: {c}", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_off += 22

        window_name = "Traffic: " + log_file_path.removesuffix(".csv")

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print("Final counts:", dict(counts))

if __name__ == "__main__":
    main()