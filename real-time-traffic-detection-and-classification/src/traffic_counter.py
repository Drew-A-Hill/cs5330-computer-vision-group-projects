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
import time
from pathlib import Path
import configs
from collections import defaultdict

from datetime import datetime

import cv2

from tracker import Tracker
from detector import Detector
from filters import is_too_far, is_stationary, direction_label, matches_direction
from logger import Logger
import overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--street", required=True, help="Name of street")
    parser.add_argument("--facing", required=True, help="Direction camera is facing, N, NE, E, SE, S, S, SW, W, NW ")
    parser.add_argument("--cross", required=False, help="Name of cross street if any")
    parser.add_argument("--direction", required=False, default="both")
    args = parser.parse_args()

    source = "http://" + args.source + ":8080/video"

    path = Path(__file__).resolve()
    root = path.parents[1]

    file_name = args.street
    if args.cross:
        file_name += "_" + args.cross

    file_name += "_" + args.facing

    log_file_path = root / f"{file_name}.csv"

    detector = Detector("yolov8n.pt")

    # Persistent state across frames.
    tracks = defaultdict(Tracker)
    counts = defaultdict(int)

    logger = Logger(log_file_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open stream: {args.source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    left_line = configs.EDGE_MARGIN_PX
    right_line = frame_w - configs.EDGE_MARGIN_PX

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream read failed to connect")
            return

        detections = detector.track(frame)

        for det in detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            box_h = det.box_h
            cx, cy = det.cx, det.cy
            class_name = det.class_name
            track_id = det.track_id

            # distance filter
            if is_too_far(box_h):
                continue

            state = tracks[track_id]
            state.update(cx, cy)

            # stationary filter
            if is_stationary(state):
                overlay.draw_filtered_box(frame, x1, y1, x2, y2)
                continue

            # direction filter
            traffic_dir_str = direction_label(state, args.facing)

            # counting line
            if not state.counted and matches_direction(state, args.direction) and state.crossed_exit(left_line, right_line):
                state.counted = True
                counts[class_name] += 1
                logger.write_row([datetime.now().isoformat(), track_id, class_name])

            overlay.draw_tracked_box(frame, x1, y1, x2, y2, class_name)

        # overlay: counting line + running totals
        overlay.draw_counts(frame, counts)

        window_name = "Traffic: " + file_name

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    print("Final counts:", dict(counts))

if __name__ == "__main__":
    main()
