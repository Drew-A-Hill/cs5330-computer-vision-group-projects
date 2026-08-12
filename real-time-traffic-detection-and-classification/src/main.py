"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

main.py

Pulls an MJPEG stream, runs YOLOv8 detection + ByteTrack tracking,
filters out distant/stationary/wrong-direction objects, counts
objects crossing a virtual line, and logs timestamped counts by class.

Usage:
    python main.py --source <phone-ip>
"""
import argparse
import time
import tkinter
from pathlib import Path
import configs
from collections import defaultdict, deque
import os

from datetime import datetime

import cv2

from tracker import Tracker
from detector import Detector
from filters import is_too_far, is_stationary, direction_label, matches_direction
from logger import Logger
from color import detect_color
from classifier import Classifier, crop_object
from video_source import get_source, is_network_source, is_file_source, check_connection
import overlay

def main():
    """
        

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--street", required=True, help="Name of street")
    parser.add_argument("--facing", required=True, help="Direction camera is facing ie N, NE, E, SE, S, S, SW, W, NW ")
    parser.add_argument("--cross", required=False, default="", help="Name of cross street if any")
    parser.add_argument("--direction", required=False, default="both")
    args = parser.parse_args()

    source = get_source(args.source)
    file_source = is_file_source(source)

    if is_network_source(source):
        ok, msg = check_connection(source)
        if not ok:
            print(msg)
            return

    path = Path(__file__).resolve()
    root = path.parents[1]

    file_name = args.street
    if args.cross:
        file_name += "&" + args.cross

    file_name += "_" + args.facing

    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    log_file_path = data_dir / f"{file_name}.csv"

    detector = Detector("weights/yolov8n.pt")
    classifier = Classifier()

    # Persistent state across frames.
    tracks = defaultdict(Tracker)
    counts = defaultdict(lambda: defaultdict(int))
    detection_log = deque(maxlen=50)

    # Start every class at 0 for both travel directions.
    all_classes = sorted(set(configs.CLASS_IDS.values()) | set(configs.TARGET_BY_INDEX.values()))
    for count_direction in configs.DIRECTIONS[args.facing]:
        for count_class in all_classes:
            counts[count_direction][count_class] += 0

    if is_network_source(source):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    writer = None

    logger = Logger(log_file_path)

    rotate = 0
    write = False
    frame_idx = 0
    fails = 0
    got_frame = False
    show_totals = True
    show_log = True
    show_line = 0
    start = time.time()

    street_name, street_type = args.street.split("_")
    
    location = f"{street_name} {street_type}"
    if args.cross:
        cross_name, cross_type = args.cross.split("_")
        location += f" & {cross_name} {cross_type}"

    window_name = "Detecting Traffic: " + location

    try:
        _tk = tkinter.Tk()
        screen_w, screen_h = _tk.winfo_screenwidth(), _tk.winfo_screenheight()
        _tk.destroy()

    except Exception:
        screen_w, screen_h = 1920, 1080

    target_w, target_h = int(screen_w * configs.WINDOW_RATIO), int(screen_h * configs.WINDOW_RATIO)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if file_source:
                    break
                fails += 1
                if fails > configs.MAX_CONSECUTIVE_FAILS:
                    if got_frame:
                        print(f"Stream lost after {fails} consecutive failed reads: {source}")
                    else:
                        print(f"Opened {source} but received no frames - the stream is not "
                              f"delivering data (isOpened was True but reads keep failing).")
                    break
                time.sleep(configs.RETRY_SLEEP_S)
                continue

            fails = 0
            got_frame = True
            frame_idx += 1

            elapsed = int(time.time() - start)
            elapsed_str = f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}"

            for each in range(rotate):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            line_x = frame.shape[1] // 2

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
                    overlay.draw_filtered_box(frame, x1, y1, x2, y2, class_name, track_id)
                    continue

                # direction filter
                traffic_dir_str = direction_label(state, args.facing)

                if class_name in configs.ATTRIBUTE_CLASSES:
                    if frame_idx % configs.SAMPLE_EVERY == 0 and box_h > state.best_h:
                        state.best_h = box_h
                        state.add_color(detect_color(frame, x1, y1, x2, y2))

                # counting line
                if not state.counted and matches_direction(state, args.direction) and state.crossed_line(line_x):
                    state.counted = True
                    is_vehicle = class_name in configs.ATTRIBUTE_CLASSES
                    if class_name in configs.REFINE_CLASSES:
                        label, conf = classifier.predict(crop_object(frame, x1, y1, x2, y2))
                        if label:
                            class_name = label
                    direction_key = traffic_dir_str if traffic_dir_str else "Unknown"
                    counts[direction_key][class_name] += 1
                    os.system('clear')
                    if traffic_dir_str:
                        print(f"{class_name}: {traffic_dir_str}")

                    else:
                        print(f"{class_name}: Unknown direction")

                    print(f"Elapsed: {elapsed_str}")
                    print("---------Total---------")
                    for direction_total in counts:
                        print(f"{direction_total}:")
                        for class_name_total in counts[direction_total]:
                            print(f"    {class_name_total}: {counts[direction_total][class_name_total]}")


                    if is_vehicle:
                        color = state.color
                    else:
                        color = "NA"

                    entry_dir = traffic_dir_str if traffic_dir_str else "N/A"
                    detection_log.append(f"{class_name} #{track_id} {color} {entry_dir}")

                    logger.write_row({
                        "timestamp": datetime.now().isoformat(),
                        "street": args.street,
                        "cross_street": args.cross,
                        "direction": traffic_dir_str,
                        "class": class_name,
                        "color": color,
                    })

                overlay.draw_tracked_box(frame, x1, y1, x2, y2, class_name, track_id, traffic_dir_str)

            if show_line == 1:
                cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)

            # scale the frame to the target size first, then draw the panel at that
            # resolution so its text is rendered crisply instead of being upscaled
            fh, fw = frame.shape[:2]
            fit = min(target_w / fw, target_h / fh)
            big = cv2.resize(frame, (int(fw * fit), int(fh * fit))) if fit != 1 else frame

            if show_totals or show_log:
                display = overlay.draw_side_panel(big, counts, log=detection_log,
                                                  show_totals=show_totals, show_log=show_log, scale=fit)
            else:
                display = big

            if write and writer is None:
                date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                record_file = root / (file_name + "_" + date_time + ".mp4")
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(record_file), fourcc, 20.0, (w, h))

            if write and writer is not None:
                writer.write(frame)

            title = f"Detecting Traffic at {location} | {elapsed_str}"
            if write:
                title = "Recording | " + title
            cv2.setWindowTitle(window_name, title)

            cv2.resizeWindow(window_name, display.shape[1], display.shape[0])
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                rotate = (rotate + 1) % 4

            if key == ord("d"):
                show_log = not show_log
            if key == ord("t"):
                show_totals = not show_totals

            if key == ord("s"):
                if writer is None:
                    write = True
                else:
                    write = False
                    writer.release()
                    writer = None

            if key == ord("p"):
                if writer is not None:
                    write = not write

            if key == ord("l"):
                if show_line == 0:
                    show_line = 1
                
                else:
                    show_line = 0

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        logger.close()

if __name__ == "__main__":
    main()
