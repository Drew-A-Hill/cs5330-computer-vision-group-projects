"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

overlay.py

"""
import cv2
import numpy as np


def draw_filtered_box(frame, x1, y1, x2, y2):
    """


    """
    # Color is red if object is filtered out ie stationary or too far.
    color = (0, 0, 255)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)


def draw_tracked_box(frame, x1, y1, x2, y2, class_name, object_id, direction):
    """


    """
    color = (0, 255, 0)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(frame, f"ID: {object_id}\n Class: {class_name}\n Direction: {direction}", (int(x1), int(y1) - 35),
                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)


def draw_recording(frame):
    """


    """
    cv2.circle(frame, (25, 30), 10, (0, 0, 255), -1)
    cv2.putText(frame, "Recording", (45, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def draw_side_panel(frame, counts, elapsed="", location=""):
    """


    """
    left_pad = 10
    lines = []
    if elapsed:
        lines.append((f"Time: {elapsed}", left_pad, 0.5, 2, 32))

    if location:
        lines.append((location, left_pad, 0.5, 1, 20))

    for direction, class_counts in counts.items():
        lines.append((f"{direction}:", left_pad, 0.5, 2, 30))
        for name, count in class_counts.items():
            lines.append((f"{name}: {count}", left_pad + 20, 0.4, 1, 26))

    max_right = 0
    for text, x_off, scale, thickness, _ in lines:
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        max_right = max(max_right, x_off + text_w)
    panel_width = max_right + left_pad

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (255, 255, 255)

    red = (0, 0, 255)
    y_off = 30
    for text, x_off, scale, thickness, advance in lines:
        cv2.putText(canvas, text, (w + x_off, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, red, thickness)
        y_off += advance
    return canvas
