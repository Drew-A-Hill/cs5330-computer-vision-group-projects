"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

overlay.py

"""
import cv2
import numpy as np

def draw_filtered_box(frame, x1, y1, x2, y2, class_name, object_id):
    """


    """
    box_color = (0, 0, 255)
    text_color = (0, 255, 255)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
    cv2.putText(frame, f"ID: {object_id}\n Class: {class_name}\n Direction: N/A", (int(x1), int(y1) - 35),
                cv2.FONT_HERSHEY_SIMPLEX, .45, text_color, 1)


def draw_tracked_box(frame, x1, y1, x2, y2, class_name, object_id, direction):
    """


    """
    color = (0, 255, 0)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(frame, f"ID: {object_id}\n Class: {class_name}\n Direction: {direction}", (int(x1), int(y1) - 35),
                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)


def draw_side_panel(frame, counts, log=None, show_totals=True, show_log=True, scale=1.0):
    """


    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    red = (0, 0, 255)
    black = (0, 0, 0)
    left_pad = int(3 * scale)
    indent = int(12 * scale)
    thick = max(1, round(scale))
    log_font = 0.4 * scale
    line_h = max(10, int(18 * scale))
    title_font = 0.5 * scale
    title_thick = thick + 1
    title_adv = int(15 * scale)

    total_lines = []
    if show_totals:
        directions = list(counts.items())
        for i, (direction, class_counts) in enumerate(directions):
            total_lines.append((f"{direction}:", left_pad, 0.30 * scale, thick, int(9 * scale)))
            for name, count in class_counts.items():
                total_lines.append((f"{name}: {count}", left_pad + indent, 0.25 * scale, thick, int(8 * scale)))
            if i < len(directions) - 1:
                total_lines.append(("", left_pad, 0.30 * scale, thick, int(6 * scale)))

    log_entries = list(log) if (show_log and log) else []

    max_right = 0
    for text, x_off, s, t, _ in total_lines:
        (text_w, _), _ = cv2.getTextSize(text, font, s, t)
        max_right = max(max_right, x_off + text_w)
    for text in log_entries:
        (text_w, _), _ = cv2.getTextSize(text, font, log_font, thick)
        max_right = max(max_right, left_pad + text_w)
    for title, shown in (("Totals", show_totals), ("Detections", show_log)):
        if shown:
            (text_w, _), _ = cv2.getTextSize(title, font, title_font, title_thick)
            max_right = max(max_right, left_pad + text_w)
    panel_width = max(int(120 * scale), max_right + left_pad)

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (255, 255, 255)

    def put(text, x_off, s, t, y):
        cv2.putText(canvas, text, (w + x_off, y), font, s, red, t)

    def hline(y):
        cv2.line(canvas, (w, y), (w + panel_width, y), black, 1)

    def draw_totals(start_y, bottom):
        y = start_y
        put("Totals", left_pad, title_font, title_thick, y)
        y += title_adv
        for text, x_off, s, t, advance in total_lines:
            if y > bottom - 6:
                break
            put(text, x_off, s, t, y)
            y += advance

    def draw_log(start_y):
        y = start_y
        put("Detections", left_pad, title_font, title_thick, y)
        y += title_adv
        rows = max(0, (h - y - 4) // line_h)
        for text in log_entries[-rows:]:
            put(text, left_pad, log_font, thick, y)
            y += line_h

    top = int(22 * scale)

    if show_totals and show_log:
        log_top = int(h * 0.8)
        draw_totals(top, log_top)
        hline(log_top)
        draw_log(log_top + line_h)
    elif show_totals:
        draw_totals(top, h)
    elif show_log:
        draw_log(top)

    return canvas
