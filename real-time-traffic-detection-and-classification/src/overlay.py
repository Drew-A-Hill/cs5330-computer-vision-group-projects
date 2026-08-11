import cv2


def draw_filtered_box(frame, x1, y1, x2, y2):
    # Color is red if object is filtered out ie stationary or too far.
    color = (0, 0, 255)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)


def draw_tracked_box(frame, x1, y1, x2, y2, class_name):
    color = (0, 255, 0)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_counts(frame, counts):
    y_off = 20
    for cname, c in counts.items():
        cv2.putText(frame, f"{cname}: {c}", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_off += 22
