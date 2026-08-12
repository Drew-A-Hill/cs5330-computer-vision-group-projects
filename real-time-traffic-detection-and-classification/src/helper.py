"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    helper.py

    Shared helper functions.
"""

def crop_object(frame, x1, y1, x2, y2, pad=0.10):
    """
        Crops the detected object out of the frame with some padding around it.

        frame: cv2.Mat - This is the frame captured.
        x1: int - Left edge of the cropped box.
        y1: int - Top edge of the cropped box.
        x2: int - Right edge of the cropped box.
        y2: int - Bottom edge of the cropped box.
        pad: float - Fraction of the box size to add as padding.

        returns cv2.Mat with the cropped and padded region of the frame
    """
    # Gets the height and width of the frame.
    height, width = frame.shape[:2]

    # Calculates padding.
    pad_x = (x2 - x1) * pad
    pad_y = (y2 - y1) * pad

    # Clamps the padding.
    left = max(0, int(x1 - pad_x))
    right = min(width, int(x2 + pad_x))
    top = max(0, int(y1 - pad_y))
    bottom = min(height, int(y2 + pad_y))

    return frame[top:bottom, left:right]
