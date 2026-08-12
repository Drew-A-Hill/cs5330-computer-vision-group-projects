"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

color.py

"""
import cv2
import numpy as np

def paint_region(frame, x1, y1, x2, y2):
    """


    """
    w, h = x2 - x1, y2 - y1
    rx1, rx2 = int(x1 + 0.20 * w), int(x1 + 0.80 * w)
    ry1, ry2 = int(y1 + 0.45 * h), int(y1 + 0.78 * h)
    return frame[max(0, ry1):ry2, max(0, rx1):rx2]

def dominant_hsv(region):
    """


    """
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    keep = (v > 25) & (v < 245)
    if keep.sum() < 50:
        return None
    return np.median(h[keep]), np.median(s[keep]), np.median(v[keep])

def classify_color(hsv):
    """


    """
    if hsv is None:
        return "unknown"
    h, s, v = hsv
    if s < 45:
        if v < 60:
            return "black"
        if v < 150:
            return "gray"
        return "white"
    if h < 8 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple"

def detect_color(frame, x1, y1, x2, y2):
    """


    """
    return classify_color(dominant_hsv(paint_region(frame, x1, y1, x2, y2)))
