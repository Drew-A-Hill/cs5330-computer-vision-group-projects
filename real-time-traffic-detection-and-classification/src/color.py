"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    color.py

    This file finds the color of a vehicle. It samples the door of the vehicle
    to determine the color.
"""
import cv2
import numpy as np
from helper import crop_object

def paint_region(frame, x1, y1, x2, y2):
    """
        Returns the area of the vehicle to likely contain the paint color. It
        avoids windows, tires and the road.

        frame: cv2.Mat - This is the frame captured.
        x1: int - Left edge of the cropped box.
        y1: int - Top edge of the cropped box.
        x2: int - Right edge of the cropped box.
        y2: int - Bottom edge of the cropped box.

        returns cv2.Mat with the region of the door.
    """
    # Gets the height and with of the box.
    width = x2 - x1
    height = y2 - y1

    # Gets the middle region of the vehicle.
    rx1, rx2 = int(x1 + 0.20 * width), int(x1 + 0.80 * width)
    ry1, ry2 = int(y1 + 0.45 * height), int(y1 + 0.78 * height)

    return crop_object(frame, rx1, ry1, rx2, ry2, 0)

def dominant_hsv(region):
    """
        Finds the main color of a region as a median HSV value.

        region: cv2.Mat - The sampled paint region.

        returns hsv medians or none if the region is empty or 
        doesnt have enough pixels to use.
    """
    # Skips empty regions.
    if region.size == 0:
        return None

    # Converts the region from BGR to HSV.
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Keeps only pixels that are not to dark or bright.
    keep = (v > 25) & (v < 245)

    # Returns none if there arent enough usable pixels
    if keep.sum() < 50:
        return None

    return np.median(h[keep]), np.median(s[keep]), np.median(v[keep])

def classify_color(hsv):
    """
       Converts the median HSV value into a color name.

       hsv: tuple(h, s, v) or none - The dominat HSV of the region.

       returns the detected color name or unknown as str if hsv is none. 
    """
    # HSV returned as none for the region.
    if hsv is None:
        return "unknown"

    h, s, v = hsv

    # Picks the color.
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
        Runs full pipeline to find color of the vehicle detected.

        frame: cv2.Mat - This is the frame captured.
        x1: int - Left edge of the cropped box.
        y1: int - Top edge of the cropped box.
        x2: int - Right edge of the cropped box.
        y2: int - Bottom edge of the cropped box.

        returns the detected color name or unknown as str if hsv is none.
    """
    return classify_color(dominant_hsv(paint_region(frame, x1, y1, x2, y2)))
