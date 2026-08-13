"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    tracker.py

    This file keeps a hastory of an object being tracked to tell if it is
    moving, the direction it is moving, and if it passes the line we use
    to determine if the object will be counted.
"""
from collections import Counter, defaultdict, deque

class Tracker:
    """
    Class used to track movement and direction.
    """

    def __init__(self):
        """
            Sets up empty history for new tracked object.
        """
        self.centroids = deque(maxlen=15)
        self.counted = False
        self.color_counts = Counter()
        self.best_h = 0

    def add_color(self, label):
        """
            Votes for a color.

            label: str - The color to vote for.
        """
        if label and label != "unknown":
            self.color_counts[label] += 1

    def most_common(self, counts):
        """
            Returns the most common item from the counter or unknown if the
            the counter is empty.

            counts: Counter - Holds the votes for a color.

            returns the most common color label.
        """
        if counts:
            return counts.most_common(1)[0][0]

        return "unknown"

    def get_tracked_color(self):
        """
            Getter that gets the color with the most votes for the tracked object.

            returns the label of the color.
        """
        return self.most_common(self.color_counts)

    def update(self, cx, cy):
        """
            Records the objects center point.

            cx: int - X cordinate for the center.
            cy: int - Y cordinate for the center.
        """
        self.centroids.append((cx, cy))

    def distance_moved(self):
        """
            How far the object has moved from its first recorded center point
            to its last.

            returns float - Straight-line distance in pixels
        """
        # Checks that there are at least 2 points recorded.
        if len(self.centroids) < 2:
            return 0.0

        # Coordinates of the center of the object for prev and curr frames.
        x0, y0 = self.centroids[0]
        x1, y1 = self.centroids[-1]

        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def direction(self):
        """
            Determines which direction object is moving horizontally.

            returns the direction as str left_to_right or right_to_left or 
            None if unknown.
        """
        # Checks that there are at least 2 points recorded.
        if len(self.centroids) < 2:
            return None

        # Coordinates of the center of the object for prev and curr frames.
        x0, y0 = self.centroids[0]
        x1, y1 = self.centroids[-1]

        if x1 > x0:
            return "left_to_right"

        return "right_to_left"

    def crossed_line(self, line_x):
        """
            Checks if the object crossed the line.

            line_x: int - X position of the counting line.

            returns True if it crossed the line this frame.
        """
        # Checks that there are at least 2 points recorded.
        if len(self.centroids) < 2:
            return False

        # Coordinates of the center of the object for prev and curr frames.
        x_prev, y_prev = self.centroids[-2]
        x_curr, y_curr = self.centroids[-1]

        return (x_prev < line_x <= x_curr) or (x_prev > line_x >= x_curr)
