"""

"""
from collections import defaultdict, deque

class Tracker:
    """
    Per-track history used for the motion / direction / counting filters.
    """

    def __init__(self):
        self.centroids = deque(maxlen=15)
        self.counted = False

    def update(self, cx, cy):
        self.centroids.append((cx, cy))

    def displacement(self):
        if len(self.centroids) < 2:
            return 0.0
        x0, y0 = self.centroids[0]
        x1, y1 = self.centroids[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def direction(self):
        if len(self.centroids) < 2:
            return None
        x0, _ = self.centroids[0]
        x1, _ = self.centroids[-1]
        return "left_to_right" if x1 > x0 else "right_to_left"

    def crossed_line(self, line_x):
        """
        True if the track's centroid history straddles line_x this frame.
        """
        if len(self.centroids) < 2:
            return False
        (x_prev, _), (x_curr, _) = self.centroids[-2], self.centroids[-1]
        return (x_prev < line_x <= x_curr) or (x_prev > line_x >= x_curr)

    def crossed_exit(self, left_x, right_x):
        """
        True if the track reaches the frame edge it is travelling toward.
        """
        d = self.direction()
        if d == "left_to_right":
            return self.crossed_line(right_x)
        if d == "right_to_left":
            return self.crossed_line(left_x)
        return False