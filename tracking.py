import cv2
import numpy as np


def intersection_area(rect1, rect2):
    x = max(rect1[0], rect2[0])
    y = max(rect1[1], rect2[1])
    w = min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - x
    h = min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - y
    return w * h if (w > 0 and h > 0) else 0


def iou(rect1, rect2):
    intersect = intersection_area(rect1, rect2)
    area_a = rect1[2] * rect1[3]
    area_b = rect2[2] * rect2[3]
    return intersect / (area_a + area_b - intersect)


def center(rect):
    transform = np.array([[1, 0], [0, 1], [.5, 0], [0, .5]])
    return (rect @ transform).astype(int)


class CsrtTracker:
    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()
        self.trajectory = np.empty((0, 2))
        self.tracked = None

    def init(self, img, tracked):
        self.tracker.init(img, tracked)
        self.tracked = tracked
        self.trajectory = np.array([center(tracked)])
        return self.tracked

    def track(self, img, detects, threshold=0.5):
        assert self.tracker is not None
        # Search for a detected rectangle that maximizes the match with the currently tracked rectangle
        iou_scores = np.array([iou(detect, self.tracked) for detect in detects])
        matching = np.argwhere(iou_scores > threshold)
        # If only one detection match above the threshold, use it. Otherwise, follow the tracker
        if matching.size != 1:
            # Use the tracker's update method
            _, self.tracked = self.tracker.update(img)
        else:
            # Reinitialize the tracker with the detection
            self.tracked = detects[matching.item()]
            self.tracker.init(img, self.tracked)

        # Add the new rectangle center to the trajectory
        self.trajectory = np.vstack([self.trajectory, center(self.tracked)])
        # Return the currently tracked rectangle
        return self.tracked
