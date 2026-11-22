import cv2
import numpy as np

class MotionDetector:
    def __init__(self, threshold=150000, min_stationary_frames=90):
        self.threshold = threshold
        self.min_stationary_frames = min_stationary_frames
        self.prev_frame = None
        self.stationary_count = 0

    def is_stationary(self, frame):
        """
        Returns True if the frame is considered stationary compared to the previous frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        # Compute absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Calculate motion score
        score = np.sum(thresh)
        self.prev_frame = gray

        if score < self.threshold:
            self.stationary_count += 1
        else:
            self.stationary_count = 0

        return self.stationary_count >= self.min_stationary_frames
