import unittest
from unittest.mock import patch
import numpy as np
import cv2
from processing.plate_detector import PlateDetector

class TestPlateDetector(unittest.TestCase):
    def test_yolo_initialization(self):
        detector = PlateDetector("models/yolo/yolov8n.pt")
        self.assertTrue(detector.use_yolo)
        self.assertIsNotNone(detector.model)

    def test_blur_plates_with_yolo(self):
        detector = PlateDetector("models/yolo/yolov8n.pt")
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        processed = detector.blur_plates(frame)
        self.assertEqual(processed.shape, (300, 300, 3))
        self.assertIsInstance(processed, np.ndarray)

    @patch("processing.plate_detector.HAS_YOLO", False)
    def test_fallback_without_cascade_classifier(self):
        # Even if OpenCV does not have CascadeClassifier, it shouldn't raise an exception
        detector = PlateDetector("models/yolo/yolov8n.pt")
        self.assertFalse(detector.use_yolo)
        self.assertIsNone(detector.model)
        
        # Verify blur_plates still succeeds as a pass-through
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        processed = detector.blur_plates(frame)
        self.assertEqual(processed.shape, (200, 200, 3))

    @patch("processing.plate_detector.YOLO", side_effect=Exception("Simulated YOLO load failure"))
    def test_fallback_on_yolo_load_failure(self, mock_yolo):
        detector = PlateDetector("models/yolo/yolov8n.pt")
        self.assertFalse(detector.use_yolo)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processed = detector.blur_plates(frame)
        self.assertEqual(processed.shape, (100, 100, 3))

if __name__ == "__main__":
    unittest.main()
