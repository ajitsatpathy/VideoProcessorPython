import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from processing.plate_detector import PlateDetector

class TestPlateDetector(unittest.TestCase):
    def test_default_initialization(self):
        detector = PlateDetector()
        self.assertTrue(detector.use_yolo)
        self.assertIsNotNone(detector.model)
        self.assertTrue(detector.is_plate_model)

    def test_yolo_initialization(self):
        detector = PlateDetector("models/yolo/yolov8n.pt")
        self.assertTrue(detector.use_yolo)
        self.assertIsNotNone(detector.model)
        self.assertFalse(detector.is_plate_model)

    def test_blur_plates_with_yolo(self):
        detector = PlateDetector("models/yolo/yolov8n.pt")
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        processed = detector.blur_plates(frame)
        self.assertEqual(processed.shape, (300, 300, 3))
        self.assertIsInstance(processed, np.ndarray)

    def test_blur_only_plate_not_whole_vehicle(self):
        """
        Empirically verifies that blurring affects ONLY the license plate,
        leaving the vehicle body completely intact and unblurred.
        """
        detector = PlateDetector()
        frame = np.ones((600, 800, 3), dtype=np.uint8) * 120

        # Draw vehicle body (red)
        cv2.rectangle(frame, (150, 200), (650, 500), (40, 40, 200), -1)
        # Draw vehicle windshield
        cv2.rectangle(frame, (250, 220), (550, 320), (220, 220, 220), -1)
        # Draw license plate (white rectangle with black text)
        cv2.rectangle(frame, (340, 430), (460, 470), (255, 255, 255), -1)
        cv2.putText(frame, "ABC-1234", (345, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        original_car_body = frame[210:250, 160:240].copy()
        original_plate = frame[435:465, 350:450].copy()

        blurred_frame = detector.blur_plates(frame)

        # Vehicle body outside plate region MUST remain strictly unmodified
        after_car_body = blurred_frame[210:250, 160:240]
        self.assertTrue(
            np.array_equal(original_car_body, after_car_body),
            "Vehicle body was modified or blurred! Only the number plate should be blurred."
        )

        # License plate region MUST be blurred
        after_plate = blurred_frame[435:465, 350:450]
        self.assertFalse(
            np.array_equal(original_plate, after_plate),
            "License plate region was not blurred."
        )

    def test_vehicle_model_does_not_blur_whole_vehicle(self):
        """
        Verifies that when a general vehicle model (COCO) is used, the whole
        vehicle bounding box is NOT blurred.
        """
        detector = PlateDetector("models/yolo/yolov8n.pt")
        self.assertFalse(detector.is_plate_model)

        frame = np.ones((500, 500, 3), dtype=np.uint8) * 100
        # Simulated car area with no plate
        cv2.rectangle(frame, (100, 100), (400, 400), (50, 150, 50), -1)
        car_body_before = frame[120:180, 120:180].copy()

        # Mock YOLO model to return a 'car' bounding box spanning [100, 100, 400, 400]
        mock_box = MagicMock()
        mock_box.xyxy = [np.array([100, 100, 400, 400])]
        mock_box.cls = [2]  # COCO class 2 = car
        mock_box.conf = [0.9]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        with patch.object(detector, "model", return_value=[mock_result]):
            blurred_frame = detector.blur_plates(frame)

        # Whole vehicle should NOT be blurred
        car_body_after = blurred_frame[120:180, 120:180]
        self.assertTrue(
            np.array_equal(car_body_before, car_body_after),
            "Vehicle bounding box was blurred when using vehicle detector!"
        )

    def test_locate_plates_in_vehicle_roi(self):
        detector = PlateDetector("models/yolo/yolov8n.pt")
        vehicle_roi = np.zeros((300, 400, 3), dtype=np.uint8)
        # Put simulated plate in lower area
        cv2.rectangle(vehicle_roi, (150, 200), (270, 240), (255, 255, 255), -1)
        cv2.putText(vehicle_roi, "XYZ-999", (155, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        candidates = detector._locate_plates_in_vehicle_roi(vehicle_roi)
        self.assertGreater(len(candidates), 0)
        px1, py1, px2, py2 = candidates[0]
        self.assertLessEqual(px1, 160)
        self.assertGreaterEqual(px2, 220)
        self.assertLessEqual(px2, 280)

    def test_empty_and_invalid_inputs(self):
        detector = PlateDetector()
        self.assertIsNone(detector.blur_plates(None))
        empty = np.array([])
        self.assertEqual(detector.blur_plates(empty).size, 0)
        zero_sized = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(detector.blur_plates(zero_sized).shape, (0, 0, 3))

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

    def test_disabled_plate_detector_does_not_load_model(self):
        with patch.object(PlateDetector, "_resolve_model_path") as mock_resolve:
            detector = PlateDetector(enabled=False)
            self.assertFalse(detector.enabled)
            self.assertIsNone(detector.model)
            self.assertFalse(detector.use_yolo)
            self.assertIsNone(detector.cascade)
            mock_resolve.assert_not_called()

        # Frame should be returned untouched without attempting detection
        frame = np.ones((50, 50, 3), dtype=np.uint8) * 200
        result = detector.blur_plates(frame)
        self.assertTrue(np.array_equal(frame, result))

    def test_disabled_plate_detector_load_model_explicitly(self):
        detector = PlateDetector(enabled=False)
        self.assertIsNone(detector.model)
        self.assertFalse(detector.use_yolo)

        detector.load_model("models/yolo/yolov8n.pt")
        self.assertTrue(detector.use_yolo)
        self.assertIsNotNone(detector.model)

if __name__ == "__main__":
    unittest.main()
