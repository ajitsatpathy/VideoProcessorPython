import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from processing.pipeline import VideoPipeline

class TestVideoPipelineModelLoading(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()

        # Create a tiny valid dummy MP4 video file so pipeline passes the video search
        dummy_video_path = self.input_dir / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(dummy_video_path), fourcc, 10, (100, 100))
        for _ in range(5):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("processing.pipeline.PlateDetector")
    @patch("processing.pipeline.AudioProcessor.add_background_music", return_value=True)
    def test_model_loaded_when_blur_plates_is_true(self, mock_audio, mock_plate_detector_cls):
        mock_detector_instance = MagicMock()
        mock_detector_instance.use_yolo = True
        mock_detector_instance.model = MagicMock()
        mock_detector_instance.blur_plates.side_effect = lambda f: f
        mock_plate_detector_cls.return_value = mock_detector_instance

        logs = []
        config = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "blur_plates": True,
            "speed_multiplier": 1.0
        }
        pipeline = VideoPipeline(config, log_callback=logs.append)
        result = pipeline.run()

        self.assertTrue(result)
        # Verify PlateDetector was initialized with enabled=True
        mock_plate_detector_cls.assert_called_once_with(enabled=True)
        self.assertIsNotNone(pipeline.plate_detector)
        self.assertTrue(any("License plate blurring is enabled" in msg for msg in logs))
        self.assertTrue(any("loaded successfully" in msg for msg in logs))

    @patch("processing.pipeline.PlateDetector")
    @patch("processing.pipeline.AudioProcessor.add_background_music", return_value=True)
    def test_model_not_loaded_when_blur_plates_is_false(self, mock_audio, mock_plate_detector_cls):
        logs = []
        config = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "blur_plates": False,
            "speed_multiplier": 1.0
        }
        pipeline = VideoPipeline(config, log_callback=logs.append)
        result = pipeline.run()

        self.assertTrue(result)
        # Verify PlateDetector was NEVER instantiated
        mock_plate_detector_cls.assert_not_called()
        self.assertIsNone(pipeline.plate_detector)
        self.assertTrue(any("License plate blurring is disabled. Skipping model loading." in msg for msg in logs))

if __name__ == "__main__":
    unittest.main()
