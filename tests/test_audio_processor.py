import unittest
from unittest.mock import patch, MagicMock
from processing.audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    def test_get_ffmpeg_path(self):
        path = AudioProcessor.get_ffmpeg_path()
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)

    @patch("ffmpeg.run")
    def test_add_background_music_success(self, mock_ffmpeg_run):
        mock_ffmpeg_run.return_value = None
        result = AudioProcessor.add_background_music(
            "dummy_video.mp4",
            "dummy_audio.mp3",
            "dummy_output.mp4"
        )
        self.assertTrue(result)
        mock_ffmpeg_run.assert_called_once()

    @patch("ffmpeg.run", side_effect=Exception("Encoding error"))
    def test_add_background_music_failure(self, mock_ffmpeg_run):
        result = AudioProcessor.add_background_music(
            "dummy_video.mp4",
            "dummy_audio.mp3",
            "dummy_output.mp4"
        )
        self.assertFalse(result)

    def test_real_audio_mixing_with_imageio_ffmpeg(self):
        import cv2, numpy as np, os
        audio_sample = "C:/workspacepython/Audios/Track2.mp3"
        if not os.path.exists(audio_sample):
            return

        test_video = "tests/temp_test_video.mp4"
        test_out = "tests/temp_test_out.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video, fourcc, 10, (160, 120))
        for _ in range(10):
            out.write(np.zeros((120, 160, 3), dtype=np.uint8))
        out.release()

        try:
            success = AudioProcessor.add_background_music(test_video, audio_sample, test_out)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(test_out))
            self.assertGreater(os.path.getsize(test_out), 0)
        finally:
            if os.path.exists(test_video):
                os.remove(test_video)
            if os.path.exists(test_out):
                os.remove(test_out)

if __name__ == "__main__":
    unittest.main()
