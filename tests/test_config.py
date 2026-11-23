import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

from utils.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = Path(self.temp_dir.name) / "config.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_initialization_when_file_not_found(self):
        cfg = Config(config_path=self.config_file)
        self.assertFalse(self.config_file.exists())
        self.assertEqual(cfg.get("speed_multiplier"), 4.0)
        self.assertEqual(cfg.get("blur_plates"), True)

    def test_load_existing_config(self):
        initial_data = {
            "speed_multiplier": 8.0,
            "blur_plates": False,
            "output_filename": "CustomOutput"
        }
        with open(self.config_file, "w") as f:
            json.dump(initial_data, f)

        cfg = Config(config_path=self.config_file)
        self.assertEqual(cfg.get("speed_multiplier"), 8.0)
        self.assertEqual(cfg.get("blur_plates"), False)
        self.assertEqual(cfg.get("output_filename"), "CustomOutput")

    def test_set_does_not_save_to_disk_by_default(self):
        initial_data = {
            "input_dir": "/initial/path",
            "speed_multiplier": 4.0
        }
        with open(self.config_file, "w") as f:
            json.dump(initial_data, f)

        cfg = Config(config_path=self.config_file)

        # Update in-memory values
        cfg.set("input_dir", "/new/path")
        cfg.set("speed_multiplier", 16.0)

        # In-memory values should be updated
        self.assertEqual(cfg.get("input_dir"), "/new/path")
        self.assertEqual(cfg.get("speed_multiplier"), 16.0)

        # File on disk should remain unchanged
        with open(self.config_file, "r") as f:
            disk_data = json.load(f)
        self.assertEqual(disk_data["input_dir"], "/initial/path")
        self.assertEqual(disk_data["speed_multiplier"], 4.0)

    def test_set_with_auto_save_true(self):
        cfg = Config(config_path=self.config_file)
        cfg.set("output_filename", "SavedOutput", auto_save=True)

        self.assertTrue(self.config_file.exists())
        with open(self.config_file, "r") as f:
            disk_data = json.load(f)
        self.assertEqual(disk_data["output_filename"], "SavedOutput")

    def test_explicit_save(self):
        cfg = Config(config_path=self.config_file)
        cfg.set("speed_multiplier", 10.0)
        self.assertFalse(self.config_file.exists())

        cfg.save()
        self.assertTrue(self.config_file.exists())
        with open(self.config_file, "r") as f:
            disk_data = json.load(f)
        self.assertEqual(disk_data["speed_multiplier"], 10.0)


class TestMainWindowConfigInteractions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        import sys
        cls.app = QApplication.instance() or QApplication(sys.argv)

    @patch("gui.main_window.app_config.save")
    @patch("gui.main_window.QFileDialog.getExistingDirectory", return_value="/selected/input/dir")
    def test_select_input_does_not_save(self, mock_dialog, mock_save):
        from gui.main_window import MainWindow, app_config
        win = MainWindow()
        win.select_input()
        self.assertEqual(app_config.get("input_dir"), "/selected/input/dir")
        mock_save.assert_not_called()

    @patch("gui.main_window.app_config.save")
    @patch("gui.main_window.QFileDialog.getOpenFileName", return_value=("/selected/music.mp3", ""))
    def test_select_music_does_not_save(self, mock_dialog, mock_save):
        from gui.main_window import MainWindow, app_config
        win = MainWindow()
        win.select_music()
        self.assertEqual(app_config.get("music_path"), "/selected/music.mp3")
        mock_save.assert_not_called()

    @patch("gui.main_window.app_config.save")
    @patch("gui.main_window.ProcessingWorker")
    def test_start_processing_does_not_save(self, mock_worker_cls, mock_save):
        from gui.main_window import MainWindow, app_config
        mock_worker = MagicMock()
        mock_worker_cls.return_value = mock_worker

        app_config.set("input_dir", "/valid/dir")
        win = MainWindow()
        win.spin_speed.setValue(8.0)
        win.chk_blur.setChecked(False)

        win.start_processing()

        self.assertEqual(app_config.get("speed_multiplier"), 8.0)
        self.assertEqual(app_config.get("blur_plates"), False)
        mock_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
