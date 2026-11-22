import sys
import unittest
from unittest.mock import MagicMock, patch
from PySide6.QtCore import QCoreApplication

# Ensure QCoreApplication exists for Qt Signals
app = QCoreApplication.instance() or QCoreApplication(sys.argv)

from gui.main_window import ProcessingWorker

class TestProcessingWorker(unittest.TestCase):
    @patch("gui.main_window.VideoPipeline")
    def test_worker_run_success(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = True
        mock_pipeline_cls.return_value = mock_pipeline

        worker = ProcessingWorker({"input_dir": "dummy"})
        finished_results = []
        worker.finished_signal.connect(finished_results.append)

        worker.run()

        self.assertEqual(finished_results, [True])

    @patch("gui.main_window.VideoPipeline")
    def test_worker_run_exception_handling(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = RuntimeError("Disk full")
        mock_pipeline_cls.return_value = mock_pipeline

        worker = ProcessingWorker({"input_dir": "dummy"})
        finished_results = []
        log_messages = []
        worker.finished_signal.connect(finished_results.append)
        worker.log_signal.connect(log_messages.append)

        # Ensure run() catches the exception and signals failure gracefully
        worker.run()

        self.assertEqual(finished_results, [False])
        self.assertTrue(any("Disk full" in msg for msg in log_messages))

if __name__ == "__main__":
    unittest.main()
