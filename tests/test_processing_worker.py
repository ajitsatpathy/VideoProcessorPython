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

    def test_processing_finished_waits_for_worker(self):
        from gui.main_window import MainWindow
        win = MainWindow()
        mock_worker = MagicMock()
        win.worker = mock_worker

        win.processing_finished(True)

        mock_worker.wait.assert_called_once_with(5000)
        self.assertIsNone(win.worker)

    def test_close_event_cancels_and_waits_running_worker(self):
        from gui.main_window import MainWindow
        from PySide6.QtGui import QCloseEvent
        win = MainWindow()
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        mock_worker.wait.return_value = True
        win.worker = mock_worker

        event = QCloseEvent()
        win.closeEvent(event)

        mock_worker.cancel.assert_called_once()
        mock_worker.wait.assert_called_with(3000)
        self.assertIsNone(win.worker)

    @patch("gui.main_window.VideoPipeline")
    def test_worker_real_thread_lifecycle(self, mock_pipeline_cls):
        import time
        from gui.main_window import MainWindow, app_config
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = True
        mock_pipeline_cls.return_value = mock_pipeline

        app_config.set("input_dir", "dummy_dir")
        win = MainWindow()
        win.start_processing()

        # Allow worker thread to run to completion in event loop
        for _ in range(50):
            if win.worker is None:
                break
            app.processEvents()
            time.sleep(0.01)

        self.assertIsNone(win.worker)
        self.assertTrue(any("Workflow finished successfully." in msg for msg in [win.txt_logs.toPlainText()]))

if __name__ == "__main__":
    unittest.main()
