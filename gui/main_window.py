import sys
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QProgressBar, 
                               QTextEdit, QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap

from utils.config import app_config
from processing.pipeline import VideoPipeline

class ProcessingWorker(QThread):
    progress_signal = Signal(int)
    log_signal = Signal(str)
    preview_signal = Signal(np.ndarray)
    finished_signal = Signal(bool)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pipeline = VideoPipeline(
            config,
            progress_callback=self.progress_signal.emit,
            log_callback=self.log_signal.emit,
            frame_callback=self.preview_signal.emit
        )

    def run(self):
        success = self.pipeline.run()
        self.finished_signal.emit(success)

    def cancel(self):
        if self.pipeline:
            self.pipeline.cancel()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoProcessor Time-Lapse Generator")
        self.resize(900, 700)
        self.worker = None

        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Settings Group
        settings_group = QGroupBox("Configuration")
        settings_layout = QFormLayout()

        # Input / Output
        self.lbl_input = QLabel(app_config.get("input_dir", "Not selected"))
        btn_input = QPushButton("Select Input Folder")
        btn_input.clicked.connect(self.select_input)
        
        self.lbl_music = QLabel(app_config.get("music_path", "Not selected"))
        btn_music = QPushButton("Select Background Music")
        btn_music.clicked.connect(self.select_music)

        settings_layout.addRow(btn_input, self.lbl_input)
        settings_layout.addRow(btn_music, self.lbl_music)

        # Multipliers
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(1.0, 32.0)
        self.spin_speed.setValue(app_config.get("speed_multiplier", 4.0))
        settings_layout.addRow("Speed Multiplier:", self.spin_speed)

        # Checkboxes
        self.chk_blur = QCheckBox("Enable License Plate Blurring")
        self.chk_blur.setChecked(app_config.get("blur_plates", True))
        settings_layout.addRow("", self.chk_blur)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # Video Preview
        self.lbl_preview = QLabel("Video Preview")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumHeight(360)
        self.lbl_preview.setStyleSheet("background-color: #000; color: #fff;")
        main_layout.addWidget(self.lbl_preview)

        # Logs
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        main_layout.addWidget(self.txt_logs)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Controls
        controls_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Processing")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        self.btn_start.clicked.connect(self.start_processing)
        self.btn_cancel.clicked.connect(self.cancel_processing)

        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_cancel)
        main_layout.addLayout(controls_layout)

    def select_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.lbl_input.setText(folder)
            app_config.set("input_dir", folder)

    def select_music(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Music File", "", "Audio Files (*.mp3 *.wav *.aac)")
        if file:
            self.lbl_music.setText(file)
            app_config.set("music_path", file)

    def start_processing(self):
        if not app_config.get("input_dir"):
            self.append_log("Error: Please select an input directory.")
            return

        # Update config from UI
        app_config.set("speed_multiplier", self.spin_speed.value())
        app_config.set("blur_plates", self.chk_blur.isChecked())

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.txt_logs.clear()
        self.progress_bar.setValue(0)

        self.worker = ProcessingWorker(app_config)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.log_signal.connect(self.append_log)
        self.worker.preview_signal.connect(self.update_preview)
        self.worker.finished_signal.connect(self.processing_finished)
        self.worker.start()

    def cancel_processing(self):
        if self.worker:
            self.append_log("Cancelling... Please wait.")
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)

    def processing_finished(self, success):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None
        if success:
            self.append_log("Workflow finished successfully.")
            self.progress_bar.setValue(100)
        else:
            self.append_log("Workflow failed or was cancelled.")

    def append_log(self, text):
        self.txt_logs.append(text)

    def update_preview(self, frame_array):
        h, w, ch = frame_array.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        # Scaled to fit label
        self.lbl_preview.setPixmap(pixmap.scaled(self.lbl_preview.width(), self.lbl_preview.height(), Qt.KeepAspectRatio))

    def apply_dark_theme(self):
        # Basic Dark Mode
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0078D7;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #1084ea;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
            QTextEdit, QSpinBox, QDoubleSpinBox {
                background-color: #1e1e1e;
                border: 1px solid #333333;
            }
            QGroupBox {
                border: 1px solid #444444;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)

