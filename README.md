# VideoProcessor Time-Lapse Generator

A professional desktop application for processing dashcam videos. Merges multiple recordings, generates configurable time-lapses, removes stationary segments, blurs license plates using object detection, and overlays background music.

## Features
- **Batch Processing:** Handles 100+ videos, merging them sequentially.
- **Stationary Segment Removal:** Uses OpenCV frame differencing to detect when the car is stopped (e.g., at traffic lights) and cuts/speeds up those segments.
- **License Plate Blurring:** Integrates YOLOv8 and Haar Cascades for dynamic privacy blurring.
- **Time-Lapse Generation:** Configurable speed multipliers (2x, 4x, etc.).
- **Audio Processing:** Adds background music (looping/trimming automatically).
- **GUI:** Modern PySide6 interface with real-time logging and preview.

## Installation

1. Install Python 3.11+.
2. Install FFmpeg and ensure it's added to your system `PATH`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application either from the built executable or
   directly with Python via the virtual environment:

  Option 1: Run the Built Executable
   ```bash
   .\dist\VideoProcessor\VideoProcessor.exe 
   ```

  Option 2: Run via Python in the Virtual Environment
   ```bash
   1 .\.venv\Scripts\Activate.ps1
   2 python main.py
  (Or directly in one command without activating:
  ```
2. Select your Input directory containing the dashcam videos.
3. Configure your output settings (speed, music, thresholds).
4. Click **Start Processing**.

## Notes
- First run will automatically download standard YOLOv8 weights and dedicated license plate detection weights (`license_plate_detector.pt`) to the `models/yolo` directory.
- License plate blurring isolates and blurs only number plates without blurring the complete vehicle. You can also place custom fine-tuned weights at `models/yolo/license_plate_detector.pt` or `models/yolo/plate.pt`.
