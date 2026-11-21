import cv2
import os
from pathlib import Path
from utils.logger import logger
from processing.motion_detector import MotionDetector
from processing.plate_detector import PlateDetector
from processing.audio_processor import AudioProcessor

class VideoPipeline:
    def __init__(self, config, progress_callback=None, log_callback=None, frame_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.frame_callback = frame_callback
        self.is_cancelled = False

    def _log(self, message):
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        input_dir = Path(self.config.get("input_dir"))
        output_dir = Path(self.config.get("output_dir"))
        output_dir.mkdir(exist_ok=True)
        
        speed = self.config.get("speed_multiplier", 4.0)
        blur_enabled = self.config.get("blur_plates", True)
        music_path = self.config.get("music_path", "")
        
        valid_exts = ['.mp4', '.mov', '.avi', '.mkv']
        videos = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts])
        
        if not videos:
            self._log("No valid video files found in the input directory.")
            return False

        self._log(f"Found {len(videos)} video files. Starting processing...")

        temp_output = output_dir / "temp_merged.mp4"
        final_output = output_dir / f"{self.config.get('output_filename', 'Timelapse')}.mp4"

        # Initialize components
        plate_detector = PlateDetector() if blur_enabled else None
        
        # Read first video to get properties
        cap = cv2.VideoCapture(str(videos[0]))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Output writer (using mp4v for basic MP4 encoding, H264 via FFmpeg is better but this works for OpenCV)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))

        total_videos = len(videos)
        frame_skip = int(speed) if speed >= 1 else 1

        for i, video_path in enumerate(videos):
            if self.is_cancelled:
                self._log("Processing cancelled by user.")
                break

            self._log(f"Processing video {i+1} of {total_videos}: {video_path.name}")
            cap = cv2.VideoCapture(str(video_path))
            
            # Motion detector per video or global? Let's do per video for simplicity, 
            # or global to handle stops across file splits. We'll do global instance.
            motion_detector = MotionDetector(
                threshold=self.config.get("motion_threshold", 150000),
                min_stationary_frames=int(self.config.get("min_stationary_seconds", 3.0) * fps)
            )

            frame_count = 0
            while cap.isOpened():
                if self.is_cancelled:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 1. Motion Detection
                is_stationary = motion_detector.is_stationary(frame)
                if is_stationary:
                    # Skip frame if stationary
                    continue

                # 2. Time-lapse (Frame skipping)
                if frame_count % frame_skip != 0:
                    continue

                # 3. Plate Blurring
                if blur_enabled and plate_detector:
                    frame = plate_detector.blur_plates(frame)

                # Write frame
                out.write(frame)

                # Send preview to GUI (throttle to avoid UI freeze)
                if self.frame_callback and frame_count % (frame_skip * 5) == 0:
                    # Resize for preview
                    preview = cv2.resize(frame, (640, 360))
                    # Convert BGR to RGB for Qt
                    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    self.frame_callback(preview)

            cap.release()
            
            if self.progress_callback:
                self.progress_callback(int(((i + 1) / total_videos) * 100))

        out.release()

        if self.is_cancelled:
            if temp_output.exists():
                os.remove(temp_output)
            return False

        # Post-Processing: Audio Overlay
        if music_path and Path(music_path).exists():
            self._log("Starting audio mixing phase...")
            success = AudioProcessor.add_background_music(
                str(temp_output), 
                music_path, 
                str(final_output),
                volume=self.config.get("music_volume", 0.5)
            )
            if success and temp_output.exists():
                os.remove(temp_output)
        else:
            self._log("No valid music path provided. Renaming temp file to final output.")
            if final_output.exists():
                os.remove(final_output)
            os.rename(temp_output, final_output)

        self._log("Processing completed successfully.")
        return True
