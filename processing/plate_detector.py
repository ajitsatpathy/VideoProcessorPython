import cv2
import os

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from utils.logger import logger

class PlateDetector:
    def __init__(self, model_path="models/yolo/yolov8n.pt"):
        self.use_yolo = False
        self.model = None
        self.cascade = None

        if HAS_YOLO:
            try:
                # We use yolov8n as a fallback, which detects cars. 
                # Ideally, point this to a fine-tuned license plate model.
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.model = YOLO(model_path)
                self.use_yolo = True
                logger.info(f"YOLO model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Falling back to Haar Cascades.")

        if not self.use_yolo:
            # Fallback to OpenCV Haar Cascade for Russian plates (commonly works as a generic plate detector)
            if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
                self.cascade = cv2.CascadeClassifier(cascade_path)
                if self.cascade.empty():
                    logger.error("Failed to load Haar Cascade for plate detection.")
                    self.cascade = None
            else:
                logger.warning("cv2.CascadeClassifier is not available in the installed OpenCV version. Plate blurring will be bypassed.")
                self.cascade = None

    def blur_plates(self, frame):
        """
        Detects and blurs license plates in the given frame.
        """
        if self.use_yolo and self.model:
            results = self.model(frame, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # By default YOLOv8n COCO dataset classes: 2 is car, 3 is motorcycle, 5 is bus, 7 is truck.
                    # A true plate model would predict only plates.
                    # Assuming we are using a custom plate model:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    self._apply_blur(frame, x1, y1, x2, y2)
        elif self.cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in plates:
                self._apply_blur(frame, x, y, x + w, y + h)

        return frame

    def _apply_blur(self, frame, x1, y1, x2, y2):
        try:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                frame[y1:y2, x1:x2] = blurred_roi
        except Exception:
            pass
