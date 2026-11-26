import os
import urllib.request
import cv2
import numpy as np

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

DEFAULT_PLATE_MODEL_PATH = "models/yolo/license_plate_detector.pt"
FALLBACK_COCO_MODEL_PATH = "models/yolo/yolov8n.pt"
REMOTE_PLATE_MODEL_URL = (
    "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
)

class PlateDetector:
    def __init__(self, model_path=None):
        self.use_yolo = False
        self.model = None
        self.is_plate_model = False
        self.plate_classes = []
        self.vehicle_classes = []
        self.cascade = None

        resolved_model_path = self._resolve_model_path(model_path)

        if HAS_YOLO and resolved_model_path:
            try:
                os.makedirs(os.path.dirname(resolved_model_path), exist_ok=True)
                self.model = YOLO(resolved_model_path)
                self.use_yolo = True
                self._configure_model_classes()
                logger.info(
                    f"YOLO model loaded from {resolved_model_path} "
                    f"(is_plate_model={self.is_plate_model})"
                )
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Falling back to secondary detection methods.")

        if not self.use_yolo:
            self._init_cascade()

    def _resolve_model_path(self, model_path=None):
        """
        Determines the appropriate YOLO model path.
        Prioritizes dedicated license plate models, auto-downloading if missing,
        and falls back to standard vehicle model or specified path.
        """
        if model_path:
            return model_path

        candidates = [
            DEFAULT_PLATE_MODEL_PATH,
            "models/yolo/plate.pt",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        # Attempt to auto-download dedicated license plate model
        try:
            os.makedirs(os.path.dirname(DEFAULT_PLATE_MODEL_PATH), exist_ok=True)
            logger.info(f"Downloading license plate detection model to {DEFAULT_PLATE_MODEL_PATH}...")
            req = urllib.request.Request(
                REMOTE_PLATE_MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(DEFAULT_PLATE_MODEL_PATH, "wb") as f:
                f.write(data)
            logger.info("License plate detection model downloaded successfully.")
            return DEFAULT_PLATE_MODEL_PATH
        except Exception as e:
            logger.warning(f"Failed to auto-download license plate model: {e}. Checking fallback models.")

        if os.path.exists(FALLBACK_COCO_MODEL_PATH):
            return FALLBACK_COCO_MODEL_PATH

        return DEFAULT_PLATE_MODEL_PATH

    def _configure_model_classes(self):
        """
        Inspects model classes to determine whether it detects license plates directly
        or vehicles (requiring plate localization within the vehicle region).
        """
        if not self.model or not hasattr(self.model, "names") or not self.model.names:
            self.is_plate_model = True
            return

        names = self.model.names
        plate_keywords = ("plate", "license", "licence")
        vehicle_keywords = ("car", "truck", "bus", "motorcycle", "vehicle")

        self.plate_classes = [
            cls_id for cls_id, name in names.items()
            if any(kw in str(name).lower() for kw in plate_keywords)
        ]

        self.vehicle_classes = [
            cls_id for cls_id, name in names.items()
            if any(kw in str(name).lower() for kw in vehicle_keywords)
        ]

        if self.plate_classes:
            self.is_plate_model = True
        elif len(names) == 1 and not self.vehicle_classes:
            # Single-class model is typically a custom license plate detector
            self.is_plate_model = True
            self.plate_classes = list(names.keys())
        else:
            self.is_plate_model = False

    def _init_cascade(self):
        # Fallback to OpenCV Haar Cascade for Russian plates if available
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
            try:
                self.cascade = cv2.CascadeClassifier(cascade_path)
                if self.cascade.empty():
                    logger.error("Failed to load Haar Cascade for plate detection.")
                    self.cascade = None
            except Exception as e:
                logger.warning(f"Error initializing Haar Cascade: {e}")
                self.cascade = None
        else:
            logger.warning("cv2.CascadeClassifier is not available in the installed OpenCV version. Plate blurring will be bypassed.")
            self.cascade = None

    def blur_plates(self, frame):
        """
        Detects and blurs only license plates in the given frame.
        Ensures the complete vehicle is never blurred.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        plates_detected = False

        if self.use_yolo and self.model:
            try:
                results = self.model(frame, verbose=False)
                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue

                    if self.is_plate_model:
                        # Model is a dedicated license plate detector: boxes are plates
                        for box in boxes:
                            cls_id = int(box.cls[0]) if box.cls is not None else 0
                            conf = float(box.conf[0]) if box.conf is not None else 1.0
                            if conf < 0.25:
                                continue
                            if not self.plate_classes or cls_id in self.plate_classes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                self._apply_blur(frame, x1, y1, x2, y2)
                                plates_detected = True
                    else:
                        # Model is a vehicle detector (e.g., COCO): boxes are vehicles.
                        # NEVER blur the whole vehicle. Locate plate within vehicle ROI.
                        for box in boxes:
                            cls_id = int(box.cls[0]) if box.cls is not None else 0
                            conf = float(box.conf[0]) if box.conf is not None else 1.0
                            if conf < 0.35:
                                continue
                            if not self.vehicle_classes or cls_id in self.vehicle_classes:
                                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                                h, w = frame.shape[:2]
                                vx1, vy1 = max(0, vx1), max(0, vy1)
                                vx2, vy2 = min(w, vx2), min(h, vy2)
                                if vx2 <= vx1 or vy2 <= vy1:
                                    continue

                                vehicle_roi = frame[vy1:vy2, vx1:vx2]
                                plate_coords = self._locate_plates_in_vehicle_roi(vehicle_roi)
                                for px1, py1, px2, py2 in plate_coords:
                                    self._apply_blur(frame, vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
                                    plates_detected = True
            except Exception as e:
                logger.warning(f"Error during YOLO plate detection: {e}")

        # Secondary fallback: Haar Cascade if available and YOLO didn't detect plates
        if not plates_detected and self.cascade:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                plates = self.cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                for (x, y, w, h) in plates:
                    self._apply_blur(frame, x, y, x + w, y + h)
            except Exception as e:
                logger.warning(f"Haar Cascade detection error: {e}")

        return frame

    def _locate_plates_in_vehicle_roi(self, vehicle_roi):
        """
        Locates license plate region(s) within a vehicle ROI using classical ALPR techniques
        (gradient analysis, morphological operations, aspect ratio and area filtering).
        Returns coordinates [(px1, py1, px2, py2), ...] relative to vehicle_roi.
        """
        vh, vw = vehicle_roi.shape[:2]
        if vh < 25 or vw < 25:
            return []

        # Plates are typically located in the lower 60% of the vehicle
        start_y = int(vh * 0.40)
        roi = vehicle_roi[start_y:, :]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 15 or roi_w < 20:
            return []

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Bilateral filter reduces noise while preserving vertical edges of plate characters
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            # Sobel horizontal gradient to emphasize vertical text edges
            grad_x = cv2.Sobel(filtered, cv2.CV_16S, 1, 0, ksize=3)
            grad_x = cv2.convertScaleAbs(grad_x)
            # Gaussian blur and Otsu thresholding
            blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Morphological close to bridge license plate characters into a single rectangular block
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # Find candidate contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates = []
            vehicle_area = vh * vw
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / max(h, 1)
                area = w * h
                # License plates typically have aspect ratio 1.8 to 6.0 and reasonable proportion of vehicle
                if 1.8 <= aspect_ratio <= 6.0 and 0.003 * vehicle_area <= area <= 0.25 * vehicle_area:
                    # Margin padding
                    px1 = max(0, x - 2)
                    py1 = max(0, start_y + y - 2)
                    px2 = min(vw, x + w + 2)
                    py2 = min(vh, start_y + y + h + 2)
                    candidates.append((px1, py1, px2, py2))

            return candidates
        except Exception:
            return []

    def _apply_blur(self, frame, x1, y1, x2, y2):
        try:
            h, w = frame.shape[:2]
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w))
            y2 = max(0, min(int(y2), h))

            if x2 <= x1 or y2 <= y1:
                return

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                roi_h, roi_w = roi.shape[:2]
                kw = max(3, (roi_w // 3) * 2 + 1)
                kh = max(3, (roi_h // 3) * 2 + 1)
                kw = min(kw, 51)
                kh = min(kh, 51)
                blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 0)
                frame[y1:y2, x1:x2] = blurred_roi
        except Exception:
            pass
