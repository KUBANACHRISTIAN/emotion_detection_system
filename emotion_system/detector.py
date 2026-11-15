import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FaceBox:
    """Represents a localized face."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None


class FaceDetector:
    """
    Unified face detector supporting both Deep Neural Network (YuNet)
    and Haar Feature Cascade Classifiers.
    """

    def __init__(
        self,
        mode: str = "dnn",
        yunet_path: Optional[str] = None,
        cascade_path: Optional[str] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")

        if yunet_path is None:
            yunet_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        if cascade_path is None:
            cascade_path = os.path.join(models_dir, "haarcascade_frontalface_default.xml")

        self.yunet_path = yunet_path
        self.cascade_path = cascade_path
        self.mode = mode.lower()

        # Initialize YuNet if file exists
        self.yunet = None
        if os.path.exists(self.yunet_path):
            self.yunet = cv2.FaceDetectorYN.create(
                self.yunet_path, "", (320, 320), score_threshold=0.6, nms_threshold=0.3
            )
            self._current_size = (320, 320)

        # Initialize Haar Cascade
        self.cascade = None
        if os.path.exists(self.cascade_path):
            self.cascade = cv2.CascadeClassifier(self.cascade_path)
        elif hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            builtin = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(builtin):
                self.cascade = cv2.CascadeClassifier(builtin)

        # Fallback if preferred detector missing
        if self.mode == "dnn" and self.yunet is None:
            self.mode = "haar"
        elif self.mode == "haar" and self.cascade is None:
            self.mode = "dnn"

    @property
    def detector_name(self) -> str:
        return "DNN (YuNet)" if self.mode == "dnn" else "Haar Cascade"

    def toggle_mode(self) -> str:
        """Switch between DNN and Haar Cascade at runtime."""
        if self.mode == "dnn" and self.cascade is not None:
            self.mode = "haar"
        elif self.mode == "haar" and self.yunet is not None:
            self.mode = "dnn"
        return self.detector_name

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """Detects faces in frame and returns list of FaceBox."""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        if self.mode == "dnn" and self.yunet is not None:
            if self._current_size != (w, h):
                self._current_size = (w, h)
                self.yunet.setInputSize((w, h))

            _, faces = self.yunet.detect(frame)
            if faces is None or len(faces) == 0:
                return []

            results = []
            for face in faces:
                x, y, bw, bh = face[0:4].astype(int)
                score = float(face[14])

                # Clamp to frame dimensions
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                landmarks = [
                    (float(face[4]), float(face[5])),
                    (float(face[6]), float(face[7])),
                    (float(face[8]), float(face[9])),
                    (float(face[10]), float(face[11])),
                    (float(face[12]), float(face[13]))
                ]
                results.append(FaceBox(bbox=(x, y, bw, bh), confidence=score, landmarks=landmarks))
            return results

        elif self.cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            results = []
            for (x, y, bw, bh) in faces:
                results.append(FaceBox(bbox=(int(x), int(y), int(bw), int(bh)), confidence=0.90))
            return results

        return []
