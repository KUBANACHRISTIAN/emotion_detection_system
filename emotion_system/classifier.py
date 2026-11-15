import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class EmotionResult:
    """Emotion classification output for a single face."""
    label: str
    confidence: float
    probabilities: Dict[str, float]


class EmotionClassifier:
    """
    Facial Emotion Classifier using a Deep Neural Network trained on FER2013 / FER+.
    Classifies faces into: Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear, Contempt.
    """

    EMOTIONS = [
        "Neutral",
        "Happy",
        "Surprise",
        "Sad",
        "Angry",
        "Disgust",
        "Fear",
        "Contempt"
    ]

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "emotion-ferplus-8.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Emotion model not found at {model_path}")

        self.model_path = model_path
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        # Prefer CPU target for universal portability
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def preprocess(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crops face with subtle padding, converts to grayscale, and resizes to 64x64.
        Returns 1x1x64x64 float32 blob.
        """
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]

        # Add 8% margin around bounding box to capture full facial expression features
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.08)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Convert to Grayscale if needed
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # Resize to model input specification: 64x64
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)

        # Shape: (1, 1, 64, 64)
        blob = resized.reshape(1, 1, 64, 64)
        return blob

    def classify(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[EmotionResult]:
        """Predicts emotion probabilities for a face bounding box."""
        blob = self.preprocess(frame, bbox)
        if blob is None:
            return None

        self.net.setInput(blob)
        logits = self.net.forward()

        # Softmax activation
        exp = np.exp(logits - np.max(logits))
        probs = (exp / np.sum(exp))[0]

        top_idx = int(np.argmax(probs))
        top_label = self.EMOTIONS[top_idx]
        top_conf = float(probs[top_idx])

        prob_dict = {
            self.EMOTIONS[i]: float(probs[i])
            for i in range(len(self.EMOTIONS))
        }

        return EmotionResult(
            label=top_label,
            confidence=top_conf,
            probabilities=prob_dict
        )
