"""
Facial Emotion Recognition System (FER2013 / FER+)
Real-time emotion detection and classification using Deep Neural Networks.
"""

from .detector import FaceDetector, FaceBox
from .classifier import EmotionClassifier, EmotionResult
from .visualizer import EmotionVisualizer

__all__ = [
    "FaceDetector",
    "FaceBox",
    "EmotionClassifier",
    "EmotionResult",
    "EmotionVisualizer"
]
