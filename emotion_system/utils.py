import os
import urllib.request
from typing import Optional, Tuple
import cv2

MODELS = {
    "emotion-ferplus-8.onnx": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "face_detection_yunet_2023mar.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
}


def get_default_models_dir() -> str:
    base_models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    if os.path.exists(base_models):
        return base_models

    tmp_models = os.path.join("/tmp", "models")
    os.makedirs(tmp_models, exist_ok=True)
    return tmp_models


def ensure_models_exist(models_dir: Optional[str] = None):
    if models_dir is None:
        models_dir = get_default_models_dir()

    os.makedirs(models_dir, exist_ok=True)

    for filename, url in MODELS.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
            print(f"[Model Loader] Downloading missing asset {filename}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(filepath, "wb") as f:
                f.write(resp.read())
            print(f"[Model Loader] Downloaded {filename} successfully.")


def open_camera(preferred_index: int = 0, width: int = 1280, height: int = 720) -> Tuple[Optional[cv2.VideoCapture], int]:
    indices_to_try = [preferred_index] + [i for i in [0, 1, 2] if i != preferred_index]

    for idx in indices_to_try:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap, idx
            cap.release()

    return None, -1
