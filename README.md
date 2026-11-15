# Real-Time Facial Emotion Detection System

A high-performance Computer Vision and Deep Learning system capable of detecting human faces and classifying emotional expressions in real time using **Deep Convolutional Neural Networks (FER2013 / FER+)** and dual face detectors (**DNN YuNet** & **Haar Cascades**).

---

## Features & Highlights

- **8 Emotion Categories**:
  - **Happy**, **Surprise**, **Neutral**, **Sad**, **Angry**, **Fear**, **Disgust**, and **Contempt**.
- **Dual Detection Engine**:
  - **DNN (YuNet)**: State-of-the-art fast face detection with 5-point landmark tracking.
  - **Haar Cascades**: Classical Viola-Jones cascade with histogram equalization.
  - **Hot-swappable**: Toggle between DNN and Haar Cascade at runtime with hotkey `m`.
- **Deep Neural Network Classification**:
  - Pretrained on the standard **FER2013 / FER+** benchmark dataset.
  - Preprocesses face crops to $64 \times 64$ grayscale with margin padding.
  - Generates full softmax probability distribution across all 8 emotional states.
- **Biometric HUD & Real-Time Visualization**:
  - Color-coded bounding boxes mapped to the detected emotion (e.g. Gold for Happy, Cyan for Surprise, Slate for Neutral, Red for Angry).
  - Floating label badges displaying emotion name and confidence percentage.
  - Live breakdown mini-bar charts showing secondary emotions (toggleable with `b`).
  - Real-time FPS telemetry and dominant emotion tracker.
- **Dual Interface (Desktop & Browser Web App)**:
  - **Desktop CLI**: Low-latency desktop webcam stream (`cv2.VideoCapture`).
  - **Web Dashboard**: Interactive HTML5 browser webcam interface and REST API built with **FastAPI** (Vercel-ready).

---

## Project Structure

```
emotion_detection_system/
├── models/                               # Pretrained Deep Learning Models
│   ├── emotion-ferplus-8.onnx            # FER+ Emotion Classifier ONNX
│   ├── face_detection_yunet_2023mar.onnx # YuNet Face Detector ONNX
│   └── haarcascade_frontalface_default.xml # Haar Cascade XML
├── emotion_system/                       # Core Package
│   ├── __init__.py
│   ├── detector.py                       # FaceDetector (DNN YuNet + Haar)
│   ├── classifier.py                     # EmotionClassifier (FER+ & Softmax)
│   ├── visualizer.py                     # Biometric HUD & probability bars
│   └── utils.py                          # Camera & model loading helpers
├── templates/
│   └── index.html                        # Interactive browser web dashboard
├── test_samples/                         # Sample test portraits
├── output/                               # Saved screenshots & annotated results
├── api/
│   └── index.py                          # Vercel entrypoint
├── main.py                               # CLI runner & FastAPI web application
├── vercel.json                           # Vercel serverless configuration
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## Quickstart Guide

### 1. Installation

```bash
cd C:\Users\chrik\emotion_detection_system
pip install -r requirements.txt
```

### 2. Live Desktop Webcam Mode

Launch real-time emotion detection through your webcam:

```bash
python main.py --mode live
```

#### Interactive Controls:
| Key | Action |
|---|---|
| `m` | Switch detector between **DNN (YuNet)** and **Haar Cascade** |
| `b` | Toggle emotion probability breakdown bars on/off |
| `s` | Capture and save snapshot to `output/snapshot_<timestamp>.jpg` |
| `q` / `ESC` | Exit application |

---

### 3. Process Still Photos

Analyze emotions on any photo file:

```bash
python main.py --mode image --source test_samples/test_dual_faces.jpg
```

Save output to custom path without opening a GUI window:
```bash
python main.py --mode image --source photo.jpg --output output/emotion_result.jpg --no-gui
```

---

### 4. Process Prerecorded Videos

Process a video file and save the annotated output video:

```bash
python main.py --mode video --source input_video.mp4 --output output/analyzed_video.mp4
```

---

### 5. Launch Browser Web Dashboard

Start the interactive web dashboard on your local browser:

```bash
python main.py --mode web --port 8001
```
Then open `http://localhost:8001` in your browser.

---

## Technical Details

### Preprocessing Pipeline:
1. **Face Localization**: Face bounding boxes are extracted using YuNet DNN or Haar Cascades.
2. **Padding**: A margin of $8\%$ is added around the bounding box to capture subtle mouth curves, cheek contours, and forehead tension.
3. **Grayscale Conversion**: Image is converted from BGR to single-channel 8-bit grayscale.
4. **Resolution Normalization**: Cropped face is resized to $64 \times 64$ pixels using area interpolation.
5. **Tensor Formatting**: Formatted into a 4D tensor with shape `(1, 1, 64, 64)` float32.

### Softmax Activation & Classification:
The 8 raw logits $\mathbf{z} = [z_0, z_1, \dots, z_7]$ are transformed into probabilities:

$$P(\text{emotion}_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=0}^{7} e^{z_j - \max(\mathbf{z})}}$$

The emotion with the highest probability is assigned as the dominant expression.
