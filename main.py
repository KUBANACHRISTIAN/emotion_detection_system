import argparse
import base64
import datetime
import os
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from emotion_system.detector import FaceDetector, FaceBox
from emotion_system.classifier import EmotionClassifier, EmotionResult
from emotion_system.visualizer import EmotionVisualizer
from emotion_system.utils import ensure_models_exist, open_camera, get_default_models_dir


class EmotionDetectionApp:
    """
    Main application engine coordinating face detection,
    expression preprocessing, FER+ neural classification, and real-time visualization.
    """

    def __init__(self, detector_mode: str = "dnn"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, "output")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError:
            pass

        ensure_models_exist()
        models_dir = get_default_models_dir()

        self.detector = FaceDetector(
            mode=detector_mode,
            yunet_path=os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"),
            cascade_path=os.path.join(models_dir, "haarcascade_frontalface_default.xml")
        )

        self.classifier = EmotionClassifier(
            model_path=os.path.join(models_dir, "emotion-ferplus-8.onnx")
        )

        self.visualizer = EmotionVisualizer()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, str]:
        """Detects faces, classifies emotions, and draws annotations."""
        faces = self.detector.detect(frame)
        annotated = frame.copy()

        dominant_emotion = "None"
        max_conf = -1.0

        for face in faces:
            emotion_res = self.classifier.classify(frame, face.bbox)
            if emotion_res is not None:
                if emotion_res.confidence > max_conf:
                    max_conf = emotion_res.confidence
                    dominant_emotion = emotion_res.label

                self.visualizer.draw_face_emotion(
                    frame=annotated,
                    face=face,
                    emotion=emotion_res,
                    draw_landmarks=(self.detector.mode == "dnn")
                )

        return annotated, len(faces), dominant_emotion

    def run_image(self, image_path: str, output_path: Optional[str] = None, show_gui: bool = True):
        if not os.path.exists(image_path):
            print(f"[Error] Image not found: {image_path}")
            return

        print(f"[Image Mode] Loading {image_path}...")
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Error] Failed to read {image_path}")
            return

        annotated, num_faces, dominant = self.process_frame(img)
        self.visualizer.draw_hud(
            frame=annotated,
            detector_name=self.detector.detector_name,
            fps=0.0,
            num_faces=num_faces,
            dominant_emotion=dominant
        )

        if output_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"emotion_{ts}.jpg")

        cv2.imwrite(output_path, annotated)
        print(f"[Result] Analyzed {num_faces} face(s). Dominant emotion: {dominant}. Saved: {output_path}")

        if show_gui:
            cv2.imshow("Emotion Detection (Press any key to close)", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run_video(self, video_path: str, output_path: Optional[str] = None, show_gui: bool = True):
        if not os.path.exists(video_path):
            print(f"[Error] Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Failed to open video: {video_path}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

        print(f"[Video Mode] Processing {video_path} ({total_frames} frames)...")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            annotated, num_faces, dominant = self.process_frame(frame)
            fps = self.visualizer.update_fps()

            self.visualizer.draw_hud(
                frame=annotated,
                detector_name=self.detector.detector_name,
                fps=fps,
                num_faces=num_faces,
                dominant_emotion=dominant
            )

            if writer is not None:
                writer.write(annotated)

            if show_gui:
                cv2.imshow("Emotion Video (Press Q to quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                elif key == ord('m'):
                    self.detector.toggle_mode()
                elif key == ord('b'):
                    self.visualizer.show_bars = not self.visualizer.show_bars

            if idx % 30 == 0 or idx == total_frames:
                print(f"  Frame {idx}/{total_frames} ({idx/total_frames*100:.1f}%)")

        cap.release()
        if writer is not None:
            writer.release()
            print(f"[Video Mode] Saved: {output_path}")
        cv2.destroyAllWindows()

    def run_live(self, camera_index: int = 0):
        cap, actual_idx = open_camera(camera_index)
        if cap is None:
            print(f"[Error] Could not open camera at index {camera_index}.")
            print("Tip: If you do not have a webcam, test on a sample photo using:")
            print("     python main.py --mode image --source test_samples/test_dual_faces.jpg")
            return

        print(f"[Live Feed] Camera #{actual_idx} running. Press 'q' to quit.")
        win_name = "Real-Time Emotion Detection"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, num_faces, dominant = self.process_frame(frame)
            fps = self.visualizer.update_fps()

            self.visualizer.draw_hud(
                frame=annotated,
                detector_name=self.detector.detector_name,
                fps=fps,
                num_faces=num_faces,
                dominant_emotion=dominant
            )

            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            elif key == ord('m'):
                new_mode = self.detector.toggle_mode()
                print(f"[Detector] Switched to {new_mode}")
            elif key == ord('b'):
                self.visualizer.show_bars = not self.visualizer.show_bars
            elif key == ord('s'):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(self.output_dir, f"snapshot_{ts}.jpg")
                cv2.imwrite(snap_path, annotated)
                self.visualizer.trigger_flash(f"Saved: {os.path.basename(snap_path)}")
                print(f"[Snapshot] Frame saved to {snap_path}")

        cap.release()
        cv2.destroyAllWindows()


# -------------------------------------------------------------
# Web Application Export (FastAPI) for Vercel / Cloud / Browser
# -------------------------------------------------------------
app = FastAPI(
    title="Facial Emotion Recognition API",
    description="Real-time facial expression and emotion classification using FER2013 / FER+ deep learning model",
    version="1.0.0"
)


@app.middleware("http")
async def fix_vercel_path_middleware(request, call_next):
    raw_path = request.scope.get("path", "")
    for prefix in ["/main.py", "/api/index.py", "/index.py"]:
        if raw_path == prefix:
            request.scope["path"] = "/"
            break
        elif raw_path.startswith(prefix + "/"):
            request.scope["path"] = raw_path[len(prefix):]
            break
    return await call_next(request)


_engine: Optional[EmotionDetectionApp] = None


def get_engine() -> EmotionDetectionApp:
    global _engine
    if _engine is None:
        _engine = EmotionDetectionApp()
    return _engine


class FramePayload(BaseModel):
    image: str


def decode_base64_image(data_url: str) -> Optional[np.ndarray]:
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.get("/", response_class=HTMLResponse)
@app.get("/main.py", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
async def serve_index():
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Facial Emotion Recognition API</h1><p>Visit /docs for API specs.</p>")


@app.get("/health")
@app.get("/api/health")
@app.get("/main.py/health")
async def health_check():
    engine = get_engine()
    return {
        "status": "healthy",
        "detector": engine.detector.detector_name,
        "classes": EmotionClassifier.EMOTIONS
    }


@app.post("/api/detect_frame")
async def detect_frame_api(payload: FramePayload):
    frame = decode_base64_image(payload.image)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image payload"})

    engine = get_engine()
    faces = engine.detector.detect(frame)
    faces_data = []
    dominant_overall = "None"
    max_c = -1.0

    for face in faces:
        res = engine.classifier.classify(frame, face.bbox)
        if res is not None:
            if res.confidence > max_c:
                max_c = res.confidence
                dominant_overall = res.label

            faces_data.append({
                "bbox": list(face.bbox),
                "label": res.label,
                "confidence": float(res.confidence),
                "probabilities": res.probabilities
            })

    return {
        "faces": faces_data,
        "count": len(faces_data),
        "dominant_emotion": dominant_overall
    }


@app.post("/api/annotate_image")
async def annotate_uploaded_image(file: UploadFile = File(...)):
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

    engine = get_engine()
    annotated, _, _ = engine.process_frame(img)
    _, buffer = cv2.imencode(".jpg", annotated)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


# -------------------------------------------------------------
# Desktop CLI Entry Point
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Facial Emotion Detection System (FER2013 / FER+)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode", type=str, choices=["live", "image", "video", "web"], default="live",
        help="Operating mode:\n  live  - real-time desktop webcam feed\n  image - process still photo\n  video - process prerecorded video\n  web   - launch local web server"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Input source:\n  Camera index (e.g. 0) for live mode\n  Path to image or video file for file modes"
    )
    parser.add_argument(
        "--detector", type=str, choices=["dnn", "haar"], default="dnn",
        help="Detection algorithm: 'dnn' (YuNet, recommended) or 'haar' (Haar Cascade)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path for processed image or video"
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Port for local web server mode"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Headless execution without GUI display window"
    )

    args = parser.parse_args()

    if args.mode == "web":
        import uvicorn
        print(f"[Web Server] Starting local emotion web dashboard on http://localhost:{args.port} ...")
        uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)
        return

    cli_app = EmotionDetectionApp(detector_mode=args.detector)

    if args.mode == "live":
        try:
            cam_idx = int(args.source)
        except ValueError:
            cam_idx = 0
        cli_app.run_live(camera_index=cam_idx)
    elif args.mode == "image":
        cli_app.run_image(image_path=args.source, output_path=args.output, show_gui=not args.no_gui)
    elif args.mode == "video":
        cli_app.run_video(video_path=args.source, output_path=args.output, show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
