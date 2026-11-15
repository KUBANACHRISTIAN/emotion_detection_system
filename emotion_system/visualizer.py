import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .classifier import EmotionResult
from .detector import FaceBox


class EmotionVisualizer:
    """
    Renders biometric emotion bounding boxes, badges, landmark points,
    probability breakdown bar charts, and real-time HUD telemetry.
    """

    # BGR Color palette mapped to emotions
    EMOTION_COLORS = {
        "Happy": (0, 215, 255),       # Gold / Amber
        "Surprise": (255, 210, 0),    # Cyan
        "Neutral": (220, 180, 100),   # Cool Slate
        "Sad": (235, 120, 80),        # Muted Blue
        "Angry": (60, 60, 240),       # Crimson Red
        "Fear": (180, 50, 180),       # Violet
        "Disgust": (60, 160, 60),     # Forest Green
        "Contempt": (80, 140, 200)    # Ochre
    }

    EMOTION_EMOJIS = {
        "Happy": ":D",
        "Surprise": ":O",
        "Neutral": ":|",
        "Sad": ":(",
        "Angry": ">:(",
        "Fear": "D:",
        "Disgust": ":P",
        "Contempt": ":/"
    }

    COLOR_HUD_BG = (20, 20, 24)
    COLOR_TEXT = (250, 250, 250)
    COLOR_MUTED = (160, 160, 160)

    def __init__(self):
        self._fps_history = []
        self._last_time = time.time()
        self._flash_counter = 0
        self._notification = ""
        self._notification_expiry = 0.0
        self.show_bars = True

    def update_fps(self) -> float:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self._fps_history.append(1.0 / dt)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            return float(np.mean(self._fps_history))
        return 0.0

    def trigger_flash(self, message: str = "Snapshot Saved!"):
        self._flash_counter = 4
        self._notification = message
        self._notification_expiry = time.time() + 2.5

    def draw_hud(
        self,
        frame: np.ndarray,
        detector_name: str,
        fps: float,
        num_faces: int,
        dominant_emotion: str
    ):
        """Draws top telemetry bar and bottom control guide."""
        h, w = frame.shape[:2]

        # Top banner overlay
        header_h = 42
        top_bar = frame.copy()
        cv2.rectangle(top_bar, (0, 0), (w, header_h), self.COLOR_HUD_BG, -1)
        cv2.addWeighted(top_bar, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, header_h), (w, header_h), (50, 50, 55), 1)

        fps_text = f"FPS: {fps:4.1f}"
        cv2.putText(frame, fps_text, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

        det_text = f"Detector: {detector_name}"
        cv2.putText(frame, det_text, (130, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_TEXT, 1, cv2.LINE_AA)

        status_text = f"Faces: {num_faces}   |   Dominant: {dominant_emotion}"
        cv2.putText(frame, status_text, (w - 380, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # Bottom footer overlay
        footer_h = 30
        bottom_bar = frame.copy()
        cv2.rectangle(bottom_bar, (0, h - footer_h), (w, h), self.COLOR_HUD_BG, -1)
        cv2.addWeighted(bottom_bar, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, h - footer_h), (w, h - footer_h), (50, 50, 55), 1)

        guide = "[M] Switch Detector   [B] Toggle Probability Bars   [S] Snapshot   [Q] Quit"
        cv2.putText(frame, guide, (16, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # Flash effect
        if self._flash_counter > 0:
            white = np.full_like(frame, 255)
            alpha = self._flash_counter / 8.0
            cv2.addWeighted(white, alpha, frame, 1.0 - alpha, 0, frame)
            self._flash_counter -= 1

        # Notification banner
        if time.time() < self._notification_expiry and self._notification:
            ts, _ = cv2.getTextSize(self._notification, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = (w - ts[0]) // 2
            by = 75
            cv2.rectangle(frame, (bx - 12, by - ts[1] - 8), (bx + ts[0] + 12, by + 8), (30, 30, 30), -1)
            cv2.rectangle(frame, (bx - 12, by - ts[1] - 8), (bx + ts[0] + 12, by + 8), (0, 215, 255), 1)
            cv2.putText(frame, self._notification, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_face_emotion(
        self,
        frame: np.ndarray,
        face: FaceBox,
        emotion: EmotionResult,
        draw_landmarks: bool = True
    ):
        """Draws stylized emotion box, label badge, and probability bars."""
        x, y, w, h = face.bbox
        color = self.EMOTION_COLORS.get(emotion.label, (200, 200, 200))

        # Translucent face box highlight
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)

        # Main bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

        # Stylized corner brackets
        corner_len = max(10, int(min(w, h) * 0.16))
        thick = 3
        cv2.line(frame, (x, y), (x + corner_len, y), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thick, cv2.LINE_AA)

        # Landmarks (if available)
        if draw_landmarks and face.landmarks:
            for lx, ly in face.landmarks:
                cv2.circle(frame, (int(round(lx)), int(round(ly))), 3, color, -1, cv2.LINE_AA)

        # Emotion Label Badge
        pct = emotion.confidence * 100.0
        label_text = f"{emotion.label.upper()} {pct:4.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        ts, _ = cv2.getTextSize(label_text, font, font_scale, 1)

        badge_h = ts[1] + 12
        badge_w = ts[0] + 14
        badge_y1 = y - badge_h - 4
        badge_y2 = y - 4
        if badge_y1 < 48:
            badge_y1 = y + 4
            badge_y2 = y + 4 + badge_h

        cv2.rectangle(frame, (x, badge_y1), (x + badge_w, badge_y2), color, -1)
        # Dark text for high-brightness colors, white for dark colors
        text_color = (15, 15, 15) if emotion.label in ("Happy", "Neutral", "Surprise") else (255, 255, 255)
        cv2.putText(frame, label_text, (x + 7, badge_y2 - 5), font, font_scale, text_color, 1, cv2.LINE_AA)

        # Probability breakdown bars (if enabled)
        if self.show_bars:
            self._draw_probability_bars(frame, face, emotion)

    def _draw_probability_bars(self, frame: np.ndarray, face: FaceBox, emotion: EmotionResult):
        """Draws top-4 emotion probability breakdown bars next to the face."""
        x, y, w, h = face.bbox
        h_frame, w_frame = frame.shape[:2]

        sorted_emotions = sorted(emotion.probabilities.items(), key=lambda item: item[1], reverse=True)[:4]

        bar_x = x + w + 12
        bar_w = 90
        bar_h = 10
        spacing = 16

        if bar_x + bar_w + 50 > w_frame:
            bar_x = max(10, x - bar_w - 60)

        start_y = max(50, y)

        # Background panel for bars
        total_panel_h = len(sorted_emotions) * spacing + 10
        panel_overlay = frame.copy()
        cv2.rectangle(
            panel_overlay,
            (bar_x - 6, start_y - 6),
            (bar_x + bar_w + 54, start_y + total_panel_h),
            (16, 16, 20),
            -1
        )
        cv2.addWeighted(panel_overlay, 0.75, frame, 0.25, 0, frame)

        for i, (name, prob) in enumerate(sorted_emotions):
            curr_y = start_y + i * spacing + 4
            emo_color = self.EMOTION_COLORS.get(name, (200, 200, 200))

            # Emotion name label
            cv2.putText(
                frame, f"{name[:3]}", (bar_x, curr_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT, 1, cv2.LINE_AA
            )

            # Bar background
            bx = bar_x + 28
            cv2.rectangle(frame, (bx, curr_y), (bx + bar_w, curr_y + bar_h), (40, 40, 45), -1)

            # Filled bar
            fill_w = int(bar_w * prob)
            if fill_w > 0:
                cv2.rectangle(frame, (bx, curr_y), (bx + fill_w, curr_y + bar_h), emo_color, -1)

            # Percentage text
            cv2.putText(
                frame, f"{int(prob * 100)}%", (bx + bar_w + 4, curr_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.COLOR_MUTED, 1, cv2.LINE_AA
            )
