import os
import urllib.request

DEST = os.path.join(r"C:\Users\chrik\emotion_detection_system\models", "emotion-ferplus-8.onnx")
URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

def download_model():
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1000000:
        print(f"Model already exists at {DEST} ({os.path.getsize(DEST)} bytes).")
        return
    print(f"Downloading Emotion FER+ model ({URL}) ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(DEST, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 64
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"\rProgress: {downloaded}/{total} bytes ({downloaded/total*100:.1f}%)", end="", flush=True)
        print("\nDownload complete!")

if __name__ == "__main__":
    download_model()
