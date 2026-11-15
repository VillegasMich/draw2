"""Regenerate the golden fixtures for draw-core's detection stage.

Runs the pinned `ygo_yolo.onnx` (the exact file listed in `ModelManifest::PINNED`)
through ultralytics on `docs/samples/sample_photo.jpg` and on a few frames of
`docs/samples/sample_video.mp4`, and writes every image's boxes (four corners in
source-frame pixels, plus the detector confidence) to `detect/golden.json`.

It also writes `detect/resize.json`: OpenCV `INTER_LINEAR` resizes of a small synthetic
image, the reference for draw-core's hand-written letterbox resize. Ultralytics (and so
the model's training) letterboxes with exactly that resize.

The video frames are stored as JPEG files next to the JSON, and ultralytics reads them
back from disk, so the Rust golden test and the reference decode the same bytes.

How to regenerate (from the repo root, in a throwaway environment that is never
committed; `.venv-fixtures/` and `.models/` are git-ignored):

    uv venv --python 3.12 .venv-fixtures
    VIRTUAL_ENV=.venv-fixtures uv pip install ultralytics onnx onnxruntime opencv-python
    mkdir -p .models
    curl -fL -o .models/ygo_yolo.onnx \
      https://huggingface.co/HichTala/draw2/resolve/<PINNED commit>/onnx/ygo_yolo.onnx
    .venv-fixtures/bin/python -I crates/draw-core/tests/fixtures/gen_detect.py .models

`onnx` is listed so ultralytics does not try to auto-install it on first run. Then run
the golden test against the new fixtures:

    DRAW_CORE_MODELS_DIR=.models cargo test -p draw-core golden -- --nocapture

The settings below mirror draw-core's detector: a 640x640 letterbox, confidence
threshold 0.20 (`RecognizerOptions::min_detector_confidence`) and NMS at IoU 0.5.
Ultralytics differs in details that the test's IoU tolerance absorbs: it pads with gray
(114) instead of black, resizes with OpenCV, and runs rotated (probabilistic) IoU NMS
instead of the axis-aligned approximation.
"""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import ultralytics
from ultralytics import YOLO

# Must match `DETECTOR.sha256` in crates/draw-core/src/manifest.rs.
MODEL_SHA256 = "449e0d42cbf8429b170abf44d50cda29beeeb05dc80890102e12c06d6358cf97"
MODEL_FILE = "ygo_yolo.onnx"

IMGSZ = 640
CONF = 0.20
IOU = 0.5

# Video frames to extract: a box rotated ~90°, ten cards with one at ~90°, and eleven
# cards with a partly visible one (confidence ~0.48).
VIDEO_FRAMES = (0, 540, 600)
JPEG_QUALITY = 90

FIXTURES = Path(__file__).resolve().parent
REPO_ROOT = FIXTURES.parents[3]
OUT_DIR = FIXTURES / "detect"
SAMPLES = REPO_ROOT / "docs" / "samples"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_frames() -> list[Path]:
    cap = cv2.VideoCapture(str(SAMPLES / "sample_video.mp4"))
    paths = []
    for index in VIDEO_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            sys.exit(f"could not read frame {index} of sample_video.mp4")
        path = OUT_DIR / f"video_frame_{index:04d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        paths.append(path)
    cap.release()
    return paths


def boxes_for(model: YOLO, path: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        sys.exit(f"could not read {path}")
    result = model.predict(
        image, imgsz=IMGSZ, conf=CONF, iou=IOU, device="cpu", verbose=False
    )[0]
    corners = result.obb.xyxyxyxy.tolist()
    confidences = result.obb.conf.tolist()
    height, width = image.shape[:2]
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "width": width,
        "height": height,
        "boxes": [
            {
                "corners": [[round(x, 2), round(y, 2)] for x, y in quad],
                "confidence": round(confidence, 4),
            }
            for quad, confidence in zip(corners, confidences)
        ],
    }


# A small RGB image with sharp and smooth regions, resized down and up by ratios that
# are not whole numbers.
RESIZE_SOURCE = (7, 5)
RESIZE_TARGETS = ((3, 2), (11, 8))


def resize_reference() -> dict:
    width, height = RESIZE_SOURCE
    y, x, c = np.indices((height, width, 3))
    rgb = ((x * 37 + y * 91 + c * 53) % 256).astype(np.uint8)
    return {
        "source": {"width": width, "height": height, "rgb": rgb.flatten().tolist()},
        "resized": [
            {
                "width": w,
                "height": h,
                "rgb": cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                .flatten()
                .tolist(),
            }
            for w, h in RESIZE_TARGETS
        ],
    }


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <models dir containing {MODEL_FILE}>")
    model_path = Path(sys.argv[1]) / MODEL_FILE
    actual = sha256(model_path)
    if actual != MODEL_SHA256:
        sys.exit(f"{model_path}: sha256 {actual}, expected {MODEL_SHA256}")

    OUT_DIR.mkdir(exist_ok=True)
    resize = OUT_DIR / "resize.json"
    resize.write_text(json.dumps(resize_reference()) + "\n")
    print(f"wrote {resize.relative_to(REPO_ROOT)}")

    model = YOLO(str(model_path), task="obb")
    images = [SAMPLES / "sample_photo.jpg", *extract_frames()]
    golden = {
        "generator": "crates/draw-core/tests/fixtures/gen_detect.py",
        "ultralytics": ultralytics.__version__,
        "onnxruntime": onnxruntime.__version__,
        "model": {"file": MODEL_FILE, "sha256": MODEL_SHA256},
        "settings": {"imgsz": IMGSZ, "conf": CONF, "iou": IOU},
        "images": [boxes_for(model, path) for path in images],
    }
    out = OUT_DIR / "golden.json"
    out.write_text(json.dumps(golden, indent=1) + "\n")
    for image in golden["images"]:
        print(f"{image['path']}: {len(image['boxes'])} boxes")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
