"""Regenerate the reference fixtures for draw-core's crop and classification stages.

For every reference box of the detection fixtures (`detect/golden.json`, written by
`gen_detect.py`), this script makes the card's 224x224 crop with OpenCV, picks its
orientation, runs the pinned `vit_int8.onnx` (the exact file listed in
`ModelManifest::PINNED`) through onnxruntime on it, and writes:

- `classify/crop_<image>_<box>.png`: the upright crop, lossless RGB;
- `classify/golden.json`: per crop, the quad it was warped from, the rotation chosen,
  and the top-k candidates (passcode and score).

The crops follow draw2's browser pipeline (`docs/scripts/pipeline.js`), the reference
for draw-core:

- The quad is put in the order draw-core's detector gives: clockwise on screen, with
  the first edge a short side of the card (its portrait frame). Corner 0 goes to the
  crop's top-left, then top-right, bottom-right, bottom-left.
- `warpPerspective` in pipeline.js samples each crop pixel at its center
  (`dx + 0.5, dy + 0.5`) and maps the quad's corners to the crop's outer edges (0 and
  224). OpenCV samples crop pixels at whole coordinates instead, so the same mapping
  in OpenCV's convention sends the corners to -0.5 and 223.5. Without that shift the
  crops differ from pipeline.js by 3 to 6 levels on average.
- `correctRotation`: 0° or 180°, 180° when the crop's top band is brighter than its
  bottom band. If the top-1 score is below 0.15, the other rotation is classified too
  and kept only if its top-1 is at least 1.25x better.
- `preprocessViT`: RGB / 255, normalized with mean = std = 0.5, NCHW.
- `topK`: softmax only when the outputs are logits (their max is above 1).

How to regenerate (from the repo root, in the throwaway environment of `gen_detect.py`;
`.venv-fixtures/` and `.models/` are git-ignored):

    uv venv --python 3.12 .venv-fixtures
    VIRTUAL_ENV=.venv-fixtures uv pip install onnxruntime opencv-python numpy
    mkdir -p .models
    for f in vit_int8.onnx cardnames_onnx.json; do
      curl -fL -o .models/$f \
        https://huggingface.co/HichTala/draw2/resolve/7a690eecf6aef5c812ba9a5a1d3e1ced4968b5b3/onnx/$f
    done
    .venv-fixtures/bin/python -I crates/draw-core/tests/fixtures/gen_classify.py .models

The URLs are `ModelManifest::PINNED.url(...)` for the int8 classifier and its labels;
if the manifest moves to another commit, update them here and in the SHA-256 values
below.

Then run the tests against the new fixtures:

    DRAW_CORE_MODELS_DIR=$PWD/.models cargo test -p draw-core -- --nocapture
"""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

# Must match the int8 entries in crates/draw-core/src/manifest.rs.
MODEL_FILE = "vit_int8.onnx"
MODEL_SHA256 = "f2da9ce6b2c02498e0c9f8fc0b7817eeebd88bfe8491c8c7f83055dfd57e5b4a"
LABELS_FILE = "cardnames_onnx.json"
LABELS_SHA256 = "70d13d586eee498b33ee493b48cd564fbffcb99b0476897978de1fd7d856f639"

CROP_SIZE = 224
# Deep enough for deck filtering, as `RecognizerOptions::top_k` and the Python pipeline.
TOP_K = 15
ROTATION_FALLBACK_CONFIDENCE = 0.15
ROTATION_FLIP_MARGIN = 1.25

FIXTURES = Path(__file__).resolve().parent
REPO_ROOT = FIXTURES.parents[3]
DETECT_GOLDEN = FIXTURES / "detect" / "golden.json"
OUT_DIR = FIXTURES / "classify"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(models_dir: Path, file: str, expected: str) -> Path:
    path = models_dir / file
    actual = sha256(path)
    if actual != expected:
        sys.exit(f"{path}: sha256 {actual}, expected {expected}")
    return path


def portrait_quad(corners: list) -> np.ndarray:
    """The corners in draw-core's detector order: clockwise on screen (y down), the
    first edge a short side."""
    quad = np.array(corners, dtype=np.float64)
    x, y = quad[:, 0], quad[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        quad = quad[::-1]
    if np.linalg.norm(quad[1] - quad[0]) > np.linalg.norm(quad[2] - quad[1]):
        quad = np.roll(quad, -1, axis=0)
    return quad


def warp(rgb: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """pipeline.js `warpPerspective`, in OpenCV's pixel convention (see the module
    docstring)."""
    edge = CROP_SIZE - 0.5
    destination = np.array(
        [[-0.5, -0.5], [edge, -0.5], [edge, edge], [-0.5, edge]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(
        rgb, matrix, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR
    )


def brightness_rotation(crop: np.ndarray) -> int:
    """pipeline.js `correctRotation`: 180 when the top band is brighter."""
    margin = round(CROP_SIZE * 0.15)
    luma = crop.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    top = luma[:margin, margin : CROP_SIZE - margin].sum()
    bottom = luma[CROP_SIZE - margin :, margin : CROP_SIZE - margin].sum()
    return 180 if top > bottom else 0


def rotated(crop: np.ndarray, rotation: int) -> np.ndarray:
    return crop[::-1, ::-1] if rotation == 180 else crop


def top_k(session: onnxruntime.InferenceSession, crop: np.ndarray) -> list:
    """pipeline.js `preprocessViT`, the model, and `topK`: (output index, score)."""
    tensor = (crop.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]
    (outputs,) = session.run(None, {session.get_inputs()[0].name: tensor})
    scores = outputs[0].astype(np.float64)
    if scores.max() > 1.0:
        exp = np.exp(scores - scores.max())
        scores = exp / exp.sum()
    order = np.argsort(-scores, kind="stable")[:TOP_K]
    return [(int(i), float(scores[i])) for i in order]


def classify(session, crop: np.ndarray) -> tuple:
    """pipeline.js `classifyWithRotationFallback`: (rotation, upright crop, top-k)."""
    guess = brightness_rotation(crop)
    best = (guess, rotated(crop, guess), top_k(session, rotated(crop, guess)))
    baseline = best[2][0][1]
    if baseline < ROTATION_FALLBACK_CONFIDENCE:
        other = 180 - guess
        result = top_k(session, rotated(crop, other))
        score = result[0][1]
        if score > baseline and score >= baseline * ROTATION_FLIP_MARGIN:
            best = (other, rotated(crop, other), result)
    return best


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <models dir with {MODEL_FILE}, {LABELS_FILE}>")
    models_dir = Path(sys.argv[1])
    model = verified(models_dir, MODEL_FILE, MODEL_SHA256)
    labels = json.loads(verified(models_dir, LABELS_FILE, LABELS_SHA256).read_text())
    session = onnxruntime.InferenceSession(
        str(model), providers=["CPUExecutionProvider"]
    )

    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("crop_*.png"):
        old.unlink()
    crops = []
    for image in json.loads(DETECT_GOLDEN.read_text())["images"]:
        bgr = cv2.imread(str(REPO_ROOT / image["path"]))
        if bgr is None:
            sys.exit(f"could not read {image['path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        stem = Path(image["path"]).stem
        for index, box in enumerate(image["boxes"]):
            quad = portrait_quad(box["corners"])
            rotation, upright, top = classify(session, warp(rgb, quad))
            png = OUT_DIR / f"crop_{stem}_{index:02d}.png"
            cv2.imwrite(
                str(png),
                cv2.cvtColor(np.ascontiguousarray(upright), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 9],
            )
            crops.append(
                {
                    "image": image["path"],
                    "box": index,
                    "quad": [[round(x, 4), round(y, 4)] for x, y in quad.tolist()],
                    "rotation": rotation,
                    "crop": png.relative_to(REPO_ROOT).as_posix(),
                    "candidates": [
                        {
                            "index": i,
                            "passcode": int(labels[str(i)]["card_id"]),
                            "score": round(score, 7),
                        }
                        for i, score in top
                    ],
                }
            )
            best = labels[str(top[0][0])]
            print(f"{png.name}: {rotation:3d}° {best['label']} {top[0][1]:.4f}")

    golden = {
        "generator": "crates/draw-core/tests/fixtures/gen_classify.py",
        "onnxruntime": onnxruntime.__version__,
        "opencv": cv2.__version__,
        "model": {"file": MODEL_FILE, "sha256": MODEL_SHA256},
        "labels": {"file": LABELS_FILE, "sha256": LABELS_SHA256},
        "top_k": TOP_K,
        "crops": crops,
    }
    out = OUT_DIR / "golden.json"
    out.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(crops)} crops and {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
