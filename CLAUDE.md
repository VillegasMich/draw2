# draw2: Rust (`draw-core`)

This file covers only the Rust part of draw2. The Python package (`src/draw/`) and the
web demo (`docs/`) have their own setup in the README.

## What `crates/draw-core` is

`draw-core` is the card recognition library for the draw-ex desktop app
(https://github.com/VillegasMich/draw-ex). draw-ex depends on it through a pinned git
tag. It owns model loading, preprocessing, orientation, classification, deck filtering
and the model manifest.

Design and vocabulary live in draw-ex: its README, `GLOSSARY.md` and
`docs/adr/0001-recognition-lives-in-draw2.md`. Use the glossary's terms (Frame, Quad,
Detection, Candidate, Recognition, Passcode, Model variant, Model manifest) in names,
docs and tests.

## Porting notes

The reference implementation is draw2's **browser** pipeline, `docs/scripts/pipeline.js`.
Port from it, not from the Python `src/draw/draw.py`: the Python version's text-box
contour orientation step is the main source of `orientation_unknown` failures.

Port map, pipeline.js → draw-core:

- `preprocessYOLO`, `parseYOLOOutput`, `xywhrToCorners`, `nmsOBB`, `iouApprox`:
  detection, in `crates/draw-core/src/detect/` (done in A1).
- `solveHomography`, `warpPerspective`: the 224×224 crop.
- `correctRotation`, `classifyWithRotationFallback`: orientation.
- `preprocessViT`, `topK`: classification.

Gotchas that are easy to drop during a port:

- **Portrait normalization**: when a box has `w > h`, swap w/h and add π/2 to the angle
  *before* building its corners (and so before warping). Without it, the crop comes out
  squeezed and confidence stays low in every rotation.
- **Quad corner order after detection**: the detector's quad follows the box's portrait
  frame, so an upside-down card has the same quad as an upright one. Orientation (A2)
  settles the 0°/180° order that `Quad` promises.
- **YOLO output layout varies**: check whether features come first (`dims[1] < dims[2]`)
  instead of hard-coding the layout. The pinned `ygo_yolo.onnx` outputs `[1, 6, 8400]`
  (x, y, w, h, confidence, angle; one class).
- **Letterbox resize**: use OpenCV `INTER_LINEAR` semantics (2×2 bilinear, no
  antialiasing), as ultralytics does when it trains and predicts. An antialiased resize
  (the browser's `drawImage`, `image`'s `Triangle`) shifts some boxes by several pixels.
  On `sample_photo.jpg` that dropped one box to IoU 0.89 against the reference. Padding
  black (the JS) or gray 114 (ultralytics) makes no measurable difference.
- **One ONNX Runtime environment per process**: in `ort` 2.0.0-rc.15 a second
  `ort::init().build()` fails, even after the first environment is dropped. Every
  session takes its environment from `runtime::environment()`. Never call `ort::init()`
  anywhere else.
- **`topK` softmax**: apply softmax only when the outputs are logits (max > 1).
- **Two label files**: the `yugiscan` ViT uses `card_labels_yugiscan.json`, with plain
  `Name-<passcode>` strings. The other ViTs use `cardnames_onnx.json`, keyed by output
  index.
- **Deck filtering needs a deep top-k**: Python uses 15. A top-3 rarely contains the deck
  card.
- **The manifest has a copy in CI**: `rust.yml` repeats the detector's URL and SHA-256
  from `ModelManifest::PINNED`. Change them together.

Golden tests compare draw-core with ultralytics running **the same ONNX file**:

- Detection (A1): `crates/draw-core/tests/fixtures/gen_detect.py` writes the reference
  boxes for `docs/samples/sample_photo.jpg` and three frames of `sample_video.mp4` to
  `crates/draw-core/tests/fixtures/detect/`. Regeneration steps are in the script. A
  reference box matches when a detection overlaps it with polygon IoU ≥ 0.9, and the
  counts must be equal. Current minimum IoU: 0.978.
- The tests need the model files: set `DRAW_CORE_MODELS_DIR` to a directory holding
  them under their manifest file names (e.g. `ygo_yolo.onnx`). Without it they print why
  and skip. CI downloads, verifies and caches the files, so they always run there.
  Locally, keep them in the git-ignored `.models/`:

  ```sh
  DRAW_CORE_MODELS_DIR=$PWD/.models cargo test -p draw-core golden -- --nocapture
  ```

- Classification (A2): `docs/samples/sample_photo.jpg` should give Monster Reborn
  `83764719`. draw2's Python pipeline is the reference there
  (`uv run python -m draw --source <img> -v --log-json out.jsonl`).

## Checks

Run all four before committing Rust changes (CI runs the same in `rust.yml` on macOS and
Windows):

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace
```

## Hard rules

- **Stateless**: one frame in, detections out. Nothing carries across frames; tracking
  belongs to draw-ex.
- **Recognition only**: app and duel logic stay in draw-ex.
- **Manifest moves with the code**: change `ModelManifest::PINNED` in the same change as
  code that expects different model files.
- **Rust work stays in the Rust paths**: the Python package and the `docs/` web demo are
  untouched by it.
- **Release** = tag `draw-core-vX.Y.Z` on `main`, matching the crate version.
