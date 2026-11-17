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
- `solveHomography`, `warpPerspective`: the 224×224 crop, in
  `crates/draw-core/src/warp/` (done in A2).
- `correctRotation`, `classifyWithRotationFallback`: orientation, in
  `crates/draw-core/src/classify/orientation.rs` (done in A2).
- `preprocessViT`, `topK`, the label files and the deck filter (from `draw.py`):
  classification, in `crates/draw-core/src/classify/` (done in A2).

Gotchas that are easy to drop during a port:

- **Portrait normalization**: when a box has `w > h`, swap w/h and add π/2 to the angle
  *before* building its corners (and so before warping). Without it, the crop comes out
  squeezed and confidence stays low in every rotation.
- **Quad corner order after detection**: the detector's quad follows the box's portrait
  frame, so an upside-down card has the same quad as an upright one. Orientation settles
  the 0°/180° order that `Quad` promises: `orientation::upright_quad` turns the
  portrait quad and the chosen rotation into the upright order. Turning a crop a half
  turn (reversing its pixels) gives exactly the crop of the quad warped from its
  opposite corner, so there is no need to warp twice.
- **Warp pixel convention**: `warpPerspective` maps the quad's corners to the crop's
  outer edges (0 and 224) and samples each crop pixel at its center (`dx + 0.5`).
  OpenCV samples at whole coordinates, so the same mapping in `cv2.warpPerspective`
  sends the corners to -0.5 and 223.5. Without that shift the crops differ by 3 to 6
  levels on average.
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
- **`topK` softmax**: apply softmax only when the outputs are logits. This deliberately
  differs from `pipeline.js`, which treats any outputs with max ≤ 1 as probabilities:
  logits that all happen to be ≤ 1 would then become scores outside 0–1, even negative
  ones. draw-core treats outputs as probabilities only when **every** value lies in
  0–1, and softmaxes them otherwise. The pinned ViTs output logits, so both rules agree
  on them.
- **Two label files**, both JSON objects keyed by output index (`"0"`, `"1"`, …): the
  `yugiscan` ViT's `card_labels_yugiscan.json` holds plain `Name-<passcode>` strings,
  and the other ViTs' `cardnames_onnx.json` holds `{EN, FR, card_id, label}` objects
  with the passcode in `card_id`.
- **ONNX Runtime builds disagree on the int8 ViT**: on the identical input tensor, the
  prebuilt runtime `ort` rc.15 downloads (ONNX Runtime 1.31) scores the int8 top-1 up to
  0.2 lower than Python's onnxruntime (1.31 or 1.32), and the top-1 card differs on near
  ties. The cause is the int8 kernels; it is not the preprocessing, the graph
  optimization level, KleidiAI or the arm64 bf16 option. On the fp32 ViT they agree
  within about 2e-3. Compare scores on fp32, and only top-1 cards on int8.
- **CPU results vary slightly between runs**: the same session's scores move by up to
  about 1e-3 depending on what else runs at the time (the fp32 golden test measured a
  worst delta of 2.0e-3 run alone and 1.0e-3 next to the other tests). Keep tolerances
  above that.
- **Deck filtering needs a deep top-k**: Python uses 15. A top-3 rarely contains the deck
  card.
- **The manifest has copies**: `rust.yml`, `crates/draw-core/tests/fixtures/gen_detect.py`
  and `gen_classify.py` repeat the pinned URLs and SHA-256 values of
  `ModelManifest::PINNED` (the detector, the int8 and fp32 ViTs and
  `cardnames_onnx.json`). Change them all together.

Golden tests compare draw-core with Python running **the same ONNX files**:

- Detection (A1): `crates/draw-core/tests/fixtures/gen_detect.py` writes the reference
  boxes for `docs/samples/sample_photo.jpg` and three frames of `sample_video.mp4` to
  `crates/draw-core/tests/fixtures/detect/`. Regeneration steps are in the script. A
  reference box matches when a detection overlaps it with polygon IoU ≥ 0.9, and the
  counts must be equal. Current minimum IoU: 0.978.
- Crop and classification (A2): `crates/draw-core/tests/fixtures/gen_classify.py`
  warps every reference detection box with OpenCV (the pipeline.js corner mapping),
  orients it with the int8 ViT, and writes the upright crops as lossless PNG plus each
  ViT's top-15 from onnxruntime to `crates/draw-core/tests/fixtures/classify/`.
  Regeneration steps are in the script. The tests check:
  - the warp against the OpenCV crops: mean absolute difference ≤ 2 levels per byte
    (currently 0.31 at worst, mostly from the two JPEG decoders);
  - the fp32 ViT: the same top-1 on every crop and every top-15 score within 5e-3
    (currently 2.0e-3 at worst, on macOS arm64);
  - the int8 ViT: the same top-1 where onnxruntime scores it at least 0.05 (30 of 39
    crops), with no score tolerance (see the int8 gotcha above);
  - end to end on `docs/samples/sample_photo.jpg`: detector, warp, orientation and
    int8 ViT recognize Monster Reborn `83764719` (0.87), with its upright quad.
- The tests need the model files: set `DRAW_CORE_MODELS_DIR` to a directory holding
  them under their manifest file names (`ygo_yolo.onnx`, `vit_int8.onnx`,
  `vit_fp32.onnx`, `cardnames_onnx.json`). Without it they print why and skip. CI
  downloads, verifies and caches the files, so they always run there. Locally, keep them
  in the git-ignored `.models/`:

  ```sh
  DRAW_CORE_MODELS_DIR=$PWD/.models cargo test -p draw-core golden -- --nocapture
  ```

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
