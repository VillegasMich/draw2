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

The porting notes for the real pipeline currently live in draw-ex's `CLAUDE.md` and move
here in ticket A1.

## Checks

Run all three before committing Rust changes (CI runs the same in `rust.yml` on macOS and
Windows):

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
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
