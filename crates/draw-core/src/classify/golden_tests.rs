//! Golden tests: the classifier against onnxruntime running the same ViT on the
//! reference crops, and the whole pipeline on `sample_photo.jpg` (see
//! `tests/fixtures/gen_classify.py`).
//!
//! ONNX Runtime builds disagree on the int8 ViT: on the identical input tensor, the
//! prebuilt runtime `ort` downloads scores the int8 top-1 up to 0.2 lower than Python's
//! onnxruntime (1.31 or 1.32), from their int8 kernels. On the fp32 ViT they agree
//! within about 2e-3. So scores are checked on fp32, within 5e-3 (some room for other
//! CPUs), and on int8 only the top-1 card is, where onnxruntime is fairly sure of it.
//!
//! Need the models: set `DRAW_CORE_MODELS_DIR` to a directory holding the int8 and fp32
//! files of [`ModelManifest::PINNED`] under their file names (`ygo_yolo.onnx`,
//! `vit_int8.onnx`, `vit_fp32.onnx`, `cardnames_onnx.json`). Without it the tests pass
//! after printing why they skipped (visible with `--nocapture`). CI always sets it.

use std::path::{Path, PathBuf};

use super::orientation::{brightness_rotation, upright_quad};
use super::{Classifier, decision};
use crate::detect::Detector;
use crate::warp::{self, reference};
use crate::{
    FrameRef, ModelManifest, ModelRole, ModelVariant, Passcode, Point, Recognition,
    RecognizerOptions,
};

/// The most an fp32 score may differ from onnxruntime's.
const MAX_FP32_SCORE_DELTA: f32 = 5e-3;

/// The int8 top-1 must match onnxruntime's where onnxruntime scores it at least this.
/// Below, the top-1 is a near tie that the runtimes' int8 differences can flip.
const MIN_INT8_REFERENCE_SCORE: f32 = 0.05;

const MONSTER_REBORN: Passcode = Passcode(83_764_719);

/// `DRAW_CORE_MODELS_DIR`, or `None` after printing why `test` skips.
fn models_dir(test: &str) -> Option<PathBuf> {
    let dir = std::env::var_os("DRAW_CORE_MODELS_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping {test}: DRAW_CORE_MODELS_DIR is unset");
    }
    dir
}

/// The `variant`'s file for `role`, in `models_dir`.
fn model_file(models_dir: &Path, variant: ModelVariant, role: ModelRole) -> PathBuf {
    let file = ModelManifest::PINNED
        .files(variant)
        .iter()
        .find(|file| file.role == role)
        .expect("every variant has one file per role");
    let file_name = Path::new(file.path)
        .file_name()
        .expect("manifest paths name a file");
    models_dir.join(file_name)
}

fn classifier(models_dir: &Path, variant: ModelVariant) -> Classifier {
    Classifier::new(
        &model_file(models_dir, variant, ModelRole::Classifier),
        &model_file(models_dir, variant, ModelRole::Labels),
    )
    .unwrap_or_else(|err| panic!("the pinned {variant:?} classifier loads: {err}"))
}

#[test]
fn the_fp32_classifier_matches_onnxruntime_on_the_reference_crops() {
    let Some(models_dir) = models_dir("the fp32 classifier golden test") else {
        return;
    };
    let mut classifier = classifier(&models_dir, ModelVariant::Fp32);
    let golden = reference::golden();
    let mut worst: f32 = 0.0;

    for reference in &golden.crops {
        let expected = &reference.candidates.fp32;

        let candidates = classifier
            .classify(&reference.load(), golden.top_k)
            .expect("classification runs");

        assert_eq!(candidates.len(), expected.len());
        assert_eq!(
            candidates[0].passcode,
            Passcode(expected[0].passcode),
            "{}: top-1",
            reference.crop
        );
        for (rank, (actual, expected)) in candidates.iter().zip(expected).enumerate() {
            let delta = (actual.score - expected.score).abs();
            assert!(
                delta <= MAX_FP32_SCORE_DELTA,
                "{} rank {rank}: score {} vs onnxruntime {}",
                reference.crop,
                actual.score,
                expected.score
            );
            worst = worst.max(delta);
        }
    }
    println!(
        "fp32 classifier vs onnxruntime: {} crops, top-{} scores, worst delta {worst:.2e}",
        golden.crops.len(),
        golden.top_k
    );
}

#[test]
fn the_int8_classifier_agrees_with_onnxruntime_on_the_top_1_card() {
    let Some(models_dir) = models_dir("the int8 classifier golden test") else {
        return;
    };
    let mut classifier = classifier(&models_dir, ModelVariant::Int8);
    let golden = reference::golden();
    let (mut checked, mut worst) = (0, 0.0_f32);

    for reference in &golden.crops {
        let expected = &reference.candidates.int8[0];
        if expected.score < MIN_INT8_REFERENCE_SCORE {
            continue;
        }

        let candidates = classifier
            .classify(&reference.load(), golden.top_k)
            .expect("classification runs");

        assert_eq!(
            candidates[0].passcode,
            Passcode(expected.passcode),
            "{}: top-1 (score {} vs onnxruntime {})",
            reference.crop,
            candidates[0].score,
            expected.score
        );
        checked += 1;
        worst = worst.max((candidates[0].score - expected.score).abs());
    }
    assert_ne!(checked, 0, "no reference crop is confident enough to check");
    println!(
        "int8 classifier vs onnxruntime: top-1 equal on {checked} of {} crops, worst top-1 \
         score delta {worst:.3} (not checked)",
        golden.crops.len()
    );
}

fn center(corners: [Point; 4]) -> Point {
    let sum = corners
        .iter()
        .fold((0.0, 0.0), |(x, y), p| (x + p.x, y + p.y));
    Point {
        x: sum.0 / 4.0,
        y: sum.1 / 4.0,
    }
}

#[test]
fn sample_photo_recognizes_monster_reborn_from_its_detection() {
    let Some(models_dir) = models_dir("the end-to-end classification test") else {
        return;
    };
    let options = RecognizerOptions {
        variant: ModelVariant::Int8,
        ..RecognizerOptions::default()
    };
    let mut detector = Detector::new(
        &model_file(&models_dir, ModelVariant::Int8, ModelRole::Detector),
        options.min_detector_confidence,
    )
    .expect("the pinned detector loads");
    let mut classifier = classifier(&models_dir, ModelVariant::Int8);
    // Where the reference pipeline finds Monster Reborn.
    let reference = reference::golden()
        .crops
        .into_iter()
        .find(|crop| {
            crop.image == "docs/samples/sample_photo.jpg"
                && crop.candidates.int8[0].passcode == MONSTER_REBORN.0
        })
        .expect("the reference recognizes Monster Reborn in sample_photo.jpg");
    let expected = upright_quad(&reference.quad(), reference.rotation());
    let rgba = image::open(reference::repo_root().join(&reference.image))
        .expect("sample_photo.jpg decodes")
        .to_rgba8();
    let frame = FrameRef::new(rgba.width(), rgba.height(), rgba.as_raw()).unwrap();

    let detections = detector.detect(frame).expect("detection runs");
    let recognized: Vec<_> = detections
        .iter()
        .map(|detection| {
            let crop = warp::warp(frame, &detection.quad).expect("detected quads warp");
            let (rotation, candidates) = classifier
                .classify_upright(&crop, options.top_k)
                .expect("classification runs");
            let quad = upright_quad(&detection.quad, rotation);
            let recognition = decision::decide(Some(&candidates), &options);
            println!(
                "card at ({:.0}, {:.0}): {rotation:?} (brightness suggested {:?}), \
                 {recognition:?}",
                center(quad.0).x,
                center(quad.0).y,
                brightness_rotation(&crop)
            );
            (quad, candidates[0].passcode, recognition)
        })
        .collect();

    let (quad, top_1, recognition) = recognized
        .iter()
        .find(|(quad, _, _)| distance(center(quad.0), center(expected.0)) < 10.0)
        .expect("a detection lies on the reference quad");
    assert_eq!(*top_1, MONSTER_REBORN);
    assert!(
        matches!(recognition, Recognition::Recognized(c) if c.passcode == MONSTER_REBORN),
        "{recognition:?}"
    );
    // The card's upright top-left corner, whichever end the detector started from.
    assert!(
        distance(quad.0[0], expected.0[0]) < 10.0,
        "upright quad {quad:?}, reference {expected:?}"
    );
}

fn distance(a: Point, b: Point) -> f32 {
    (a.x - b.x).hypot(a.y - b.y)
}
