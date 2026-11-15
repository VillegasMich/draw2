//! Golden test: the detector against reference boxes from ultralytics running the same
//! `ygo_yolo.onnx` (see `tests/fixtures/gen_detect.py`).
//!
//! Needs the model: set `DRAW_CORE_MODELS_DIR` to a directory holding the detector file
//! of [`ModelManifest::PINNED`] under its file name (`ygo_yolo.onnx`). Without it the
//! test passes after printing why it skipped (visible with `--nocapture`). CI always sets
//! it.
//!
//! A reference box matches when a detection overlaps it with polygon IoU ≥ 0.9 (the
//! exact overlap of the two quads, not of their axis-aligned bounds), and each image must
//! give as many detections as reference boxes.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::Detector;
use crate::{FrameRef, ModelManifest, ModelRole, ModelVariant, Quad, RecognizerOptions};

const MIN_IOU: f64 = 0.9;

#[derive(Debug, Deserialize)]
struct Golden {
    images: Vec<GoldenImage>,
}

#[derive(Debug, Deserialize)]
struct GoldenImage {
    /// Relative to the repo root.
    path: String,
    width: u32,
    height: u32,
    boxes: Vec<GoldenBox>,
}

#[derive(Debug, Deserialize)]
struct GoldenBox {
    corners: [[f64; 2]; 4],
    confidence: f32,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn detector_model(models_dir: &Path) -> PathBuf {
    let detector = ModelManifest::PINNED
        .files(ModelVariant::default())
        .iter()
        .find(|file| file.role == ModelRole::Detector)
        .expect("every variant has a detector");
    let file_name = Path::new(detector.path)
        .file_name()
        .expect("manifest paths name a file");
    models_dir.join(file_name)
}

#[test]
fn golden_detections_match_the_ultralytics_reference_boxes() {
    let Some(models_dir) = std::env::var_os("DRAW_CORE_MODELS_DIR") else {
        eprintln!("skipping the detection golden test: DRAW_CORE_MODELS_DIR is unset");
        return;
    };
    let mut detector = Detector::new(
        &detector_model(Path::new(&models_dir)),
        RecognizerOptions::default().min_detector_confidence,
    )
    .expect("the pinned detector loads");
    let golden: Golden =
        serde_json::from_str(include_str!("../../tests/fixtures/detect/golden.json"))
            .expect("golden.json parses");

    for image in &golden.images {
        let rgba = image::open(repo_root().join(&image.path))
            .unwrap_or_else(|err| panic!("{}: {err}", image.path))
            .to_rgba8();
        assert_eq!(
            rgba.dimensions(),
            (image.width, image.height),
            "{}",
            image.path
        );
        let frame = FrameRef::new(image.width, image.height, rgba.as_raw()).unwrap();

        let detections = detector.detect(frame).expect("detection runs");

        assert_eq!(
            detections.len(),
            image.boxes.len(),
            "{}: detection count",
            image.path
        );
        let ious: Vec<f64> = image
            .boxes
            .iter()
            .map(|reference| {
                let best = detections
                    .iter()
                    .map(|d| polygon_iou(&reference.corners, &corners(&d.quad)))
                    .fold(0.0, f64::max);
                assert!(
                    best >= MIN_IOU,
                    "{}: reference box {:?} (confidence {}) best IoU {best:.3}",
                    image.path,
                    reference.corners,
                    reference.confidence
                );
                best
            })
            .collect();
        let min = ious.iter().copied().fold(1.0, f64::min);
        let mean = ious.iter().sum::<f64>() / ious.len() as f64;
        println!(
            "{}: {} boxes, IoU min {min:.4} mean {mean:.4}",
            image.path,
            ious.len()
        );
    }
}

fn corners(quad: &Quad) -> [[f64; 2]; 4] {
    quad.0.map(|p| [f64::from(p.x), f64::from(p.y)])
}

/// The exact IoU of two convex quads, whatever their winding.
fn polygon_iou(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> f64 {
    let (a, b) = (counter_clockwise(a), counter_clockwise(b));
    let intersection = signed_area(&clip(&a, &b));
    let union = signed_area(&a) + signed_area(&b) - intersection;
    intersection / union
}

/// The shoelace area: positive for counter-clockwise winding in y-up axes (clockwise on
/// screen, where y points down), negative for the other winding.
fn signed_area(polygon: &[[f64; 2]]) -> f64 {
    let n = polygon.len();
    (0..n)
        .map(|i| {
            let ([x0, y0], [x1, y1]) = (polygon[i], polygon[(i + 1) % n]);
            x0 * y1 - x1 * y0
        })
        .sum::<f64>()
        / 2.0
}

fn counter_clockwise(quad: &[[f64; 2]; 4]) -> Vec<[f64; 2]> {
    let mut polygon = quad.to_vec();
    if signed_area(&polygon) < 0.0 {
        polygon.reverse();
    }
    polygon
}

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Sutherland–Hodgman: `subject` clipped to the convex `clipper`, both counter-clockwise.
fn clip(subject: &[[f64; 2]], clipper: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut output = subject.to_vec();
    for i in 0..clipper.len() {
        let (edge_start, edge_end) = (clipper[i], clipper[(i + 1) % clipper.len()]);
        let input = std::mem::take(&mut output);
        for j in 0..input.len() {
            let (current, previous) = (input[j], input[(j + input.len() - 1) % input.len()]);
            let current_inside = cross(edge_start, edge_end, current) >= 0.0;
            let previous_inside = cross(edge_start, edge_end, previous) >= 0.0;
            if current_inside != previous_inside {
                output.push(line_intersection(previous, current, edge_start, edge_end));
            }
            if current_inside {
                output.push(current);
            }
        }
        if output.is_empty() {
            break;
        }
    }
    output
}

/// Where the segment `p0`–`p1` crosses the line through `q0` and `q1`.
fn line_intersection(p0: [f64; 2], p1: [f64; 2], q0: [f64; 2], q1: [f64; 2]) -> [f64; 2] {
    let (d0, d1) = (cross(q0, q1, p0), cross(q0, q1, p1));
    let t = d0 / (d0 - d1);
    [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])]
}

#[test]
fn polygon_iou_is_exact_for_known_overlaps() {
    let square = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
    let shifted = [[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]];
    let diamond = [[1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]];
    let mut reversed = square;
    reversed.reverse();

    assert!((polygon_iou(&square, &square) - 1.0).abs() < 1e-12);
    assert!((polygon_iou(&reversed, &square) - 1.0).abs() < 1e-12);
    assert!((polygon_iou(&square, &shifted) - 2.0 / 6.0).abs() < 1e-12);
    // The diamond (area 2) lies inside the square (area 4).
    assert!((polygon_iou(&square, &diamond) - 0.5).abs() < 1e-12);
}
