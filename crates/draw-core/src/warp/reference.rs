//! Test fixtures: the reference crops of `tests/fixtures/classify/`, written by
//! `tests/fixtures/gen_classify.py`, shared by the warp, orientation and classifier tests.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::Crop;
use crate::classify::orientation::Rotation;
use crate::{Point, Quad};

/// `golden.json`: every reference crop and each ViT's top-k for it.
#[derive(Debug, Deserialize)]
pub(crate) struct Golden {
    /// How many candidates each reference list holds.
    pub(crate) top_k: usize,
    pub(crate) crops: Vec<ReferenceCrop>,
}

/// One card of a reference image, warped and oriented by OpenCV.
#[derive(Debug, Deserialize)]
pub(crate) struct ReferenceCrop {
    /// The source frame, relative to the repo root.
    pub(crate) image: String,
    /// The detector's portrait-frame quad the crop was warped from, in source-frame
    /// pixels.
    quad: [[f32; 2]; 4],
    /// Degrees, 0 or 180.
    rotation: u16,
    /// The upright crop (lossless PNG), relative to the repo root.
    pub(crate) crop: String,
    pub(crate) candidates: ReferenceCandidates,
}

/// Each ViT's top-k for a crop, from onnxruntime, highest score first.
#[derive(Debug, Deserialize)]
pub(crate) struct ReferenceCandidates {
    pub(crate) int8: Vec<ReferenceCandidate>,
    pub(crate) fp32: Vec<ReferenceCandidate>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReferenceCandidate {
    pub(crate) passcode: u32,
    pub(crate) score: f32,
}

impl ReferenceCrop {
    /// The portrait-frame quad the crop was warped from.
    pub(crate) fn quad(&self) -> Quad {
        quad(self.quad)
    }

    /// The rotation that turned the warped crop upright.
    pub(crate) fn rotation(&self) -> Rotation {
        match self.rotation {
            0 => Rotation::Zero,
            180 => Rotation::HalfTurn,
            other => panic!("{}: rotation {other} is not 0 or 180", self.crop),
        }
    }

    /// The upright crop.
    pub(crate) fn load(&self) -> Crop {
        load_crop(&repo_root().join(&self.crop))
    }
}

/// Parses `golden.json`.
pub(crate) fn golden() -> Golden {
    serde_json::from_str(include_str!("../../tests/fixtures/classify/golden.json"))
        .expect("classify golden.json parses")
}

/// The draw2 repo root, which fixture paths are relative to.
pub(crate) fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The 224x224 PNG crop at `path`.
pub(crate) fn load_crop(path: &Path) -> Crop {
    let rgb = image::open(path)
        .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
        .to_rgb8()
        .into_raw();
    Crop::from_rgb(rgb).unwrap_or_else(|| panic!("{} is not 224x224", path.display()))
}

/// A quad from `[x, y]` corners, in source-frame pixels.
pub(crate) fn quad(corners: [[f32; 2]; 4]) -> Quad {
    Quad(corners.map(|[x, y]| Point { x, y }))
}
