//! Classification: scores the candidates for a card crop with a ViT classifier, settles
//! the card's orientation, and decides the recognition.
//!
//! A port of `correctRotation`, `classifyWithRotationFallback`, `preprocessViT` and
//! `topK` from draw2's browser pipeline (`docs/scripts/pipeline.js`), and of the deck
//! filter of the Python pipeline (`src/draw/draw.py`).

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::TensorRef;

use thiserror::Error;

use crate::warp::{CROP_SIZE, Crop};
use crate::{Candidate, runtime};

use self::labels::{Labels, LabelsError};
use self::orientation::Rotation;

pub(crate) mod decision;
pub(crate) mod labels;
pub(crate) mod orientation;
mod scores;

#[cfg(test)]
mod golden_tests;

/// Runs a ViT card classifier (one `vit_*.onnx` model variant) on crops, on the CPU.
///
/// Stateless across crops: each call depends only on its crop.
#[derive(Debug)]
pub(crate) struct Classifier {
    session: Session,
    labels: Labels,
}

impl Classifier {
    /// Loads the classifier model at `model_path` and the labels file at `labels_path`
    /// that maps its outputs to passcodes.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::ReadLabels`] or [`ClassifyError::Labels`] if the labels file
    /// cannot be read, and [`ClassifyError::Runtime`] if ONNX Runtime cannot start or
    /// load the model.
    pub(crate) fn new(model_path: &Path, labels_path: &Path) -> Result<Self, ClassifyError> {
        let json =
            std::fs::read_to_string(labels_path).map_err(|source| ClassifyError::ReadLabels {
                path: labels_path.to_owned(),
                source,
            })?;
        let labels = Labels::from_json(&json).map_err(|source| ClassifyError::Labels {
            path: labels_path.to_owned(),
            source,
        })?;
        let session = Session::builder(&runtime::environment()?)?.commit_from_file(model_path)?;
        Ok(Self { session, labels })
    }

    /// The `top_k` candidates for `crop` as it stands, highest score first, with scores
    /// from 0 to 1.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::Runtime`] if inference fails, or
    /// [`ClassifyError::UnexpectedOutputShape`] if the model does not give one score per
    /// label.
    pub(crate) fn classify(
        &mut self,
        crop: &Crop,
        top_k: usize,
    ) -> Result<Vec<Candidate>, ClassifyError> {
        let input = to_input_tensor(crop);
        let input = TensorRef::from_array_view(([1, 3, CROP_SIZE, CROP_SIZE], input.as_slice()))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, scores) = outputs[0].try_extract_tensor::<f32>()?;
        let labels = self.labels.len();
        if !matches!(**shape, [1, n] if usize::try_from(n) == Ok(labels)) {
            return Err(ClassifyError::UnexpectedOutputShape {
                shape: shape.to_vec(),
                labels,
            });
        }
        let candidates = scores::top_k(scores, top_k)
            .into_iter()
            .map(|(index, score)| Candidate {
                passcode: self
                    .labels
                    .passcode(index)
                    .expect("the output has one score per label"),
                score,
            })
            .collect();
        Ok(candidates)
    }

    /// Classifies `crop` upright: turned by the rotation its brightness suggests, or by
    /// the other one when that scores clearly better (see [`orientation::orient`]).
    /// Returns the rotation that makes the crop upright and the `top_k` candidates for
    /// the crop turned by it.
    ///
    /// # Errors
    ///
    /// As [`classify`](Self::classify).
    pub(crate) fn classify_upright(
        &mut self,
        crop: &Crop,
        top_k: usize,
    ) -> Result<(Rotation, Vec<Candidate>), ClassifyError> {
        orientation::orient(orientation::brightness_rotation(crop), |rotation| {
            self.classify(&rotation.apply(crop), top_k)
        })
    }
}

/// The classifier input for `crop`, as `preprocessViT` in `pipeline.js` builds it: RGB
/// / 255, normalized with mean = std = 0.5 (so from -1 to 1), in NCHW order
/// (`[1, 3, 224, 224]`, `f32`).
fn to_input_tensor(crop: &Crop) -> Vec<f32> {
    let area = CROP_SIZE * CROP_SIZE;
    let mut tensor = vec![0.0; 3 * area];
    for (pixel, rgb) in crop.rgb().as_chunks::<3>().0.iter().enumerate() {
        for (channel, &value) in rgb.iter().enumerate() {
            tensor[channel * area + pixel] = (f32::from(value) / 255.0 - 0.5) / 0.5;
        }
    }
    tensor
}

/// Why classification failed.
#[derive(Debug, Error)]
pub(crate) enum ClassifyError {
    /// ONNX Runtime failed to start, load the model or run it.
    #[error("ONNX Runtime failed: {0}")]
    Runtime(#[from] ort::Error),
    /// The labels file could not be read.
    #[error("cannot read the labels file {}: {source}", path.display())]
    ReadLabels {
        /// The labels file.
        path: PathBuf,
        /// The read error.
        source: std::io::Error,
    },
    /// The labels file is not in a known format.
    #[error("invalid labels file {}: {source}", path.display())]
    Labels {
        /// The labels file.
        path: PathBuf,
        /// What is wrong with it.
        source: LabelsError,
    },
    /// The model's output is not one score per label for one crop.
    #[error("classifier output has shape {shape:?}, expected [1, {labels}] (one per label)")]
    UnexpectedOutputShape {
        /// The output's dimensions, as reported by ONNX Runtime.
        shape: Vec<i64>,
        /// How many labels the labels file has.
        labels: usize,
    },
}
