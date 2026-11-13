//! The recognition entry point: the [`Recognize`] trait and its real implementation.

use std::collections::HashSet;
use std::convert::Infallible;
use std::path::PathBuf;

use thiserror::Error;

use crate::{Detection, FrameRef, ModelVariant, Passcode};

/// Finds and identifies the cards in one frame.
///
/// Calls are independent: one frame in, detections out, with nothing carried across
/// frames.
///
/// `recognize` takes `&mut self` because ONNX Runtime sessions need mutable access to
/// run. The app runs recognition on a single dedicated thread, so implementations only
/// need to be [`Send`] (to move onto that thread), not [`Sync`].
pub trait Recognize: Send {
    /// Returns one detection per card found in `frame`. Quads are in source-frame pixels
    /// of `frame`.
    ///
    /// # Errors
    ///
    /// Returns a [`RecognizeError`] if inference itself fails. A frame with no cards is
    /// not an error: it gives an empty `Vec`.
    fn recognize(&mut self, frame: FrameRef<'_>) -> Result<Vec<Detection>, RecognizeError>;
}

/// Local paths to the model files for one [`ModelVariant`], as listed in the
/// [`ModelManifest`](crate::ModelManifest).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    /// The card detector (`ygo_yolo.onnx`).
    pub detector: PathBuf,
    /// The classifier for the chosen variant (a `vit_*.onnx` file).
    pub classifier: PathBuf,
    /// The labels file mapping classifier outputs to passcodes.
    pub labels: PathBuf,
}

/// How [`Recognizer`] filters and ranks its detections and candidates.
///
/// The defaults match the DRAW 2 reference pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct RecognizerOptions {
    /// The deck list: only these passcodes may be recognized. `None` (the default) means
    /// no deck filtering. `Some` of an empty set behaves the same as `None`, as in the
    /// DRAW 2 Python pipeline.
    pub deck: Option<HashSet<Passcode>>,
    /// Detections with a detector confidence below this are dropped, from 0 to 1.
    /// Default: 0.20.
    pub min_detector_confidence: f32,
    /// A best candidate scoring below this gives
    /// [`Recognition::LowConfidence`](crate::Recognition::LowConfidence), from 0 to 1.
    /// Default: 0.05.
    pub min_score: f32,
    /// How many candidates each detection keeps, highest score first. Default: 15, deep
    /// enough for deck filtering to find the deck card.
    pub top_k: usize,
    /// Which classifier build to run. Default: [`ModelVariant::Fp16`], provisional until
    /// real-hardware benchmarks pick the default (draw-ex ticket I2).
    pub variant: ModelVariant,
}

impl Default for RecognizerOptions {
    fn default() -> Self {
        Self {
            deck: None,
            min_detector_confidence: 0.20,
            min_score: 0.05,
            top_k: 15,
            variant: ModelVariant::default(),
        }
    }
}

/// The real recognition pipeline.
///
/// Not implemented yet: [`Recognizer::new`] always fails with
/// [`RecognizeError::NotImplemented`], so no `Recognizer` can exist.
#[derive(Debug)]
pub struct Recognizer {
    // Uninhabited until the pipeline lands, which makes `recognize` statically unreachable.
    never: Infallible,
}

impl Recognizer {
    /// Loads the models at `paths` and prepares to recognize with `options`.
    ///
    /// # Errors
    ///
    /// Always returns [`RecognizeError::NotImplemented`] in this version.
    pub fn new(paths: ModelPaths, options: RecognizerOptions) -> Result<Self, RecognizeError> {
        let _ = (paths, options);
        Err(RecognizeError::NotImplemented)
    }
}

impl Recognize for Recognizer {
    fn recognize(&mut self, _frame: FrameRef<'_>) -> Result<Vec<Detection>, RecognizeError> {
        match self.never {}
    }
}

/// Why recognition failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RecognizeError {
    /// The real pipeline is not implemented in this version of draw-core.
    #[error("recognition is not implemented in this version of draw-core")]
    NotImplemented,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizer_new_returns_not_implemented() {
        let paths = ModelPaths {
            detector: "ygo_yolo.onnx".into(),
            classifier: "vit_fp16.onnx".into(),
            labels: "cardnames_onnx.json".into(),
        };

        let result = Recognizer::new(paths, RecognizerOptions::default());

        assert!(matches!(result, Err(RecognizeError::NotImplemented)));
    }
}
