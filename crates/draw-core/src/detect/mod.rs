//! Detection: finds the quads of the cards in one frame with the YOLO-OBB detector.
//!
//! A port of `preprocessYOLO`, `parseYOLOOutput`, `xywhrToCorners`, `nmsOBB` and
//! `iouApprox` from draw2's browser pipeline (`docs/scripts/pipeline.js`).

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;
use thiserror::Error;

use crate::{FrameRef, Quad, runtime};

mod letterbox;
mod nms;
mod parse;

#[cfg(test)]
mod golden_tests;

/// Detections overlapping a more confident one by more than this (axis-aligned IoU) are
/// dropped, as in `pipeline.js`.
const NMS_IOU_THRESHOLD: f32 = 0.5;

/// Runs the YOLO-OBB card detector (`ygo_yolo.onnx`) on frames, on the CPU.
///
/// Stateless across frames: each [`detect`](Self::detect) call depends only on its frame.
#[derive(Debug)]
pub(crate) struct Detector {
    session: Session,
    min_confidence: f32,
}

impl Detector {
    /// Loads the detector model at `model_path`. Detections with a confidence below
    /// `min_confidence` (from 0 to 1, normally
    /// [`RecognizerOptions::min_detector_confidence`](crate::RecognizerOptions)) are
    /// dropped.
    ///
    /// # Errors
    ///
    /// [`DetectError::Runtime`] if ONNX Runtime cannot start or load the model.
    pub(crate) fn new(model_path: &Path, min_confidence: f32) -> Result<Self, DetectError> {
        let session = Session::builder(&runtime::environment()?)?.commit_from_file(model_path)?;
        Ok(Self {
            session,
            min_confidence,
        })
    }

    /// Finds the cards in `frame`, most confident first, with quads in source-frame
    /// pixels.
    ///
    /// # Errors
    ///
    /// [`DetectError::Runtime`] if inference fails, or
    /// [`DetectError::UnexpectedOutputShape`] if the model's output is not a YOLO-OBB
    /// output.
    pub(crate) fn detect(&mut self, frame: FrameRef<'_>) -> Result<Vec<RawDetection>, DetectError> {
        let (input, letterbox) = letterbox::to_input_tensor(frame);
        let side = letterbox::INPUT_SIZE as usize;
        let input = TensorRef::from_array_view(([1, 3, side, side], input.as_slice()))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let detections = parse::parse_output(shape, data, &letterbox, self.min_confidence)?;
        Ok(nms::nms(detections, NMS_IOU_THRESHOLD))
    }
}

/// One card the detector found, before orientation and classification.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawDetection {
    /// Where the card is, in source-frame pixels.
    ///
    /// The corners follow the box's **portrait frame** (its long side vertical), not yet
    /// the card read upright: a card lying upside down has the same quad as one lying
    /// upright. Settling the 0°/180° order is orientation's job (ticket A2).
    pub(crate) quad: Quad,
    /// How sure the detector is that this is a card, from 0 to 1.
    pub(crate) confidence: f32,
}

/// Why detection failed.
#[derive(Debug, Error)]
pub(crate) enum DetectError {
    /// ONNX Runtime failed to start, load the model or run it.
    #[error("ONNX Runtime failed: {0}")]
    Runtime(#[from] ort::Error),
    /// The model's output is not one image of anchors with six features each.
    #[error("detector output has shape {shape:?}, expected [1, 6, anchors] or [1, anchors, 6]")]
    UnexpectedOutputShape {
        /// The output's dimensions, as reported by ONNX Runtime.
        shape: Vec<i64>,
    },
}
