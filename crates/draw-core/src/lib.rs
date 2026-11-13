//! draw-core: DRAW 2 card recognition.

#![forbid(unsafe_code)]

mod detection;
mod frame;
mod manifest;
mod recognize;

pub use detection::{Candidate, Detection, Passcode, Point, Quad, Recognition};
pub use frame::{FrameError, FrameRef};
pub use manifest::ModelVariant;
pub use recognize::{ModelPaths, Recognize, RecognizeError, Recognizer, RecognizerOptions};
