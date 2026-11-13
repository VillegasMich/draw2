//! draw-core: DRAW 2 card recognition.

#![forbid(unsafe_code)]

mod detection;
#[cfg(feature = "fake")]
mod fake;
mod frame;
mod manifest;
mod recognize;

pub use detection::{Candidate, Detection, Passcode, Point, Quad, Recognition};
#[cfg(feature = "fake")]
pub use fake::FakeRecognizer;
pub use frame::{FrameError, FrameRef};
pub use manifest::{ModelFile, ModelManifest, ModelRole, ModelVariant};
pub use recognize::{ModelPaths, Recognize, RecognizeError, Recognizer, RecognizerOptions};
