//! draw-core: DRAW 2 card recognition for the draw-ex desktop app.
//!
//! draw-core finds the _Yu-Gi-Oh!_ cards in a camera frame and identifies each one by
//! its [`Passcode`]. It owns everything about recognition: model loading, preprocessing,
//! orientation, classification, deck filtering and the [`ModelManifest`].
//!
//! # Contract
//!
//! Recognition is stateless: one frame in, detections out. A [`Recognize`]
//! implementation takes a [`FrameRef`] and returns one [`Detection`] per card found,
//! and nothing carries over from one frame to the next. Following cards across frames,
//! and anything specific to a duel, belongs to the caller.
//!
//! [`Recognizer`] is the real pipeline. In this version it is a stub whose constructor
//! returns [`RecognizeError::NotImplemented`]. With the `fake` cargo feature,
//! `FakeRecognizer` returns preset detections, for building against this API.
//!
//! # Units and coordinate spaces
//!
//! - Frames are tightly packed RGBA8, row-major, with no row padding.
//! - Coordinates ([`Point`], [`Quad`]) are `f32` pixels of the source frame, with the
//!   origin at the top-left and y pointing down.
//! - A [`Quad`]'s corners are ordered top-left, top-right, bottom-right, bottom-left of
//!   the card read upright.
//! - Scores and confidences are `f32` from 0 to 1.
//! - Passcodes are integers; leading zeros are not significant.
//!
//! # Models
//!
//! [`ModelManifest::PINNED`] lists the model files this crate version expects, pinned to
//! one commit of the Hugging Face repo, with their SHA-256 and size. The manifest
//! changes together with code that expects different model files. draw-core never
//! downloads anything: the app fetches the files and passes their local paths in
//! [`ModelPaths`].

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
