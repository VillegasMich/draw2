//! A test double for [`Recognize`], available with the `fake` cargo feature.

use std::time::Duration;

use crate::{Detection, FrameRef, Recognize, RecognizeError};

/// A [`Recognize`] that returns preset detections, optionally after a delay.
///
/// Use it to build and test code that consumes recognition without loading models. It
/// ignores the frame's pixels and validates nothing beyond what [`FrameRef`] already
/// guarantees.
#[derive(Debug, Clone, PartialEq)]
pub struct FakeRecognizer {
    detections: Vec<Detection>,
    delay: Option<Duration>,
}

impl FakeRecognizer {
    /// Creates a fake that returns `detections` (in source-frame pixels of whatever frame
    /// it is given) from every call, after waiting `delay` if one is set.
    #[must_use]
    pub fn new(detections: Vec<Detection>, delay: Option<Duration>) -> Self {
        Self { detections, delay }
    }
}

impl Recognize for FakeRecognizer {
    /// Sleeps for the configured delay, then returns a clone of the configured detections.
    ///
    /// The delay blocks the calling thread with [`std::thread::sleep`]. That is fine here:
    /// [`Recognize`] is a synchronous API that the app calls on a dedicated recognition
    /// thread, which is exactly the thread a real model run would occupy.
    fn recognize(&mut self, _frame: FrameRef<'_>) -> Result<Vec<Detection>, RecognizeError> {
        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        }
        Ok(self.detections.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::{Candidate, Passcode, Point, Quad, Recognition};

    fn monster_reborn() -> Detection {
        let candidate = Candidate {
            passcode: Passcode(83_764_719),
            score: 0.93,
        };
        Detection {
            quad: Quad([
                Point { x: 10.0, y: 20.0 },
                Point { x: 110.0, y: 20.0 },
                Point { x: 110.0, y: 165.0 },
                Point { x: 10.0, y: 165.0 },
            ]),
            detector_confidence: 0.88,
            candidates: vec![
                candidate,
                Candidate {
                    passcode: Passcode(5_318_639),
                    score: 0.04,
                },
            ],
            recognition: Recognition::Recognized(candidate),
        }
    }

    fn frame(rgba: &[u8]) -> FrameRef<'_> {
        FrameRef::new(2, 2, rgba).expect("2x2 RGBA8 frame is 16 bytes")
    }

    #[test]
    fn recognize_returns_exactly_the_configured_detections() {
        let detections = vec![monster_reborn(), monster_reborn()];
        let mut fake = FakeRecognizer::new(detections.clone(), None);
        let rgba = [0u8; 16];

        assert_eq!(fake.recognize(frame(&rgba)).unwrap(), detections);
        assert_eq!(fake.recognize(frame(&rgba)).unwrap(), detections);
    }

    #[test]
    fn recognize_with_no_detections_returns_none() {
        let mut fake = FakeRecognizer::new(Vec::new(), None);
        let rgba = [0u8; 16];

        assert_eq!(fake.recognize(frame(&rgba)).unwrap(), []);
    }

    #[test]
    fn recognize_takes_at_least_the_configured_delay() {
        let delay = Duration::from_millis(30);
        let mut fake = FakeRecognizer::new(vec![monster_reborn()], Some(delay));
        let rgba = [0u8; 16];

        let started = Instant::now();
        let detections = fake.recognize(frame(&rgba)).unwrap();

        assert!(started.elapsed() >= delay);
        assert_eq!(detections, [monster_reborn()]);
    }
}
