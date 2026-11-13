//! What recognition reports about the cards in one frame.

/// A card's identity: the 8-digit number printed on the card.
///
/// Leading zeros are not significant, so `00012345` and `12345` are the same passcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Passcode(pub u32);

/// A position in the source frame, in pixels.
///
/// The origin is the frame's top-left corner, x points right and y points down. Values are
/// fractional because corners come from sub-pixel model output.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    /// Horizontal position, in source-frame pixels from the left edge.
    pub x: f32,
    /// Vertical position, in source-frame pixels from the top edge.
    pub y: f32,
}

/// The four corners of a card as seen in a frame, in source-frame pixels.
///
/// Corners are ordered top-left, top-right, bottom-right, bottom-left **of the card read
/// upright**, not of the frame: for a card lying upside down, the first corner is the
/// lowest-right one on screen.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Quad(pub [Point; 4]);

/// A passcode the classifier considers possible for a detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The candidate card.
    pub passcode: Passcode,
    /// How likely the classifier finds this candidate, from 0 to 1.
    pub score: f32,
}

/// The verdict on a single detection.
///
/// Only [`Recognition::Recognized`] carries a card, so "recognized" without a candidate
/// cannot be represented.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Recognition {
    /// The detection is this candidate.
    Recognized(Candidate),
    /// The best candidate scored below the minimum score.
    LowConfidence,
    /// No candidate above the minimum score is in the deck list.
    NotInDeck,
    /// The crop or the card's orientation could not be determined.
    Unreadable,
}

/// One card found in one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Where the card is, in source-frame pixels.
    pub quad: Quad,
    /// How sure the detector is that this is a card, from 0 to 1.
    pub detector_confidence: f32,
    /// The top-k candidates, sorted by score, highest first. Scores are 0 to 1.
    pub candidates: Vec<Candidate>,
    /// The verdict on this detection.
    pub recognition: Recognition,
}
