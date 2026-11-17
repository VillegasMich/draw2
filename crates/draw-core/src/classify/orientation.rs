//! Settling whether a crop shows the card upright or upside down.
//!
//! The detector's quad follows the card's portrait frame, so its crop shows the card
//! either upright or turned a half turn. A port of `correctRotation` and
//! `classifyWithRotationFallback` from `pipeline.js`.

use std::borrow::Cow;

use crate::warp::{CROP_SIZE, Crop};
use crate::{Candidate, Quad};

/// Below this top-1 score, the other rotation is classified too.
const FALLBACK_BELOW: f32 = 0.15;

/// The other rotation wins only when its top-1 score is at least this many times the
/// first one's: a marginal gain on two low scores is noise.
const FLIP_MARGIN: f32 = 1.25;

/// The turn that makes a crop show its card upright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rotation {
    /// The crop is already upright.
    Zero,
    /// The crop shows the card upside down.
    HalfTurn,
}

impl Rotation {
    /// The other rotation.
    #[must_use]
    fn other(self) -> Self {
        match self {
            Self::Zero => Self::HalfTurn,
            Self::HalfTurn => Self::Zero,
        }
    }

    /// `crop` turned by this rotation.
    #[must_use]
    pub(crate) fn apply(self, crop: &Crop) -> Cow<'_, Crop> {
        match self {
            Self::Zero => Cow::Borrowed(crop),
            Self::HalfTurn => Cow::Owned(crop.rotated_half_turn()),
        }
    }
}

/// The rotation the card's brightness suggests: a half turn when the crop's top band is
/// brighter than its bottom band, since the card's bright text box sits at its bottom.
///
/// Port of `correctRotation`: each band is the crop's outer 15% (34 rows), without the
/// 15% at either side, summed as luma (BT.601 weights).
#[must_use]
pub(crate) fn brightness_rotation(crop: &Crop) -> Rotation {
    // round(224 * 0.15), as pipeline.js computes it.
    const MARGIN: usize = 34;
    let luma = |rows: std::ops::Range<usize>| -> f64 {
        rows.flat_map(|row| {
            let start = (row * CROP_SIZE + MARGIN) * 3;
            let end = (row * CROP_SIZE + CROP_SIZE - MARGIN) * 3;
            crop.rgb()[start..end].as_chunks::<3>().0
        })
        .map(|&[r, g, b]| 0.299 * f64::from(r) + 0.587 * f64::from(g) + 0.114 * f64::from(b))
        .sum()
    };
    if luma(0..MARGIN) > luma(CROP_SIZE - MARGIN..CROP_SIZE) {
        Rotation::HalfTurn
    } else {
        Rotation::Zero
    }
}

/// Classifies the crop at the `guess` rotation, and at the other one if the guess scores
/// low, keeping the rotation whose candidates win. `classify` gives the candidates of
/// the crop turned by a rotation, highest score first.
///
/// Port of `classifyWithRotationFallback`: the other rotation is tried only when the
/// guess's top-1 score is below 0.15, and kept only when its top-1 score beats the
/// guess's and is at least 1.25 times it.
///
/// # Errors
///
/// The first error `classify` returns.
pub(crate) fn orient<E>(
    guess: Rotation,
    mut classify: impl FnMut(Rotation) -> Result<Vec<Candidate>, E>,
) -> Result<(Rotation, Vec<Candidate>), E> {
    let candidates = classify(guess)?;
    let baseline = top_score(&candidates);
    if baseline >= FALLBACK_BELOW {
        return Ok((guess, candidates));
    }
    let other = guess.other();
    let other_candidates = classify(other)?;
    let score = top_score(&other_candidates);
    if score > baseline && score >= baseline * FLIP_MARGIN {
        Ok((other, other_candidates))
    } else {
        Ok((guess, candidates))
    }
}

/// The top-1 score of `candidates` (highest first), or 0 if there are none.
fn top_score(candidates: &[Candidate]) -> f32 {
    candidates.first().map_or(0.0, |candidate| candidate.score)
}

/// The quad in the [`Quad`] order (top-left, top-right, bottom-right, bottom-left of the
/// card read upright), from a detector quad that follows the card's portrait frame and
/// the `rotation` that turns its crop upright.
#[must_use]
pub(crate) fn upright_quad(portrait: &Quad, rotation: Rotation) -> Quad {
    let Quad([a, b, c, d]) = *portrait;
    match rotation {
        Rotation::Zero => *portrait,
        // The crop's far corners are the card's top-left and top-right.
        Rotation::HalfTurn => Quad([c, d, a, b]),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{Passcode, Point};

    /// An upright reference crop of Monster Reborn, its text box at the bottom.
    fn upright_card() -> Crop {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/classify/crop_sample_photo_06.png");
        let rgb = image::open(&path)
            .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
            .to_rgb8()
            .into_raw();
        Crop::from_rgb(rgb).expect("reference crops are 224x224")
    }

    #[test]
    fn the_brightness_rule_keeps_an_upright_card_and_turns_a_flipped_one() {
        let upright = upright_card();
        let flipped = upright.rotated_half_turn();

        assert_eq!(brightness_rotation(&upright), Rotation::Zero);
        assert_eq!(brightness_rotation(&flipped), Rotation::HalfTurn);
    }

    /// Candidates whose top-1 scores `top`, for a fake classifier.
    fn scored(top: f32) -> Vec<Candidate> {
        vec![
            Candidate {
                passcode: Passcode(1),
                score: top,
            },
            Candidate {
                passcode: Passcode(2),
                score: top / 10.0,
            },
        ]
    }

    /// Runs [`orient`] from a `Zero` guess with a fake classifier scoring `zero` and
    /// `half_turn`, returning the rotation kept and the rotations classified.
    fn orient_with(zero: f32, half_turn: f32) -> (Rotation, f32, Vec<Rotation>) {
        let mut tried = Vec::new();
        let (rotation, candidates) = orient(Rotation::Zero, |rotation| {
            tried.push(rotation);
            Ok::<_, ()>(scored(match rotation {
                Rotation::Zero => zero,
                Rotation::HalfTurn => half_turn,
            }))
        })
        .unwrap();
        (rotation, candidates[0].score, tried)
    }

    #[test]
    fn a_confident_guess_is_kept_without_trying_the_other_rotation() {
        assert_eq!(
            orient_with(0.15, 0.9),
            (Rotation::Zero, 0.15, vec![Rotation::Zero])
        );
    }

    #[test]
    fn a_low_scoring_guess_gives_way_to_a_rotation_at_least_one_and_a_quarter_times_better() {
        let both = vec![Rotation::Zero, Rotation::HalfTurn];

        assert_eq!(
            orient_with(0.1, 0.125),
            (Rotation::HalfTurn, 0.125, both.clone())
        );
        assert_eq!(orient_with(0.1, 0.6), (Rotation::HalfTurn, 0.6, both));
    }

    #[test]
    fn a_low_scoring_guess_is_kept_when_the_other_rotation_is_less_than_a_quarter_better() {
        let both = vec![Rotation::Zero, Rotation::HalfTurn];

        assert_eq!(orient_with(0.1, 0.124), (Rotation::Zero, 0.1, both.clone()));
        assert_eq!(orient_with(0.1, 0.05), (Rotation::Zero, 0.1, both.clone()));
        // Two zero scores are no evidence either way.
        assert_eq!(orient_with(0.0, 0.0), (Rotation::Zero, 0.0, both));
    }

    #[test]
    fn errors_from_the_classifier_are_returned() {
        assert_eq!(
            orient(Rotation::Zero, |_| Err::<Vec<_>, _>("boom")),
            Err("boom")
        );
    }

    fn points(corners: [(f32, f32); 4]) -> Quad {
        Quad(corners.map(|(x, y)| Point { x, y }))
    }

    #[test]
    fn the_upright_quad_keeps_the_portrait_order_at_zero_and_swaps_ends_at_a_half_turn() {
        let portrait = points([(10.0, 20.0), (60.0, 20.0), (60.0, 90.0), (10.0, 90.0)]);

        assert_eq!(upright_quad(&portrait, Rotation::Zero), portrait);
        assert_eq!(
            upright_quad(&portrait, Rotation::HalfTurn),
            points([(60.0, 90.0), (10.0, 90.0), (10.0, 20.0), (60.0, 20.0)])
        );
    }
}
