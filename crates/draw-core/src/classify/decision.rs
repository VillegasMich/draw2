//! The verdict on one detection, from its candidates and the recognizer's options.

use crate::{Candidate, Recognition, RecognizerOptions};

/// Decides the [`Recognition`] of a detection from its top-k `candidates` (highest score
/// first), as the Python pipeline (`src/draw/draw.py`) does:
///
/// - `None` (the crop was unreadable, such as a degenerate quad) gives
///   [`Recognition::Unreadable`];
/// - a top-1 score below `options.min_score` (or no candidate) gives
///   [`Recognition::LowConfidence`], whatever the deck list;
/// - with a nonempty deck list, the highest-scoring candidate in it is recognized, and
///   [`Recognition::NotInDeck`] when none is;
/// - otherwise (no deck list, or an empty one) the top-1 is recognized.
#[must_use]
pub(crate) fn decide(candidates: Option<&[Candidate]>, options: &RecognizerOptions) -> Recognition {
    let Some(candidates) = candidates else {
        return Recognition::Unreadable;
    };
    let Some(&top) = candidates.first() else {
        return Recognition::LowConfidence;
    };
    if top.score < options.min_score {
        return Recognition::LowConfidence;
    }
    match options.deck.as_ref().filter(|deck| !deck.is_empty()) {
        None => Recognition::Recognized(top),
        Some(deck) => candidates
            .iter()
            .find(|candidate| deck.contains(&candidate.passcode))
            .map_or(Recognition::NotInDeck, |&candidate| {
                Recognition::Recognized(candidate)
            }),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::Passcode;

    /// Fifteen candidates, passcodes 100 to 114, scores from 0.40 down by 0.02.
    fn top_15() -> Vec<Candidate> {
        (0u8..15)
            .map(|rank| Candidate {
                passcode: Passcode(100 + u32::from(rank)),
                score: 0.40 - 0.02 * f32::from(rank),
            })
            .collect()
    }

    fn with_deck(deck: &[u32]) -> RecognizerOptions {
        RecognizerOptions {
            deck: Some(deck.iter().copied().map(Passcode).collect::<HashSet<_>>()),
            ..RecognizerOptions::default()
        }
    }

    #[test]
    fn an_unreadable_crop_is_unreadable() {
        assert_eq!(
            decide(None, &RecognizerOptions::default()),
            Recognition::Unreadable
        );
    }

    #[test]
    fn without_a_deck_list_the_top_1_is_recognized() {
        let candidates = top_15();

        assert_eq!(
            decide(Some(&candidates), &RecognizerOptions::default()),
            Recognition::Recognized(candidates[0])
        );
    }

    #[test]
    fn an_empty_deck_list_means_no_filtering() {
        let candidates = top_15();

        assert_eq!(
            decide(Some(&candidates), &with_deck(&[])),
            Recognition::Recognized(candidates[0])
        );
    }

    #[test]
    fn the_deck_filter_picks_the_best_in_deck_candidate_from_a_top_15() {
        let candidates = top_15();
        // Ranks 10 and 13 are in the deck, and so is a card outside the top 15.
        let options = with_deck(&[113, 999, 110]);

        assert_eq!(
            decide(Some(&candidates), &options),
            Recognition::Recognized(candidates[10])
        );
    }

    #[test]
    fn a_top_15_without_a_deck_card_is_not_in_deck() {
        assert_eq!(
            decide(Some(&top_15()), &with_deck(&[999, 98])),
            Recognition::NotInDeck
        );
    }

    #[test]
    fn a_top_1_below_the_minimum_score_is_low_confidence_even_with_a_deck_match() {
        let mut candidates = top_15();
        candidates.truncate(2);
        candidates[0].score = 0.049;
        candidates[1].score = 0.01;
        let options = with_deck(&[100]);

        assert_eq!(
            decide(Some(&candidates), &options),
            Recognition::LowConfidence
        );
    }

    #[test]
    fn no_candidates_at_all_is_low_confidence() {
        assert_eq!(
            decide(Some(&[]), &RecognizerOptions::default()),
            Recognition::LowConfidence
        );
    }

    #[test]
    fn a_top_1_at_exactly_the_minimum_score_is_recognized() {
        let mut candidates = top_15();
        candidates[0].score = RecognizerOptions::default().min_score;

        assert_eq!(
            decide(Some(&candidates), &RecognizerOptions::default()),
            Recognition::Recognized(candidates[0])
        );
    }
}
