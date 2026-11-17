//! Turning the classifier's raw outputs into ranked scores.

/// How far past 1 an output may be and still count as a probability: a softmax computed
/// in `f32` inside the model can overshoot 1 by rounding.
const PROBABILITY_SLACK: f32 = 1e-5;

/// The `k` best outputs as `(output index, score)`, highest score first, with scores
/// from 0 to 1. Equal scores keep output order. `None` if an output is NaN or infinite,
/// which no score can be made from.
///
/// Port of `topK` in `pipeline.js`, with one deliberate difference. pipeline.js treats
/// the outputs as probabilities whenever their maximum is at most 1, and as logits
/// (softmaxed) otherwise. Logits that all happen to be at most 1 would then pass
/// through as scores outside 0 to 1, even negative ones. Here the outputs pass through
/// only when **every** value lies in 0 to 1 (give or take rounding), and are softmaxed
/// otherwise. For the pinned ViTs, which output logits well above 1, both rules agree.
#[must_use]
pub(crate) fn top_k(outputs: &[f32], k: usize) -> Option<Vec<(usize, f32)>> {
    if !outputs.iter().all(|output| output.is_finite()) {
        return None;
    }
    let are_probabilities = outputs
        .iter()
        .all(|p| (0.0..=1.0 + PROBABILITY_SLACK).contains(p));
    let mut scores: Vec<(usize, f32)> = if are_probabilities {
        outputs.iter().map(|&p| p.min(1.0)).enumerate().collect()
    } else {
        softmax(outputs).into_iter().enumerate().collect()
    };
    // Stable, so equal scores keep output order.
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    scores.truncate(k);
    Some(scores)
}

/// The softmax of `logits`, computed in `f64` from the largest logit down.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f64> = logits
        .iter()
        .map(|&logit| (f64::from(logit) - f64::from(max)).exp())
        .collect();
    let sum: f64 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|exponential| {
            #[allow(clippy::cast_possible_truncation)] // A probability: f32 keeps enough of it.
            let probability = (exponential / sum) as f32;
            probability
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn top_k_of(outputs: &[f32], k: usize) -> Vec<(usize, f32)> {
        top_k(outputs, k).expect("finite outputs rank")
    }

    fn assert_near(actual: &[(usize, f32)], expected: &[(usize, f32)]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} != {expected:?}");
        for (&(ai, ap), &(ei, ep)) in actual.iter().zip(expected) {
            assert!(
                ai == ei && (ap - ep).abs() < 1e-6,
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn logits_are_softmaxed_then_ranked_and_truncated_to_k() {
        // Exponentials 1, 6, 3 and e^-100 (about 0), out of 10.
        let logits = [0.0, 6f32.ln(), 3f32.ln(), -100.0];

        assert_near(&top_k_of(&logits, 2), &[(1, 0.6), (2, 0.3)]);
        assert_near(
            &top_k_of(&logits, 9),
            &[(1, 0.6), (2, 0.3), (0, 0.1), (3, 0.0)],
        );
    }

    #[test]
    fn logits_all_at_most_one_are_still_softmaxed() {
        // pipeline.js would return these raw, as scores of -1, 0 and -2.
        let logits = [-1.0, 0.0, -2.0];

        // e^0, e^-1, e^-2 over their sum 1.503215.
        assert_near(
            &top_k_of(&logits, 3),
            &[(1, 0.665_241), (0, 0.244_728), (2, 0.090_031)],
        );
    }

    #[test]
    fn probabilities_pass_through_ranked() {
        let probabilities = [0.1, 0.2, 0.0, 0.7];

        assert_near(
            &top_k_of(&probabilities, 3),
            &[(3, 0.7), (1, 0.2), (0, 0.1)],
        );
    }

    #[test]
    fn equal_scores_keep_output_order() {
        let probabilities = [0.25, 0.25, 0.25, 0.25];

        assert_near(&top_k_of(&probabilities, 2), &[(0, 0.25), (1, 0.25)]);
    }

    #[test]
    fn probabilities_overshooting_one_by_rounding_pass_through_capped_at_one() {
        let probabilities = [1.000_001, 0.0, 0.0];

        assert_near(&top_k_of(&probabilities, 2), &[(0, 1.0), (1, 0.0)]);
    }

    #[test]
    fn outputs_with_a_nan_or_an_infinity_give_no_scores() {
        assert_eq!(top_k(&[0.5, f32::NAN, 2.0], 2), None);
        assert_eq!(top_k(&[0.5, f32::INFINITY, 2.0], 2), None);
        assert_eq!(top_k(&[0.5, f32::NEG_INFINITY, 2.0], 2), None);
    }
}
