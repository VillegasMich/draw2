//! Non-maximum suppression over detections, with `pipeline.js`'s IoU approximation.

use super::RawDetection;
use crate::Quad;

/// Keeps the most confident of each group of overlapping detections, most confident
/// first. A detection is dropped when its [`iou_approx`] with a kept, more confident one
/// is above `iou_threshold`. Ties in confidence keep their input order.
#[must_use]
pub(crate) fn nms(mut detections: Vec<RawDetection>, iou_threshold: f32) -> Vec<RawDetection> {
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<RawDetection> = Vec::with_capacity(detections.len());
    for detection in detections {
        if kept
            .iter()
            .all(|k| iou_approx(&k.quad, &detection.quad) <= iou_threshold)
        {
            kept.push(detection);
        }
    }
    kept
}

/// The intersection over union of the axis-aligned bounds of `a` and `b`, from 0 to 1.
///
/// This is `pipeline.js`'s `iouApprox`: cheap, and exact for axis-aligned quads, but it
/// overestimates the overlap of rotated ones. It is kept as is so that NMS suppresses
/// what the reference suppresses.
#[must_use]
pub(crate) fn iou_approx(a: &Quad, b: &Quad) -> f32 {
    let ([ax0, ay0], [ax1, ay1]) = bounds(a);
    let ([bx0, by0], [bx1, by1]) = bounds(b);
    let overlap_w = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let overlap_h = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let intersection = overlap_w * overlap_h;
    let area_a = (ax1 - ax0) * (ay1 - ay0);
    let area_b = (bx1 - bx0) * (by1 - by0);
    // The epsilon keeps degenerate (zero-area) quads from dividing by zero.
    intersection / (area_a + area_b - intersection + 1e-8)
}

/// The minimum and maximum corner of `quad`'s axis-aligned bounds.
fn bounds(quad: &Quad) -> ([f32; 2], [f32; 2]) {
    quad.0.iter().fold(
        ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]),
        |([min_x, min_y], [max_x, max_y]), p| {
            (
                [min_x.min(p.x), min_y.min(p.y)],
                [max_x.max(p.x), max_y.max(p.y)],
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;

    /// An axis-aligned card-sized detection with its top-left corner at `(x, y)`.
    fn at(x: f32, y: f32, confidence: f32) -> RawDetection {
        let (w, h) = (100.0, 150.0);
        RawDetection {
            quad: Quad(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)].map(|(x, y)| Point { x, y }),
            ),
            confidence,
        }
    }

    #[test]
    fn of_overlapping_detections_only_the_most_confident_survives() {
        // Shifted by 10 px: IoU 1350 / 1650 ≈ 0.82 with its neighbor, above 0.5.
        let detections = vec![at(0.0, 0.0, 0.6), at(10.0, 0.0, 0.9), at(20.0, 0.0, 0.7)];

        let kept = nms(detections, 0.5);

        assert_eq!(kept, [at(10.0, 0.0, 0.9)]);
    }

    #[test]
    fn disjoint_detections_all_survive_most_confident_first() {
        let detections = vec![at(0.0, 0.0, 0.6), at(200.0, 0.0, 0.9), at(400.0, 0.0, 0.7)];

        let kept = nms(detections, 0.5);

        assert_eq!(
            kept,
            [at(200.0, 0.0, 0.9), at(400.0, 0.0, 0.7), at(0.0, 0.0, 0.6)]
        );
    }

    #[test]
    fn detections_overlapping_at_or_below_the_threshold_both_survive() {
        // Shifted by half a width: IoU 50*150 / (2*15000 - 7500) = 1/3.
        let detections = vec![at(0.0, 0.0, 0.9), at(50.0, 0.0, 0.8)];

        let kept = nms(detections.clone(), 0.5);

        assert_eq!(kept, detections);
    }

    #[test]
    fn iou_approx_compares_the_axis_aligned_bounds_of_the_quads() {
        let diamond = RawDetection {
            quad: Quad(
                [(50.0, 0.0), (100.0, 75.0), (50.0, 150.0), (0.0, 75.0)]
                    .map(|(x, y)| Point { x, y }),
            ),
            confidence: 0.9,
        };

        // A diamond inscribed in the card's box has the same bounds, so IoU 1.
        assert!((iou_approx(&diamond.quad, &at(0.0, 0.0, 0.9).quad) - 1.0).abs() < 1e-6);
        assert!(
            (iou_approx(&at(0.0, 0.0, 0.9).quad, &at(50.0, 0.0, 0.9).quad) - 1.0 / 3.0).abs()
                < 1e-6
        );
        assert_eq!(
            iou_approx(&at(0.0, 0.0, 0.9).quad, &at(200.0, 0.0, 0.9).quad),
            0.0
        );
    }
}
