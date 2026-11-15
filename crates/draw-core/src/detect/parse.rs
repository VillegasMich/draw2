//! Turning the detector's raw output tensor into quads in source-frame pixels.

use super::letterbox::Letterbox;
use super::{DetectError, RawDetection};
use crate::{Point, Quad};

/// Features per anchor in a single-class YOLO-OBB output: center x, center y, width,
/// height (input pixels), confidence, angle (radians).
const FEATURES: usize = 6;

/// Reads the detector output `data` of shape `dims`, keeps the anchors with confidence
/// at least `min_confidence`, and maps their boxes through `letterbox` to quads in
/// source-frame pixels, in anchor order (before NMS).
///
/// The output is `[1, 6, anchors]` (features first, as exported by ultralytics) or
/// `[1, anchors, 6]`. As in `pipeline.js`, features come first when `dims[1] < dims[2]`,
/// since a real model has far more anchors than features.
///
/// # Errors
///
/// [`DetectError::UnexpectedOutputShape`] if `dims` is not one of those layouts with six
/// features, or `data` does not hold exactly that many values.
pub(crate) fn parse_output(
    dims: &[i64],
    data: &[f32],
    letterbox: &Letterbox,
    min_confidence: f32,
) -> Result<Vec<RawDetection>, DetectError> {
    let unexpected = || DetectError::UnexpectedOutputShape {
        shape: dims.to_vec(),
    };
    let &[1, rows, cols] = dims else {
        return Err(unexpected());
    };
    let rows = usize::try_from(rows).map_err(|_| unexpected())?;
    let cols = usize::try_from(cols).map_err(|_| unexpected())?;
    let features_first = rows < cols;
    let (anchors, features) = if features_first {
        (cols, rows)
    } else {
        (rows, cols)
    };
    if features != FEATURES || data.len() != anchors * FEATURES {
        return Err(unexpected());
    }

    let feature = |anchor: usize, index: usize| {
        if features_first {
            data[index * anchors + anchor]
        } else {
            data[anchor * FEATURES + index]
        }
    };
    let detections = (0..anchors)
        .filter_map(|anchor| {
            let confidence = feature(anchor, 4);
            (confidence >= min_confidence).then(|| RawDetection {
                quad: to_quad(
                    letterbox,
                    [0, 1, 2, 3, 5].map(|index| feature(anchor, index)),
                ),
                confidence,
            })
        })
        .collect();
    Ok(detections)
}

/// Builds the source-frame quad of the box `[cx, cy, w, h, angle]` (input pixels,
/// radians).
///
/// A landscape box (`w > h`) is first turned portrait: its sides are swapped and a
/// quarter turn is added to its angle. The warp to the classifier's crop relies on this:
/// without it, the card comes out sideways and squeezed, and no rotation recovers the
/// aspect the classifier was trained on.
fn to_quad(letterbox: &Letterbox, [cx, cy, w, h, angle]: [f32; 5]) -> Quad {
    let center = letterbox.to_source(cx, cy);
    let (mut w, mut h) = (letterbox.to_source_length(w), letterbox.to_source_length(h));
    let mut angle = angle;
    if w > h {
        std::mem::swap(&mut w, &mut h);
        angle += std::f32::consts::FRAC_PI_2;
    }
    let (sin, cos) = angle.sin_cos();
    let (half_w, half_h) = (w / 2.0, h / 2.0);
    // Top-left, top-right, bottom-right, bottom-left of the box in its own frame.
    Quad(
        [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        .map(|(dx, dy)| Point {
            x: center.x + dx * cos - dy * sin,
            y: center.y + dx * sin + dy * cos,
        }),
    )
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use super::*;

    /// One anchor's features: center x, center y, width, height (input pixels),
    /// confidence, angle (radians).
    type Anchor = [f32; 6];

    const ANCHORS: usize = 8;

    /// The output as the real model lays it out: `[1, 6, anchors]`, one row per feature.
    fn features_first(anchors: &[Anchor]) -> (Vec<i64>, Vec<f32>) {
        let padded = padded(anchors);
        let data = (0..6)
            .flat_map(|feature| padded.iter().map(move |anchor| anchor[feature]))
            .collect();
        (vec![1, 6, ANCHORS as i64], data)
    }

    /// The transposed layout: `[1, anchors, 6]`, one row per anchor.
    fn anchors_first(anchors: &[Anchor]) -> (Vec<i64>, Vec<f32>) {
        let data = padded(anchors).into_iter().flatten().collect();
        (vec![1, ANCHORS as i64, 6], data)
    }

    /// Fills up to `ANCHORS` with zero-confidence anchors, so there are more anchors than
    /// features, as in the real model.
    fn padded(anchors: &[Anchor]) -> Vec<Anchor> {
        let mut padded = anchors.to_vec();
        padded.resize(ANCHORS, [0.0; 6]);
        padded
    }

    fn quad(corners: [(f32, f32); 4]) -> Quad {
        Quad(corners.map(|(x, y)| Point { x, y }))
    }

    fn assert_quad_near(actual: &Quad, expected: &Quad) {
        for (a, e) in actual.0.iter().zip(&expected.0) {
            assert!(
                (a.x - e.x).abs() < 1e-3 && (a.y - e.y).abs() < 1e-3,
                "{actual:?} != {expected:?}"
            );
        }
    }

    /// A 640x640 frame: the letterbox is the identity.
    fn identity() -> Letterbox {
        Letterbox::fit(640, 640)
    }

    #[test]
    fn both_layouts_give_the_same_detections() {
        let anchors = [
            [100.0, 200.0, 50.0, 80.0, 0.9, 0.0],
            [400.0, 300.0, 60.0, 90.0, 0.6, 0.1],
        ];
        let (dims, data) = features_first(&anchors);
        let from_features_first = parse_output(&dims, &data, &identity(), 0.2).unwrap();
        let (dims, data) = anchors_first(&anchors);
        let from_anchors_first = parse_output(&dims, &data, &identity(), 0.2).unwrap();

        assert_eq!(from_features_first, from_anchors_first);
        assert_eq!(from_features_first.len(), 2);
        assert_eq!(from_features_first[0].confidence, 0.9);
        assert_quad_near(
            &from_features_first[0].quad,
            &quad([(75.0, 160.0), (125.0, 160.0), (125.0, 240.0), (75.0, 240.0)]),
        );
    }

    #[test]
    fn detections_below_the_minimum_confidence_are_dropped() {
        let (dims, data) = features_first(&[
            [100.0, 100.0, 50.0, 80.0, 0.19, 0.0],
            [300.0, 100.0, 50.0, 80.0, 0.2, 0.0],
            [500.0, 100.0, 50.0, 80.0, 0.7, 0.0],
        ]);

        let detections = parse_output(&dims, &data, &identity(), 0.2).unwrap();

        let confidences: Vec<f32> = detections.iter().map(|d| d.confidence).collect();
        assert_eq!(confidences, [0.2, 0.7]);
    }

    #[test]
    fn quads_are_mapped_through_the_letterbox_to_source_frame_pixels() {
        // 1280x720 fits as 640x360 at scale 0.5, with 140 rows of padding above.
        let (dims, data) = features_first(&[[320.0, 320.0, 50.0, 80.0, 0.9, 0.0]]);

        let detections = parse_output(&dims, &data, &Letterbox::fit(1280, 720), 0.2).unwrap();

        assert_quad_near(
            &detections[0].quad,
            &quad([
                (590.0, 280.0),
                (690.0, 280.0),
                (690.0, 440.0),
                (590.0, 440.0),
            ]),
        );
    }

    #[test]
    fn a_landscape_box_is_turned_portrait_by_swapping_its_sides_and_rotating_a_quarter_turn() {
        // 80 wide and 50 tall becomes 50 wide and 80 tall at angle π/2: the same area,
        // with the first corner on the box's (new) top-left in its rotated frame.
        let (dims, data) = features_first(&[[100.0, 200.0, 80.0, 50.0, 0.9, 0.0]]);

        let detections = parse_output(&dims, &data, &identity(), 0.2).unwrap();

        assert_quad_near(
            &detections[0].quad,
            &quad([(140.0, 175.0), (140.0, 225.0), (60.0, 225.0), (60.0, 175.0)]),
        );
        // The same quad the model would give for the portrait box at a quarter turn.
        let (dims, data) = features_first(&[[100.0, 200.0, 50.0, 80.0, 0.9, FRAC_PI_2]]);
        let portrait = parse_output(&dims, &data, &identity(), 0.2).unwrap();
        assert_quad_near(&detections[0].quad, &portrait[0].quad);
    }

    #[test]
    fn an_output_without_six_features_per_anchor_is_rejected() {
        let dims = [1, 7, ANCHORS as i64];
        let data = vec![0.0; 7 * ANCHORS];

        let err = parse_output(&dims, &data, &identity(), 0.2).unwrap_err();

        assert!(
            matches!(err, DetectError::UnexpectedOutputShape { ref shape } if shape == &dims),
            "{err:?}"
        );
    }

    #[test]
    fn an_output_whose_data_does_not_match_its_shape_is_rejected() {
        let (dims, mut data) = features_first(&[]);
        data.pop();

        let err = parse_output(&dims, &data, &identity(), 0.2).unwrap_err();

        assert!(
            matches!(err, DetectError::UnexpectedOutputShape { .. }),
            "{err:?}"
        );
    }
}
