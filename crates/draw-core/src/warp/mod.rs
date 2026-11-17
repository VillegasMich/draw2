//! The card crop the classifier reads: a 224×224 perspective warp of a detection's quad.
//!
//! A port of `solveHomography` and `warpPerspective` from draw2's browser pipeline
//! (`docs/scripts/pipeline.js`), with no image library.

use crate::{FrameRef, Quad};

mod homography;
#[cfg(test)]
pub(crate) mod reference;

/// Side of the square crop the classifier reads, in pixels.
pub(crate) const CROP_SIZE: usize = 224;

/// Bytes in a crop: [`CROP_SIZE`] × [`CROP_SIZE`] RGB8 pixels.
const CROP_BYTES: usize = CROP_SIZE * CROP_SIZE * 3;

/// A card warped flat: [`CROP_SIZE`] × [`CROP_SIZE`] pixels, packed RGB8, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Crop {
    rgb: Vec<u8>,
}

impl Crop {
    /// Wraps `rgb` as a crop, or `None` if it is not [`CROP_SIZE`] × [`CROP_SIZE`] RGB8
    /// pixels. Only the tests build crops from pixels; the pipeline warps them.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_rgb(rgb: Vec<u8>) -> Option<Self> {
        (rgb.len() == CROP_BYTES).then_some(Self { rgb })
    }

    /// The pixels: packed RGB8, row-major, [`CROP_SIZE`] × [`CROP_SIZE`].
    #[must_use]
    pub(crate) fn rgb(&self) -> &[u8] {
        &self.rgb
    }

    /// The crop turned a half turn: pixel `(x, y)` moves to `(223 - x, 223 - y)`.
    ///
    /// Exactly the crop that warping the quad from its opposite corner gives.
    #[must_use]
    pub(crate) fn rotated_half_turn(&self) -> Self {
        let rgb = self.rgb.as_chunks::<3>().0.iter().rev().flatten().copied();
        Self { rgb: rgb.collect() }
    }
}

/// Warps the card at `quad` (source-frame pixels) in `frame` to a crop: corner 0 of
/// the quad goes to the crop's top-left, then top-right, bottom-right and bottom-left.
///
/// Each crop pixel samples the frame bilinearly at the image of its center, as
/// `warpPerspective` in `pipeline.js` does. Pixels whose 2×2 neighborhood leaves the
/// frame are black.
///
/// Returns `None` when the quad is degenerate (collinear or repeated corners, or a
/// non-finite coordinate), so no homography maps it onto the crop.
#[must_use]
pub(crate) fn warp(frame: FrameRef<'_>, quad: &Quad) -> Option<Crop> {
    let side = CROP_SIZE as f64;
    let corners = quad.0.map(|p| [f64::from(p.x), f64::from(p.y)]);
    if corners.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let crop_corners = [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]];
    let to_frame = homography::invert(&homography::solve(&corners, &crop_corners)?)?;

    let mut rgb = Vec::with_capacity(CROP_BYTES);
    for y in 0..CROP_SIZE {
        for x in 0..CROP_SIZE {
            let (sx, sy) = homography::apply(&to_frame, x as f64 + 0.5, y as f64 + 0.5);
            rgb.extend_from_slice(&sample(frame, sx, sy));
        }
    }
    Some(Crop { rgb })
}

/// The bilinear blend of the 2×2 pixels of `frame` around `(x, y)` (source-frame pixels,
/// pixel centers on whole coordinates), as RGB8. Black when the neighborhood is not
/// wholly inside the frame, or the position is not finite.
fn sample(frame: FrameRef<'_>, x: f64, y: f64) -> [u8; 3] {
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let rgba = frame.rgba();
    let (x0, y0) = (x.floor(), y.floor());
    // Also false for NaN.
    let inside = x0 >= 0.0 && y0 >= 0.0 && x0 + 1.0 < width as f64 && y0 + 1.0 < height as f64;
    if !inside {
        return [0; 3];
    }
    // Whole, nonnegative and below the frame size: the casts are exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (column, row) = (x0 as usize, y0 as usize);
    let (fx, fy) = (x - x0, y - y0);
    let at = |dx: usize, dy: usize| ((row + dy) * width + column + dx) * 4;
    let weights = [
        (at(0, 0), (1.0 - fx) * (1.0 - fy)),
        (at(1, 0), fx * (1.0 - fy)),
        (at(0, 1), (1.0 - fx) * fy),
        (at(1, 1), fx * fy),
    ];
    [0, 1, 2].map(|channel| {
        let value: f64 = weights
            .iter()
            .map(|&(pixel, weight)| f64::from(rgba[pixel + channel]) * weight)
            .sum();
        // A convex blend of bytes, so already in 0..=255 before the cast.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let level = value.round().clamp(0.0, 255.0) as u8;
        level
    })
}

#[cfg(test)]
mod tests {
    use super::reference::{self, quad};
    use super::*;

    /// Mean absolute difference per byte, in levels (0 to 255).
    fn mean_abs_diff(a: &Crop, b: &Crop) -> f64 {
        let total: u64 = a
            .rgb()
            .iter()
            .zip(b.rgb())
            .map(|(&x, &y)| u64::from(x.abs_diff(y)))
            .sum();
        total as f64 / CROP_BYTES as f64
    }

    #[test]
    fn warp_matches_the_opencv_reference_crops_within_two_levels() {
        // OpenCV crops from `tests/fixtures/gen_classify.py`, with the pipeline.js corner
        // mapping. OpenCV interpolates in fixed point and decodes JPEG with libjpeg, so
        // values may differ slightly.
        let golden = reference::golden();
        let mut worst: f64 = 0.0;
        let mut image_path = String::new();
        let mut frame_rgba = image::RgbaImage::new(1, 1);

        for reference in &golden.crops {
            if reference.image != image_path {
                image_path.clone_from(&reference.image);
                frame_rgba = image::open(reference::repo_root().join(&image_path))
                    .unwrap_or_else(|err| panic!("{image_path}: {err}"))
                    .to_rgba8();
            }
            let (width, height) = frame_rgba.dimensions();
            let frame = FrameRef::new(width, height, frame_rgba.as_raw()).unwrap();
            let expected = reference.load();

            let warped = warp(frame, &reference.quad()).expect("fixture quads warp");
            let actual = reference.rotation().apply(&warped);

            let diff = mean_abs_diff(&actual, &expected);
            assert!(diff <= 2.0, "{}: mean difference {diff:.3}", reference.crop);
            worst = worst.max(diff);
        }
        println!(
            "warp vs OpenCV: {} crops, worst mean difference {worst:.4} levels",
            golden.crops.len()
        );
    }

    #[test]
    fn a_degenerate_quad_gives_no_crop() {
        let rgba = vec![128; 64 * 64 * 4];
        let frame = FrameRef::new(64, 64, &rgba).unwrap();
        let collinear = quad([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]);
        let one_point = quad([[5.0, 5.0]; 4]);
        let not_finite = quad([[0.0, 0.0], [10.0, 0.0], [10.0, f32::NAN], [0.0, 10.0]]);

        assert_eq!(warp(frame, &collinear), None);
        assert_eq!(warp(frame, &one_point), None);
        assert_eq!(warp(frame, &not_finite), None);
    }
}
