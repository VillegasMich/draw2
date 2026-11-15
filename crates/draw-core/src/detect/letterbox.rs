//! The 640×640 letterbox the detector reads, and the mapping back to the source frame.

use crate::{FrameRef, Point};

/// Side of the square detector input, in pixels.
pub(crate) const INPUT_SIZE: u32 = 640;

/// How a frame was fitted into the detector input: scaled to fit, then centered on
/// black padding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Letterbox {
    /// Input pixels per source-frame pixel. `f64` so that mapping back lands on whole
    /// source pixels where the reference (JavaScript doubles) does.
    pub(crate) scale: f64,
    /// Width of the scaled frame inside the input, in input pixels.
    pub(crate) resized_width: u32,
    /// Height of the scaled frame inside the input, in input pixels.
    pub(crate) resized_height: u32,
    /// Padding left of the scaled frame, in input pixels.
    pub(crate) pad_x: u32,
    /// Padding above the scaled frame, in input pixels.
    pub(crate) pad_y: u32,
}

impl Letterbox {
    /// Fits a `width` × `height` frame (source pixels) into the input, as
    /// `preprocessYOLO` in `pipeline.js` does: the largest scale that fits, the scaled
    /// size rounded, and the padding split with the extra pixel below or right.
    #[must_use]
    pub(crate) fn fit(width: u32, height: u32) -> Self {
        let side = f64::from(INPUT_SIZE);
        let scale = (side / f64::from(width)).min(side / f64::from(height));
        let scaled = |length: u32| {
            // In 1..=INPUT_SIZE: the larger side scales to exactly INPUT_SIZE, and a
            // sliver frame keeps at least one row or column.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let rounded = (f64::from(length) * scale).round() as u32;
            rounded.clamp(1, INPUT_SIZE)
        };
        let (resized_width, resized_height) = (scaled(width), scaled(height));
        Self {
            scale,
            resized_width,
            resized_height,
            pad_x: (INPUT_SIZE - resized_width) / 2,
            pad_y: (INPUT_SIZE - resized_height) / 2,
        }
    }

    /// Maps a position in input pixels back to source-frame pixels.
    #[must_use]
    pub(crate) fn to_source(self, x: f32, y: f32) -> Point {
        Point {
            x: self.unscale(f64::from(x) - f64::from(self.pad_x)),
            y: self.unscale(f64::from(y) - f64::from(self.pad_y)),
        }
    }

    /// Maps a length in input pixels back to source-frame pixels.
    #[must_use]
    pub(crate) fn to_source_length(self, length: f32) -> f32 {
        self.unscale(f64::from(length))
    }

    #[allow(clippy::cast_possible_truncation)] // Source-frame pixels fit f32 by design.
    fn unscale(self, input: f64) -> f32 {
        (input / self.scale) as f32
    }
}

/// Builds the detector input for `frame`: the frame scaled into the letterbox with
/// [`resize_bilinear`], as RGB / 255 in NCHW order (`[1, 3, 640, 640]`, `f32`). Alpha is
/// ignored and the padding is black.
#[must_use]
pub(crate) fn to_input_tensor(frame: FrameRef<'_>) -> (Vec<f32>, Letterbox) {
    let fit = Letterbox::fit(frame.width(), frame.height());
    let resized = resize_bilinear(frame, fit.resized_width, fit.resized_height);

    let side = INPUT_SIZE as usize;
    let area = side * side;
    let resized_width = fit.resized_width as usize;
    let mut tensor = vec![0.0; 3 * area];
    for (row, pixels) in resized.chunks_exact(resized_width * 3).enumerate() {
        let row_start = (row + fit.pad_y as usize) * side + fit.pad_x as usize;
        for (column, rgb) in pixels.as_chunks::<3>().0.iter().enumerate() {
            for (channel, &value) in rgb.iter().enumerate() {
                tensor[channel * area + row_start + column] = f32::from(value) / 255.0;
            }
        }
    }
    (tensor, fit)
}

/// Scales `frame` to `width` × `height` (both nonzero) pixels with OpenCV's
/// `INTER_LINEAR` semantics, returning packed RGB8, row-major. Alpha is dropped.
///
/// This is the resize ultralytics letterboxes with, so the detector sees what it was
/// trained on: each output pixel blends the 2×2 source pixels around its center, with
/// no antialiasing when shrinking. An antialiased resize (such as `image`'s `Triangle`)
/// shifts some boxes enough to fail the golden test.
#[must_use]
pub(crate) fn resize_bilinear(frame: FrameRef<'_>, width: u32, height: u32) -> Vec<u8> {
    let source_width = frame.width() as usize;
    let rgba = frame.rgba();
    let columns = taps(frame.width(), width);
    let rows = taps(frame.height(), height);

    let mut rgb = Vec::with_capacity(columns.len() * rows.len() * 3);
    for &(y0, y1, wy) in &rows {
        for &(x0, x1, wx) in &columns {
            let pixel = |x: usize, y: usize| (y * source_width + x) * 4;
            let [p00, p01, p10, p11] = [pixel(x0, y0), pixel(x1, y0), pixel(x0, y1), pixel(x1, y1)];
            for channel in 0..3 {
                let value = |p: usize| f32::from(rgba[p + channel]);
                let top = value(p00) + (value(p01) - value(p00)) * wx;
                let bottom = value(p10) + (value(p11) - value(p10)) * wx;
                let blended = top + (bottom - top) * wy;
                // Rounded and clamped into 0..=255, so the cast is exact.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                rgb.push(blended.round().clamp(0.0, 255.0) as u8);
            }
        }
    }
    rgb
}

/// For each of `output` positions along one axis, the two source positions (out of
/// `source`) it blends and the weight of the second, as OpenCV's `INTER_LINEAR` does:
/// pixel centers aligned, and positions past either edge clamped to the edge pixel.
fn taps(source: u32, output: u32) -> Vec<(usize, usize, f32)> {
    let scale = f64::from(source) / f64::from(output);
    let last = source as usize - 1;
    (0..output)
        .map(|position| {
            let center = (f64::from(position) + 0.5) * scale - 0.5;
            let floor = center.floor();
            if floor < 0.0 {
                return (0, 0, 0.0);
            }
            // Nonnegative and below `source` (centers lie inside the source), so exact.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let first = floor as usize;
            if first >= last {
                (last, last, 0.0)
            } else {
                #[allow(clippy::cast_possible_truncation)] // A fraction in 0..1.
                let weight = (center - floor) as f32;
                (first, first + 1, weight)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn letterbox(scale: f64, resized: (u32, u32), pad: (u32, u32)) -> Letterbox {
        Letterbox {
            scale,
            resized_width: resized.0,
            resized_height: resized.1,
            pad_x: pad.0,
            pad_y: pad.1,
        }
    }

    #[test]
    fn fit_scales_a_landscape_frame_to_the_input_width_and_pads_above_and_below() {
        assert_eq!(
            Letterbox::fit(1920, 1080),
            letterbox(1.0 / 3.0, (640, 360), (0, 140))
        );
    }

    #[test]
    fn fit_scales_a_portrait_frame_to_the_input_height_and_pads_left_and_right() {
        assert_eq!(
            Letterbox::fit(720, 1280),
            letterbox(0.5, (360, 640), (140, 0))
        );
    }

    #[test]
    fn fit_scales_a_square_frame_up_to_fill_the_input_without_padding() {
        assert_eq!(Letterbox::fit(320, 320), letterbox(2.0, (640, 640), (0, 0)));
    }

    #[test]
    fn fit_rounds_the_scaled_size_and_floors_odd_padding() {
        // 333 * 0.64 = 213.12 rounds to 213, leaving 427 rows: 213 above, 214 below.
        assert_eq!(
            Letterbox::fit(1000, 333),
            letterbox(0.64, (640, 213), (0, 213))
        );
    }

    #[test]
    fn to_source_maps_input_pixels_back_to_source_frame_pixels() {
        let fit = Letterbox::fit(1920, 1080);

        assert_eq!(fit.to_source(0.0, 140.0), Point { x: 0.0, y: 0.0 });
        assert_eq!(fit.to_source(320.0, 320.0), Point { x: 960.0, y: 540.0 });
        assert_eq!(
            fit.to_source(640.0, 500.0),
            Point {
                x: 1920.0,
                y: 1080.0
            }
        );
        assert_eq!(fit.to_source_length(64.0), 192.0);
    }

    #[derive(serde::Deserialize)]
    struct RgbImage {
        width: u32,
        height: u32,
        rgb: Vec<u8>,
    }

    #[derive(serde::Deserialize)]
    struct ResizeReference {
        source: RgbImage,
        resized: Vec<RgbImage>,
    }

    #[test]
    fn resize_bilinear_matches_opencv_inter_linear_within_one_level() {
        // OpenCV's reference values, from `tests/fixtures/gen_detect.py`. OpenCV blends
        // 8-bit pixels in 11-bit fixed point, so values may differ by one level.
        let reference: ResizeReference =
            serde_json::from_str(include_str!("../../tests/fixtures/detect/resize.json"))
                .expect("resize.json parses");
        let rgba: Vec<u8> = reference
            .source
            .rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect();
        let frame = FrameRef::new(reference.source.width, reference.source.height, &rgba)
            .expect("the source is width * height pixels");

        for expected in &reference.resized {
            let actual = resize_bilinear(frame, expected.width, expected.height);

            assert_eq!(actual.len(), expected.rgb.len());
            for (i, (&a, &e)) in actual.iter().zip(&expected.rgb).enumerate() {
                assert!(
                    a.abs_diff(e) <= 1,
                    "{}x{} byte {i}: {a} vs OpenCV {e}",
                    expected.width,
                    expected.height
                );
            }
        }
    }

    #[test]
    fn to_input_tensor_is_nchw_rgb_over_255_with_black_padding() {
        // A 4x2 frame of one color fills the 640x320 middle band of the input.
        let rgba = [255, 128, 0, 255].repeat(4 * 2);
        let frame = FrameRef::new(4, 2, &rgba).expect("4x2 RGBA8 frame is 32 bytes");

        let (tensor, fit) = to_input_tensor(frame);

        assert_eq!(fit, letterbox(160.0, (640, 320), (0, 160)));
        let area = (INPUT_SIZE * INPUT_SIZE) as usize;
        assert_eq!(tensor.len(), 3 * area);
        let at = |channel: usize, x: usize, y: usize| tensor[channel * area + y * 640 + x];
        for (x, y) in [(0, 0), (639, 159), (320, 480), (639, 639)] {
            assert_eq!(
                [at(0, x, y), at(1, x, y), at(2, x, y)],
                [0.0; 3],
                "({x}, {y})"
            );
        }
        for (x, y) in [(0, 160), (320, 320), (639, 479)] {
            assert_eq!(
                [at(0, x, y), at(1, x, y), at(2, x, y)],
                [1.0, 128.0 / 255.0, 0.0],
                "({x}, {y})"
            );
        }
    }
}
