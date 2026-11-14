//! The frame type that recognition reads.

use thiserror::Error;

/// A borrowed view of one frame: tightly packed RGBA8 pixels, row-major, no row padding.
///
/// This is the layout a browser's `CanvasRenderingContext2D.getImageData` produces. Pixel
/// `(x, y)` (in source-frame pixels, origin top-left, y down) starts at byte
/// `(y * width + x) * 4`. Construction checks that the buffer length matches the
/// dimensions, so every `FrameRef` is well formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRef<'a> {
    width: u32,
    height: u32,
    rgba: &'a [u8],
}

impl<'a> FrameRef<'a> {
    /// Wraps `rgba` as a `width` × `height` frame (both in pixels).
    ///
    /// # Errors
    ///
    /// - [`FrameError::ZeroDimension`] if `width` or `height` is zero.
    /// - [`FrameError::TooLarge`] if `width * height * 4` does not fit in `usize`.
    /// - [`FrameError::LengthMismatch`] if `rgba.len()` is not exactly
    ///   `width * height * 4` bytes.
    pub fn new(width: u32, height: u32, rgba: &'a [u8]) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::ZeroDimension { width, height });
        }
        let expected = usize::try_from(width)
            .ok()
            .zip(usize::try_from(height).ok())
            .and_then(|(w, h)| w.checked_mul(h))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(FrameError::TooLarge { width, height })?;
        if rgba.len() != expected {
            return Err(FrameError::LengthMismatch {
                expected,
                actual: rgba.len(),
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    /// Frame width, in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height, in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The pixel bytes: RGBA8, row-major, `width * height * 4` bytes long.
    #[must_use]
    pub fn rgba(&self) -> &'a [u8] {
        self.rgba
    }
}

/// Why a buffer could not be wrapped as a [`FrameRef`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum FrameError {
    /// The width or height (in pixels) is zero.
    #[error("frame dimensions must be nonzero, got {width}x{height} pixels")]
    ZeroDimension {
        /// Requested width, in pixels.
        width: u32,
        /// Requested height, in pixels.
        height: u32,
    },
    /// `width * height * 4` bytes does not fit in memory on this platform.
    #[error("frame of {width}x{height} pixels is too large to address")]
    TooLarge {
        /// Requested width, in pixels.
        width: u32,
        /// Requested height, in pixels.
        height: u32,
    },
    /// The buffer length does not match the dimensions.
    #[error("RGBA buffer is {actual} bytes, expected {expected} (width * height * 4)")]
    LengthMismatch {
        /// Required length, in bytes.
        expected: usize,
        /// Actual buffer length, in bytes.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_a_buffer_of_exactly_width_times_height_times_four_bytes() {
        let rgba = vec![0u8; 3 * 2 * 4];

        let frame = FrameRef::new(3, 2, &rgba).expect("exact length is valid");

        assert_eq!(frame.width(), 3);
        assert_eq!(frame.height(), 2);
        assert_eq!(frame.rgba().len(), 24);
    }

    #[test]
    fn new_rejects_a_buffer_that_is_too_short() {
        let rgba = vec![0u8; 23];

        let err = FrameRef::new(3, 2, &rgba).unwrap_err();

        assert_eq!(
            err,
            FrameError::LengthMismatch {
                expected: 24,
                actual: 23
            }
        );
    }

    #[test]
    fn new_rejects_a_buffer_that_is_too_long() {
        let rgba = vec![0u8; 25];

        let err = FrameRef::new(3, 2, &rgba).unwrap_err();

        assert_eq!(
            err,
            FrameError::LengthMismatch {
                expected: 24,
                actual: 25
            }
        );
    }

    #[test]
    fn new_rejects_zero_width_or_height() {
        assert_eq!(
            FrameRef::new(0, 2, &[]).unwrap_err(),
            FrameError::ZeroDimension {
                width: 0,
                height: 2
            }
        );
        assert_eq!(
            FrameRef::new(3, 0, &[]).unwrap_err(),
            FrameError::ZeroDimension {
                width: 3,
                height: 0
            }
        );
    }

    #[test]
    fn new_rejects_dimensions_whose_byte_length_overflows() {
        let err = FrameRef::new(u32::MAX, u32::MAX, &[]).unwrap_err();

        assert_eq!(
            err,
            FrameError::TooLarge {
                width: u32::MAX,
                height: u32::MAX
            }
        );
    }
}
