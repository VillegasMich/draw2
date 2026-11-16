//! The projective map between a quad and the crop, solved by hand as in `pipeline.js`.

/// A 3×3 projective transform, row-major, acting on `(x, y, 1)`.
pub(super) type Homography = [[f64; 3]; 3];

/// Pivots smaller than this (in absolute value) make a system singular, as in
/// `pipeline.js`.
const SINGULAR: f64 = 1e-12;

/// The homography that maps each of `source` to the matching `target`, or `None` when
/// the points are degenerate (three collinear, or repeated).
///
/// Port of `solveHomography`: the eight unknowns of the transform (its bottom-right
/// entry fixed at 1) solved from the two equations each point pair gives.
pub(super) fn solve(source: &[[f64; 2]; 4], target: &[[f64; 2]; 4]) -> Option<Homography> {
    let mut system = [[0.0; 9]; 8];
    for (i, (&[sx, sy], &[tx, ty])) in source.iter().zip(target).enumerate() {
        system[2 * i] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -tx * sx, -tx * sy, tx];
        system[2 * i + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -ty * sx, -ty * sy, ty];
    }
    let h = gauss_jordan(system)?;
    Some([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
}

/// Solves the 8×8 linear system whose augmented rows are `system` (coefficients, then
/// the right-hand side), with partial pivoting. `None` if it is singular.
fn gauss_jordan(mut system: [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for column in 0..8 {
        // The first row with the largest magnitude, as `pipeline.js` picks it.
        let pivot_row = (column + 1..8).fold(column, |best, row| {
            if system[row][column].abs() > system[best][column].abs() {
                row
            } else {
                best
            }
        });
        system.swap(column, pivot_row);
        let pivot = system[column][column];
        if negligible(pivot) {
            return None;
        }
        let pivot_values = system[column].map(|value| value / pivot);
        system[column] = pivot_values;
        for (row, values) in system.iter_mut().enumerate() {
            if row != column {
                let factor = values[column];
                for (value, &pivot_value) in values.iter_mut().zip(&pivot_values).skip(column) {
                    *value -= factor * pivot_value;
                }
            }
        }
    }
    Some(system.map(|row| row[8]))
}

/// The inverse of `h`, or `None` if it is singular. Port of `invertH`.
pub(super) fn invert(h: &Homography) -> Option<Homography> {
    let [[a, b, c], [d, e, f], [g, h, i]] = *h;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if negligible(determinant) {
        return None;
    }
    let v = 1.0 / determinant;
    Some([
        [
            (e * i - f * h) * v,
            (c * h - b * i) * v,
            (b * f - c * e) * v,
        ],
        [
            (f * g - d * i) * v,
            (a * i - c * g) * v,
            (c * d - a * f) * v,
        ],
        [
            (d * h - e * g) * v,
            (b * g - a * h) * v,
            (a * e - b * d) * v,
        ],
    ])
}

/// Whether `value` is too close to zero (or NaN) to divide by.
fn negligible(value: f64) -> bool {
    value.is_nan() || value.abs() < SINGULAR
}

/// The image of `(x, y)` under `h`.
pub(super) fn apply(h: &Homography, x: f64, y: f64) -> (f64, f64) {
    let w = h[2][0] * x + h[2][1] * y + h[2][2];
    (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    )
}
