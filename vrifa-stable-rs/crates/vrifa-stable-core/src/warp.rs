use crate::{cvutil, Result, VrifaError};
use ndarray::{Array2, Array3};
use opencv::core::{self, Mat, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct AffineWarp {
    pub rows: [[f32; 3]; 2],
}

impl Default for AffineWarp {
    fn default() -> Self {
        Self::identity()
    }
}

impl AffineWarp {
    pub fn identity() -> Self {
        Self {
            rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    }

    pub fn from_translation(dx: f32, dy: f32) -> Self {
        Self {
            rows: [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        }
    }

    pub fn translation(&self) -> (f32, f32) {
        (self.rows[0][2], self.rows[1][2])
    }

    pub fn map_point(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.rows[0][0] * x + self.rows[0][1] * y + self.rows[0][2],
            self.rows[1][0] * x + self.rows[1][1] * y + self.rows[1][2],
        )
    }

    pub fn displacement_at(&self, x: f32, y: f32) -> (f32, f32) {
        let (mapped_x, mapped_y) = self.map_point(x, y);
        (mapped_x - x, mapped_y - y)
    }

    pub fn center_displacement(&self, width: usize, height: usize) -> (f32, f32) {
        let center_x = (width.saturating_sub(1) as f32) * 0.5;
        let center_y = (height.saturating_sub(1) as f32) * 0.5;
        self.displacement_at(center_x, center_y)
    }

    pub fn compose(&self, rhs: &Self) -> Self {
        let a = &self.rows;
        let b = &rhs.rows;
        Self {
            rows: [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1],
                    a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1],
                    a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2],
                ],
            ],
        }
    }

    pub fn is_identityish(&self, epsilon: f32) -> bool {
        (self.rows[0][0] - 1.0).abs() <= epsilon
            && self.rows[0][1].abs() <= epsilon
            && self.rows[0][2].abs() <= epsilon
            && self.rows[1][0].abs() <= epsilon
            && (self.rows[1][1] - 1.0).abs() <= epsilon
            && self.rows[1][2].abs() <= epsilon
    }

    pub fn invert(&self) -> Result<Self> {
        let mat = self.to_mat()?;
        let mut inverted = Mat::default();
        imgproc::invert_affine_transform(&mat, &mut inverted)?;
        Self::from_mat(&inverted)
    }

    pub fn to_mat(&self) -> Result<Mat> {
        let mut mat = Mat::new_rows_cols_with_default(2, 3, core::CV_32FC1, Scalar::default())?;
        let data = mat.data_typed_mut::<f32>()?;
        data.copy_from_slice(&[
            self.rows[0][0],
            self.rows[0][1],
            self.rows[0][2],
            self.rows[1][0],
            self.rows[1][1],
            self.rows[1][2],
        ]);
        Ok(mat)
    }

    pub fn from_mat(mat: &Mat) -> Result<Self> {
        if mat.rows() != 2 || mat.cols() != 3 || mat.channels() != 1 {
            return Err(VrifaError::Shape(format!(
                "expected 2x3 single-channel affine matrix, got {}x{} with {} channel(s)",
                mat.rows(),
                mat.cols(),
                mat.channels()
            )));
        }
        let values = mat.data_typed::<f32>()?;
        if values.len() < 6 {
            return Err(VrifaError::Shape(
                "affine matrix did not contain six f32 values".to_string(),
            ));
        }
        Ok(Self {
            rows: [
                [values[0], values[1], values[2]],
                [values[3], values[4], values[5]],
            ],
        })
    }
}

pub fn apply_warp(reference: &Array3<f32>, matrix: &AffineWarp) -> Result<Array3<f32>> {
    let (height, width, channels) = reference.dim();
    if channels == 0 {
        return Err(VrifaError::Shape(
            "warp requires at least one channel".to_string(),
        ));
    }
    let mut output = Array3::<f32>::zeros((height, width, channels));
    for channel in 0..channels {
        let plane = reference.slice(ndarray::s![.., .., channel]).to_owned();
        let warped = apply_warp_plane(&plane, matrix)?;
        output
            .slice_mut(ndarray::s![.., .., channel])
            .assign(&warped);
    }
    Ok(output)
}

pub fn apply_warp_plane(reference: &Array2<f32>, matrix: &AffineWarp) -> Result<Array2<f32>> {
    let (height, width) = reference.dim();
    let src = cvutil::array2_f32_to_mat(reference)?;
    let transform = matrix.to_mat()?;
    let mut dst = Mat::default();
    imgproc::warp_affine(
        &src,
        &mut dst,
        &transform,
        Size::new(width as i32, height as i32),
        imgproc::INTER_CUBIC,
        core::BORDER_REPLICATE,
        Scalar::default(),
    )?;
    cvutil::mat_to_array2_f32(&dst)
}

#[cfg(test)]
mod tests {
    use super::AffineWarp;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn center_displacement_matches_translation() {
        let warp = AffineWarp::from_translation(-5.0, 3.0);
        let (dx, dy) = warp.center_displacement(1920, 1080);
        assert_close(dx, -5.0);
        assert_close(dy, 3.0);
    }

    #[test]
    fn compose_applies_rhs_then_self() {
        let base = AffineWarp {
            rows: [[1.0, 0.1, -2.0], [0.0, 1.0, 4.0]],
        };
        let delta = AffineWarp::from_translation(3.0, -1.0);
        let composed = base.compose(&delta);
        let (x, y) = composed.map_point(10.0, 20.0);
        assert_close(x, 12.9);
        assert_close(y, 23.0);
    }
}
