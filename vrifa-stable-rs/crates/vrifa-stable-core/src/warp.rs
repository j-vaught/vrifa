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
    if reference.dim().2 == 0 {
        return Err(VrifaError::Shape(
            "warp requires at least one channel".to_string(),
        ));
    }
    let (height, width, _) = reference.dim();
    let src = cvutil::array3_f32_to_mat(reference)?;
    let transform = matrix.to_mat()?;
    let mut dst = Mat::default();
    imgproc::warp_affine(
        &src,
        &mut dst,
        &transform,
        Size::new(width as i32, height as i32),
        imgproc::INTER_LINEAR,
        core::BORDER_REPLICATE,
        Scalar::default(),
    )?;
    cvutil::mat_to_array3_f32(&dst)
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
        imgproc::INTER_LINEAR,
        core::BORDER_REPLICATE,
        Scalar::default(),
    )?;
    cvutil::mat_to_array2_f32(&dst)
}
