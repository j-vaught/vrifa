use crate::cvutil;
use crate::warp::AffineWarp;
use crate::{Result, VrifaError};
use ndarray::Array3;
use opencv::core;
use opencv::video;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MotionModel {
    Translation,
    Affine,
}

impl MotionModel {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "translation" => Ok(Self::Translation),
            "affine" => Ok(Self::Affine),
            other => Err(VrifaError::InvalidParameter(format!(
                "unsupported motion model: {other}"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Translation => "translation",
            Self::Affine => "affine",
        }
    }

    fn opencv_code(self) -> i32 {
        match self {
            Self::Translation => video::MOTION_TRANSLATION,
            Self::Affine => video::MOTION_AFFINE,
        }
    }
}

pub fn fit_affine_warp(
    curr_lab: &Array3<f32>,
    ref_lab: &Array3<f32>,
    init_matrix: &AffineWarp,
    motion_model: MotionModel,
) -> Result<AffineWarp> {
    if curr_lab.dim() != ref_lab.dim() {
        return Err(VrifaError::Shape(
            "registration frames must share the same shape".to_string(),
        ));
    }
    if curr_lab.dim().2 == 0 {
        return Err(VrifaError::Shape(
            "registration requires at least one channel".to_string(),
        ));
    }

    let curr = cvutil::array3_f32_channel_to_mat(curr_lab, 0)?;
    let reference = cvutil::array3_f32_channel_to_mat(ref_lab, 0)?;
    let mut warp = init_matrix.to_mat()?;
    let criteria = core::TermCriteria::new(
        core::TermCriteria_Type::COUNT as i32 | core::TermCriteria_Type::EPS as i32,
        100,
        1e-4,
    )?;
    video::find_transform_ecc(
        &curr,
        &reference,
        &mut warp,
        motion_model.opencv_code(),
        criteria,
        &core::no_array(),
        5,
    )?;
    AffineWarp::from_mat(&warp)
}
