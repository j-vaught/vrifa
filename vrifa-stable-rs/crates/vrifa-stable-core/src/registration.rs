use crate::cvutil;
use crate::warp::AffineWarp;
use crate::{Result, VrifaError};
use ndarray::Array2;
use ndarray::Array3;
use opencv::core;
use opencv::imgproc;
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
    registration_mask: Option<&Array2<u8>>,
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

    let curr = cvutil::array2_f32_to_mat(&registration_signal(curr_lab)?)?;
    let reference = cvutil::array2_f32_to_mat(&registration_signal(ref_lab)?)?;
    let mut warp = init_matrix.to_mat()?;
    let criteria = core::TermCriteria::new(
        core::TermCriteria_Type::COUNT as i32 | core::TermCriteria_Type::EPS as i32,
        100,
        1e-4,
    )?;
    if let Some(mask) = registration_mask {
        let mask = cvutil::array2_u8_to_mat(mask)?;
        video::find_transform_ecc(
            &reference,
            &curr,
            &mut warp,
            motion_model.opencv_code(),
            criteria,
            &mask,
            5,
        )?;
    } else {
        video::find_transform_ecc(
            &reference,
            &curr,
            &mut warp,
            motion_model.opencv_code(),
            criteria,
            &core::no_array(),
            5,
        )?;
    }
    AffineWarp::from_mat(&warp)
}

pub fn strong_gradient_mask(frame_lab: &Array3<f32>, percentile: f32) -> Result<Array2<u8>> {
    let signal = registration_signal(frame_lab)?;
    let mut values = signal
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(Array2::<u8>::zeros(signal.dim()));
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let percentile = percentile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f32 * percentile).round() as usize;
    let threshold = values[index];
    Ok(signal.mapv(|value| if value >= threshold { 255 } else { 0 }))
}

fn registration_signal(frame_lab: &Array3<f32>) -> Result<Array2<f32>> {
    if frame_lab.dim().2 == 0 {
        return Err(VrifaError::Shape(
            "registration requires at least one channel".to_string(),
        ));
    }

    let source = cvutil::array3_f32_channel_to_mat(frame_lab, 0)?;
    let mut blurred = core::Mat::default();
    imgproc::gaussian_blur(
        &source,
        &mut blurred,
        core::Size::new(0, 0),
        1.5,
        1.5,
        core::BORDER_REPLICATE,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut grad_x = core::Mat::default();
    let mut grad_y = core::Mat::default();
    imgproc::sobel(
        &blurred,
        &mut grad_x,
        core::CV_32FC1,
        1,
        0,
        3,
        1.0,
        0.0,
        core::BORDER_REPLICATE,
    )?;
    imgproc::sobel(
        &blurred,
        &mut grad_y,
        core::CV_32FC1,
        0,
        1,
        3,
        1.0,
        0.0,
        core::BORDER_REPLICATE,
    )?;

    let grad_x = cvutil::mat_to_array2_f32(&grad_x)?;
    let grad_y = cvutil::mat_to_array2_f32(&grad_y)?;
    let mut magnitude = Array2::<f32>::zeros(grad_x.dim());
    for ((y, x), value) in magnitude.indexed_iter_mut() {
        *value = grad_x[(y, x)].hypot(grad_y[(y, x)]);
    }
    Ok(magnitude)
}
