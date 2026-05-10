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
    Euclidean,
    Affine,
}

impl MotionModel {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "translation" => Ok(Self::Translation),
            "euclidean" => Ok(Self::Euclidean),
            "affine" => Ok(Self::Affine),
            other => Err(VrifaError::InvalidParameter(format!(
                "unsupported motion model: {other}"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Translation => "translation",
            Self::Euclidean => "euclidean",
            Self::Affine => "affine",
        }
    }

    fn opencv_code(self) -> i32 {
        match self {
            Self::Translation => video::MOTION_TRANSLATION,
            Self::Euclidean => video::MOTION_EUCLIDEAN,
            Self::Affine => video::MOTION_AFFINE,
        }
    }

    fn candidate_models(self) -> &'static [Self] {
        const TRANSLATION_ONLY: [MotionModel; 1] = [MotionModel::Translation];
        const EUCLIDEAN_LADDER: [MotionModel; 2] =
            [MotionModel::Translation, MotionModel::Euclidean];
        const AFFINE_LADDER: [MotionModel; 3] = [
            MotionModel::Translation,
            MotionModel::Euclidean,
            MotionModel::Affine,
        ];
        match self {
            Self::Translation => &TRANSLATION_ONLY,
            Self::Euclidean => &EUCLIDEAN_LADDER,
            Self::Affine => &AFFINE_LADDER,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegistrationFit {
    pub warp: AffineWarp,
    pub score: f32,
    pub model: MotionModel,
}

pub fn fit_affine_warp(
    curr_lab: &Array3<f32>,
    ref_lab: &Array3<f32>,
    init_matrix: &AffineWarp,
    motion_model: MotionModel,
    registration_mask: Option<&Array2<u8>>,
) -> Result<RegistrationFit> {
    const SCORE_TOLERANCE: f32 = 0.0015;

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
    let criteria = core::TermCriteria::new(
        core::TermCriteria_Type::COUNT as i32 | core::TermCriteria_Type::EPS as i32,
        100,
        1e-4,
    )?;
    let (height, width, _) = curr_lab.dim();
    let mask_mat = registration_mask
        .map(cvutil::array2_u8_to_mat)
        .transpose()?;

    let mut fits = Vec::new();
    let mut last_error = None;
    for candidate_model in motion_model.candidate_models() {
        match fit_once(
            &reference,
            &curr,
            init_matrix,
            *candidate_model,
            width,
            height,
            criteria,
            mask_mat.as_ref(),
        ) {
            Ok(fit) => fits.push(fit),
            Err(err) => last_error = Some(err),
        }
    }

    let Some(best_score) = fits.iter().map(|fit| fit.score).reduce(f32::max) else {
        return Err(last_error.unwrap_or_else(|| {
            VrifaError::InvalidParameter("registration failed without producing a fit".to_string())
        }));
    };

    fits.into_iter()
        .find(|fit| best_score - fit.score <= SCORE_TOLERANCE)
        .ok_or_else(|| {
            VrifaError::InvalidParameter(
                "registration candidates were scored but none satisfied the selection rule"
                    .to_string(),
            )
        })
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

fn fit_once(
    reference: &core::Mat,
    curr: &core::Mat,
    init_matrix: &AffineWarp,
    motion_model: MotionModel,
    width: usize,
    height: usize,
    criteria: core::TermCriteria,
    registration_mask: Option<&core::Mat>,
) -> Result<RegistrationFit> {
    let mut warp = project_init_warp(init_matrix, motion_model, width, height).to_mat()?;
    let score = if let Some(mask) = registration_mask {
        video::find_transform_ecc(
            reference,
            curr,
            &mut warp,
            motion_model.opencv_code(),
            criteria,
            mask,
            5,
        )?
    } else {
        video::find_transform_ecc(
            reference,
            curr,
            &mut warp,
            motion_model.opencv_code(),
            criteria,
            &core::no_array(),
            5,
        )?
    };
    Ok(RegistrationFit {
        warp: AffineWarp::from_mat(&warp)?,
        score: score as f32,
        model: motion_model,
    })
}

fn project_init_warp(
    init_matrix: &AffineWarp,
    motion_model: MotionModel,
    width: usize,
    height: usize,
) -> AffineWarp {
    let (center_dx, center_dy) = init_matrix.center_displacement(width, height);
    match motion_model {
        MotionModel::Translation => AffineWarp::from_translation(center_dx, center_dy),
        MotionModel::Euclidean => {
            let a = init_matrix.rows[0][0];
            let b = init_matrix.rows[0][1];
            let c = init_matrix.rows[1][0];
            let d = init_matrix.rows[1][1];
            let theta = (c - b).atan2(a + d);
            let (sin_theta, cos_theta) = theta.sin_cos();
            let center_x = (width.saturating_sub(1) as f32) * 0.5;
            let center_y = (height.saturating_sub(1) as f32) * 0.5;
            let mapped_center_x = cos_theta * center_x - sin_theta * center_y;
            let mapped_center_y = sin_theta * center_x + cos_theta * center_y;
            let tx = center_dx + center_x - mapped_center_x;
            let ty = center_dy + center_y - mapped_center_y;
            AffineWarp {
                rows: [[cos_theta, -sin_theta, tx], [sin_theta, cos_theta, ty]],
            }
        }
        MotionModel::Affine => init_matrix.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{project_init_warp, MotionModel};
    use crate::warp::AffineWarp;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn translation_projection_preserves_center_shift() {
        let warp = AffineWarp {
            rows: [[1.0, 0.05, 4.0], [0.02, 1.0, -3.0]],
        };
        let projected = project_init_warp(&warp, MotionModel::Translation, 1920, 1080);
        let (actual_dx, actual_dy) = projected.center_displacement(1920, 1080);
        let (expected_dx, expected_dy) = warp.center_displacement(1920, 1080);
        assert_close(actual_dx, expected_dx);
        assert_close(actual_dy, expected_dy);
    }

    #[test]
    fn euclidean_projection_preserves_center_shift() {
        let warp = AffineWarp {
            rows: [[1.01, -0.04, 8.0], [0.03, 0.99, -6.0]],
        };
        let projected = project_init_warp(&warp, MotionModel::Euclidean, 1920, 1080);
        let (actual_dx, actual_dy) = projected.center_displacement(1920, 1080);
        let (expected_dx, expected_dy) = warp.center_displacement(1920, 1080);
        assert_close(actual_dx, expected_dx);
        assert_close(actual_dy, expected_dy);
    }
}
