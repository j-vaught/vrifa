use crate::{cvutil, Result, VrifaError};
use ndarray::{Array2, Array3};
use opencv::core::{self, Mat, Size};
use opencv::imgproc;

const MOTION_DOWNSAMPLE_SIZE: i32 = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionEstimate {
    pub dx: f32,
    pub dy: f32,
    pub confidence: f32,
}

pub fn estimate_translation(
    curr_lab: &Array3<f32>,
    prev_lab: &Array3<f32>,
) -> Result<MotionEstimate> {
    if curr_lab.dim() != prev_lab.dim() {
        return Err(VrifaError::Shape(
            "motion estimation frames must share the same shape".to_string(),
        ));
    }
    if curr_lab.dim().2 == 0 {
        return Err(VrifaError::Shape(
            "motion estimation requires at least one channel".to_string(),
        ));
    }

    let (height, width, _) = curr_lab.dim();
    let curr = downsample_l_channel(curr_lab)?;
    let prev = downsample_l_channel(prev_lab)?;
    let curr = normalize_with_hanning(&curr)?;
    let prev = normalize_with_hanning(&prev)?;
    let mut response = 0.0f64;
    let shift = imgproc::phase_correlate(&curr, &prev, &core::no_array(), &mut response)?;
    let scale_x = width as f32 / MOTION_DOWNSAMPLE_SIZE as f32;
    let scale_y = height as f32 / MOTION_DOWNSAMPLE_SIZE as f32;
    Ok(MotionEstimate {
        dx: shift.x as f32 * scale_x,
        dy: shift.y as f32 * scale_y,
        confidence: response as f32,
    })
}

fn downsample_l_channel(frame: &Array3<f32>) -> Result<Mat> {
    let src = cvutil::array3_f32_channel_to_mat(frame, 0)?;
    let mut dst = Mat::default();
    imgproc::resize(
        &src,
        &mut dst,
        Size::new(MOTION_DOWNSAMPLE_SIZE, MOTION_DOWNSAMPLE_SIZE),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    Ok(dst)
}

fn normalize_with_hanning(input: &Mat) -> Result<Mat> {
    let mut array = cvutil::mat_to_array2_f32(input)?;
    standardize_in_place(&mut array);
    apply_hanning_in_place(&mut array);
    cvutil::array2_f32_to_mat(&array)
}

fn standardize_in_place(array: &mut Array2<f32>) {
    let len = array.len();
    if len == 0 {
        return;
    }
    let mean = array.iter().copied().sum::<f32>() / len as f32;
    let variance = array
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / len as f32;
    let stddev = variance.sqrt().max(1e-6);
    for value in array.iter_mut() {
        *value = (*value - mean) / stddev;
    }
}

fn apply_hanning_in_place(array: &mut Array2<f32>) {
    let (height, width) = array.dim();
    if height <= 1 || width <= 1 {
        return;
    }
    for y in 0..height {
        let wy = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * y as f32) / (height - 1) as f32).cos();
        for x in 0..width {
            let wx =
                0.5 - 0.5 * ((2.0 * std::f32::consts::PI * x as f32) / (width - 1) as f32).cos();
            array[(y, x)] *= wx * wy;
        }
    }
}
