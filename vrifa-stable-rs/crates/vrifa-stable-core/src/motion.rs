use crate::{cvutil, Result, VrifaError};
use ndarray::Array3;
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
