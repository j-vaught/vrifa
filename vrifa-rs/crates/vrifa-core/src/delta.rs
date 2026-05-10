use crate::{cvutil, Result, VrifaError};
use ndarray::{Array2, Array3};
use opencv::core::Size;
use opencv::imgproc;

pub fn compute_delta(
    frame_converted: &Array3<f32>,
    reference_converted: &Array3<f32>,
    roi_mask: &Array2<u8>,
    channel_weights: &[f32],
    darken_only: bool,
    peak_brightness_map: Option<&Array2<f32>>,
) -> Result<Array2<f32>> {
    let (height, width, channels) = frame_converted.dim();
    if reference_converted.dim() != (height, width, channels) {
        return Err(VrifaError::Shape(
            "frame and reference shapes differ".to_string(),
        ));
    }
    if roi_mask.dim() != (height, width) {
        return Err(VrifaError::Shape(
            "ROI mask shape does not match frame".to_string(),
        ));
    }
    if channel_weights.len() != channels {
        return Err(VrifaError::InvalidParameter(format!(
            "expected {channels} channel weights, got {}",
            channel_weights.len()
        )));
    }
    if let Some(peak) = peak_brightness_map {
        if peak.dim() != (height, width) {
            return Err(VrifaError::Shape(
                "peak brightness map shape does not match frame".to_string(),
            ));
        }
    }

    let mut delta = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut value = if darken_only {
                if let Some(peak) = peak_brightness_map {
                    (peak[(y, x)] - frame_converted[(y, x, 0)]) * channel_weights[0]
                } else {
                    (reference_converted[(y, x, 0)] - frame_converted[(y, x, 0)])
                        * channel_weights[0]
                }
            } else {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    let diff = (frame_converted[(y, x, c)] - reference_converted[(y, x, c)])
                        * channel_weights[c];
                    sum += diff * diff;
                }
                sum.sqrt()
            };
            if darken_only && value < 0.0 {
                value = 0.0;
            }
            delta[(y, x)] = value * roi_mask[(y, x)] as f32;
        }
    }
    Ok(delta)
}

pub fn blur_frame(frame_converted: &Array3<f32>, kernel: usize) -> Result<Array3<f32>> {
    let mut kernel = kernel;
    if kernel <= 1 {
        return Ok(frame_converted.clone());
    }
    if kernel % 2 == 0 {
        kernel += 1;
    }
    let (height, width, channels) = frame_converted.dim();
    let mut blurred = Array3::<f32>::zeros((height, width, channels));
    for channel in 0..channels {
        let plane = blur_plane(
            &frame_converted
                .slice(ndarray::s![.., .., channel])
                .to_owned(),
            kernel,
        )?;
        blurred
            .slice_mut(ndarray::s![.., .., channel])
            .assign(&plane);
    }
    Ok(blurred)
}

pub fn blur_plane(plane: &Array2<f32>, kernel: usize) -> Result<Array2<f32>> {
    let mut kernel = kernel;
    if kernel <= 1 {
        return Ok(plane.clone());
    }
    if kernel % 2 == 0 {
        kernel += 1;
    }
    let src = cvutil::array2_f32_to_mat(plane)?;
    let mut dst = opencv::core::Mat::default();
    imgproc::gaussian_blur_def(&src, &mut dst, Size::new(kernel as i32, kernel as i32), 0.0)?;
    cvutil::mat_to_array2_f32(&dst)
}
