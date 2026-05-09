use crate::{cvutil, Result};
use ndarray::Array2;
use opencv::imgproc;

pub fn choose_threshold(
    delta_norm: &Array2<u8>,
    roi_mask: &Array2<u8>,
    manual_threshold: Option<f32>,
    percentile: Option<f32>,
    offset: f32,
) -> Result<f32> {
    let mut threshold_value = if let Some(percentile) = percentile {
        let mut values: Vec<u8> = delta_norm
            .iter()
            .zip(roi_mask.iter())
            .filter_map(|(value, mask)| (*mask > 0).then_some(*value))
            .collect();
        if values.is_empty() {
            otsu_threshold(delta_norm)? as f32
        } else {
            values.sort_unstable();
            numpy_linear_percentile(&values, percentile.clamp(0.0, 100.0))
        }
    } else if let Some(manual) = manual_threshold {
        manual
    } else {
        otsu_threshold(delta_norm)? as f32
    };

    threshold_value += offset;
    Ok(threshold_value.clamp(0.0, 255.0))
}

fn otsu_threshold(delta_norm: &Array2<u8>) -> Result<f64> {
    let src = cvutil::array2_u8_to_mat(delta_norm)?;
    let mut dst = opencv::core::Mat::default();
    Ok(imgproc::threshold(
        &src,
        &mut dst,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?)
}

fn numpy_linear_percentile(sorted: &[u8], percentile: f32) -> f32 {
    if sorted.len() == 1 {
        return sorted[0] as f32;
    }
    let rank = percentile / 100.0 * (sorted.len() - 1) as f32;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower] as f32
    } else {
        let weight = rank - lower as f32;
        sorted[lower] as f32 * (1.0 - weight) + sorted[upper] as f32 * weight
    }
}
