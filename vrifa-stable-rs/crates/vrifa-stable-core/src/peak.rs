use crate::{Result, VrifaError};
use ndarray::{Array2, Array3};

pub fn update_peak_brightness(
    frame_converted: &Array3<f32>,
    previous_peak: Option<&Array2<f32>>,
) -> Result<Array2<f32>> {
    let (height, width, channels) = frame_converted.dim();
    if channels == 0 {
        return Err(VrifaError::Shape("frame has no channels".to_string()));
    }
    if let Some(previous) = previous_peak {
        if previous.dim() != (height, width) {
            return Err(VrifaError::Shape(
                "peak map shape does not match frame".to_string(),
            ));
        }
    }

    let mut peak = previous_peak
        .cloned()
        .unwrap_or_else(|| Array2::<f32>::zeros((height, width)));
    for y in 0..height {
        for x in 0..width {
            peak[(y, x)] = peak[(y, x)].max(frame_converted[(y, x, 0)]);
        }
    }
    Ok(peak)
}
