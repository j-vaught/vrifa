use crate::{cvutil, Result, VrifaError};
use ndarray::{Array2, Array3};
use opencv::core::{self, Point, Size};
use opencv::imgproc;

pub fn create_overlay(frame_bgr: &Array3<u8>, mask: &Array2<u8>) -> Result<Array3<u8>> {
    let (height, width, channels) = frame_bgr.dim();
    if channels != 3 {
        return Err(VrifaError::Shape(
            "overlay frame must be BGR with 3 channels".to_string(),
        ));
    }
    if mask.dim() != (height, width) {
        return Err(VrifaError::Shape(
            "mask shape does not match frame".to_string(),
        ));
    }

    let mask_mat = cvutil::array2_u8_to_mat(mask)?;
    let kernel = imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(5, 5))?;
    let mut edges = opencv::core::Mat::default();
    imgproc::morphology_ex(
        &mask_mat,
        &mut edges,
        imgproc::MORPH_GRADIENT,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    let edges = cvutil::mat_to_array2_u8(&edges)?;

    let mut overlay = frame_bgr.clone();
    for ((y, x), edge) in edges.indexed_iter() {
        if *edge > 0 {
            overlay[(y, x, 0)] = 0;
            overlay[(y, x, 1)] = 0;
            overlay[(y, x, 2)] = 255;
        }
    }
    Ok(overlay)
}
