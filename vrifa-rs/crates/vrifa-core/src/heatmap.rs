use crate::{cvutil, Result};
use ndarray::{Array2, Array3};
use opencv::imgproc;

pub fn apply_turbo_colormap(delta_norm: &Array2<u8>) -> Result<Array3<u8>> {
    let src = cvutil::array2_u8_to_mat(delta_norm)?;
    let mut dst = opencv::core::Mat::default();
    imgproc::apply_color_map(&src, &mut dst, imgproc::COLORMAP_TURBO)?;
    cvutil::mat_to_array3_u8(&dst)
}
