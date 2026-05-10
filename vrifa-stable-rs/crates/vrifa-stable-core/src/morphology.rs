use crate::{cvutil, threshold, Result};
use ndarray::Array2;
use opencv::core::{self, Point, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MorphShape {
    Ellipse,
    Rect,
    Cross,
}

impl MorphShape {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "rect" => Self::Rect,
            "cross" => Self::Cross,
            _ => Self::Ellipse,
        }
    }

    fn opencv_code(self) -> i32 {
        match self {
            Self::Ellipse => imgproc::MORPH_ELLIPSE,
            Self::Rect => imgproc::MORPH_RECT,
            Self::Cross => imgproc::MORPH_CROSS,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Ellipse => "ellipse",
            Self::Rect => "rect",
            Self::Cross => "cross",
        }
    }
}

#[derive(Clone, Debug)]
pub struct MorphologyParams {
    pub blur_kernel: usize,
    pub morph_kernel: usize,
    pub min_area: usize,
    pub manual_threshold: Option<f32>,
    pub percentile_threshold: Option<f32>,
    pub threshold_offset: f32,
    pub blur_enabled: bool,
    pub morph_shape: MorphShape,
    pub morph_close_iterations: usize,
    pub morph_open_iterations: usize,
}

#[derive(Clone, Debug)]
pub struct MorphologyDebug {
    pub delta_blur: Array2<f32>,
    pub delta_norm: Array2<u8>,
    pub binary: Array2<u8>,
    pub mask: Array2<u8>,
}

pub fn detect_mask_from_delta(
    delta: &Array2<f32>,
    roi_mask: &Array2<u8>,
    params: &MorphologyParams,
) -> Result<(Array2<u8>, Array2<u8>)> {
    let debug = detect_mask_from_delta_debug(delta, roi_mask, params)?;
    Ok((debug.mask, debug.delta_norm))
}

pub fn detect_mask_from_delta_debug(
    delta: &Array2<f32>,
    roi_mask: &Array2<u8>,
    params: &MorphologyParams,
) -> Result<MorphologyDebug> {
    let delta_blur = if params.blur_enabled {
        let mut kernel = params.blur_kernel;
        if kernel % 2 == 0 {
            kernel += 1;
        }
        let src = cvutil::array2_f32_to_mat(delta)?;
        let mut dst = opencv::core::Mat::default();
        imgproc::gaussian_blur_def(&src, &mut dst, Size::new(kernel as i32, kernel as i32), 0.0)?;
        cvutil::mat_to_array2_f32(&dst)?
    } else {
        delta.clone()
    };

    let delta_norm = normalize_minmax_to_u8(&delta_blur)?;
    let threshold_value = threshold::choose_threshold(
        &delta_norm,
        roi_mask,
        params.manual_threshold,
        params.percentile_threshold,
        params.threshold_offset,
    )?;

    let norm_mat = cvutil::array2_u8_to_mat(&delta_norm)?;
    let mut binary = opencv::core::Mat::default();
    imgproc::threshold(
        &norm_mat,
        &mut binary,
        threshold_value as f64,
        255.0,
        imgproc::THRESH_BINARY,
    )?;
    let binary_pre_morph = binary.try_clone()?;

    let mut kernel_size = params.morph_kernel + (1 - params.morph_kernel % 2);
    if kernel_size == 0 {
        kernel_size = 1;
    }
    let kernel = imgproc::get_structuring_element_def(
        params.morph_shape.opencv_code(),
        Size::new(kernel_size as i32, kernel_size as i32),
    )?;

    for _ in 0..params.morph_close_iterations {
        let mut next = opencv::core::Mat::default();
        imgproc::morphology_ex(
            &binary,
            &mut next,
            imgproc::MORPH_CLOSE,
            &kernel,
            Point::new(-1, -1),
            1,
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;
        binary = next;
    }
    for _ in 0..params.morph_open_iterations {
        let mut next = opencv::core::Mat::default();
        imgproc::morphology_ex(
            &binary,
            &mut next,
            imgproc::MORPH_OPEN,
            &kernel,
            Point::new(-1, -1),
            1,
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;
        binary = next;
    }

    if params.min_area > 0 {
        binary = filter_min_area(&binary, params.min_area)?;
    }

    Ok(MorphologyDebug {
        delta_blur,
        delta_norm,
        binary: cvutil::mat_to_array2_u8(&binary_pre_morph)?,
        mask: cvutil::mat_to_array2_u8(&binary)?,
    })
}

fn normalize_minmax_to_u8(delta: &Array2<f32>) -> Result<Array2<u8>> {
    let src = cvutil::array2_f32_to_mat(delta)?;
    let mut normalized = opencv::core::Mat::default();
    core::normalize(
        &src,
        &mut normalized,
        0.0,
        255.0,
        core::NORM_MINMAX,
        core::CV_32F,
        &core::no_array(),
    )?;
    let normalized = cvutil::mat_to_array2_f32(&normalized)?;
    Ok(normalized.mapv(|value| value.clamp(0.0, 255.0) as u8))
}

fn filter_min_area(binary: &opencv::core::Mat, min_area: usize) -> Result<opencv::core::Mat> {
    let mut labels = opencv::core::Mat::default();
    let mut stats = opencv::core::Mat::default();
    let mut centroids = opencv::core::Mat::default();
    let num_labels = imgproc::connected_components_with_stats_def(
        binary,
        &mut labels,
        &mut stats,
        &mut centroids,
    )?;

    let rows = binary.rows();
    let cols = binary.cols();
    let mut filtered = opencv::core::Mat::new_rows_cols_with_default(
        rows,
        cols,
        core::CV_8UC1,
        Scalar::default(),
    )?;
    let label_values = labels.data_typed::<i32>()?;
    let stats_values = stats.data_typed::<i32>()?;
    let out = filtered.data_bytes_mut()?;
    let stat_cols = stats.cols() as usize;

    for (offset, label) in label_values.iter().enumerate() {
        let label = *label as usize;
        if label > 0 && label < num_labels as usize {
            let area = stats_values[label * stat_cols + imgproc::CC_STAT_AREA as usize] as usize;
            if area >= min_area {
                out[offset] = 255;
            }
        }
    }
    Ok(filtered)
}
