pub mod colorspace;
pub mod contours;
pub mod cvutil;
pub mod delta;
pub mod heatmap;
pub mod lock;
pub mod morphology;
pub mod overlay;
pub mod peak;
pub mod reference;
pub mod roi;
pub mod sampling;
pub mod threshold;

use ndarray::{Array2, Array3};
use thiserror::Error;

pub use colorspace::ColorSpace;
pub use contours::AnnotationBox;
pub use morphology::{MorphShape, MorphologyParams};

#[derive(Debug, Error)]
pub enum VrifaError {
    #[error("OpenCV error: {0}")]
    OpenCv(#[from] opencv::Error),
    #[error("invalid shape: {0}")]
    Shape(String),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, VrifaError>;

#[derive(Clone, Debug)]
pub struct DetectFrontParams {
    pub blur_kernel: usize,
    pub morph_kernel: usize,
    pub min_area: usize,
    pub manual_threshold: Option<f32>,
    pub percentile_threshold: Option<f32>,
    pub threshold_offset: f32,
    pub channel_weights: Vec<f32>,
    pub blur_enabled: bool,
    pub morph_shape: MorphShape,
    pub morph_close_iterations: usize,
    pub morph_open_iterations: usize,
    pub darken_only: bool,
}

#[derive(Clone, Debug)]
pub struct DetectFrontDebug {
    pub delta: Array2<f32>,
    pub delta_blur: Array2<f32>,
    pub delta_norm: Array2<u8>,
    pub binary: Array2<u8>,
    pub mask: Array2<u8>,
    pub heatmap: Array3<u8>,
}

impl Default for DetectFrontParams {
    fn default() -> Self {
        Self {
            blur_kernel: 9,
            morph_kernel: 13,
            min_area: 400,
            manual_threshold: None,
            percentile_threshold: None,
            threshold_offset: -30.0,
            channel_weights: vec![1.0, 1.0, 1.0],
            blur_enabled: true,
            morph_shape: MorphShape::Ellipse,
            morph_close_iterations: 1,
            morph_open_iterations: 1,
            darken_only: true,
        }
    }
}

pub fn detect_front(
    frame_converted: &Array3<f32>,
    reference_converted: &Array3<f32>,
    roi_mask: &Array2<u8>,
    params: &DetectFrontParams,
    peak_brightness_map: Option<&Array2<f32>>,
) -> Result<(Array2<u8>, Array3<u8>)> {
    let debug = detect_front_debug(
        frame_converted,
        reference_converted,
        roi_mask,
        params,
        peak_brightness_map,
    )?;
    Ok((debug.mask, debug.heatmap))
}

pub fn detect_front_debug(
    frame_converted: &Array3<f32>,
    reference_converted: &Array3<f32>,
    roi_mask: &Array2<u8>,
    params: &DetectFrontParams,
    peak_brightness_map: Option<&Array2<f32>>,
) -> Result<DetectFrontDebug> {
    let delta = delta::compute_delta(
        frame_converted,
        reference_converted,
        roi_mask,
        &params.channel_weights,
        params.darken_only,
        peak_brightness_map,
    )?;
    let morphology = morphology::detect_mask_from_delta_debug(
        &delta,
        roi_mask,
        &MorphologyParams {
            blur_kernel: params.blur_kernel,
            morph_kernel: params.morph_kernel,
            min_area: params.min_area,
            manual_threshold: params.manual_threshold,
            percentile_threshold: params.percentile_threshold,
            threshold_offset: params.threshold_offset,
            blur_enabled: params.blur_enabled,
            morph_shape: params.morph_shape,
            morph_close_iterations: params.morph_close_iterations,
            morph_open_iterations: params.morph_open_iterations,
        },
    )?;
    let heatmap = heatmap::apply_turbo_colormap(&morphology.delta_norm)?;
    Ok(DetectFrontDebug {
        delta,
        delta_blur: morphology.delta_blur,
        delta_norm: morphology.delta_norm,
        binary: morphology.binary,
        mask: morphology.mask,
        heatmap,
    })
}
