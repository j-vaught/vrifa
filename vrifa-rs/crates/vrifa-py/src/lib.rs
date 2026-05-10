use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;
use vrifa_core::colorspace::{convert_frame_to_colorspace, ColorSpace};
use vrifa_core::morphology::MorphShape;
use vrifa_core::{detect_front as detect_front_core, DetectFrontParams};

#[pyclass(name = "Config", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyConfig {
    #[pyo3(get, set)]
    pub video_path: String,
    #[pyo3(get, set)]
    pub output_dir: String,
    #[pyo3(get, set)]
    pub write_videos: bool,
    #[pyo3(get, set)]
    pub roi_margin: f32,
    #[pyo3(get, set)]
    pub annotation_formats: Vec<String>,
    #[pyo3(get, set)]
    pub write_mask_pngs: bool,
    #[pyo3(get, set)]
    pub write_overlay_pngs: bool,
    #[pyo3(get, set)]
    pub write_heatmap_pngs: bool,
    #[pyo3(get, set)]
    pub colorspace: String,
    #[pyo3(get, set)]
    pub lock_frames: usize,
    #[pyo3(get, set)]
    pub darken_only: bool,
    #[pyo3(get, set)]
    pub peak_reference: bool,
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (
        video_path,
        output_dir = "flow_front_outputs".to_string(),
        write_videos = false,
        roi_margin = 0.15,
        annotation_formats = vec!["coco".to_string()]
    ))]
    fn new(
        video_path: String,
        output_dir: String,
        write_videos: bool,
        roi_margin: f32,
        annotation_formats: Vec<String>,
    ) -> Self {
        Self {
            video_path,
            output_dir,
            write_videos,
            roi_margin,
            annotation_formats,
            write_mask_pngs: false,
            write_overlay_pngs: false,
            write_heatmap_pngs: false,
            colorspace: "CIELAB".to_string(),
            lock_frames: 3,
            darken_only: true,
            peak_reference: true,
        }
    }
}

impl PyConfig {
    fn to_config(&self) -> PyResult<vrifa_cli::Config> {
        let colorspace = ColorSpace::parse(&self.colorspace)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut config = vrifa_cli::Config {
            video_path: PathBuf::from(&self.video_path),
            output_dir: PathBuf::from(&self.output_dir),
            roi_margin: self.roi_margin,
            write_videos: self.write_videos,
            annotation_formats: self.annotation_formats.clone(),
            write_mask_pngs: self.write_mask_pngs,
            write_overlay_pngs: self.write_overlay_pngs,
            write_heatmap_pngs: self.write_heatmap_pngs,
            colorspace,
            lock_frames: self.lock_frames,
            darken_only: self.darken_only,
            peak_reference: self.peak_reference,
            ..vrifa_cli::Config::default()
        };
        if self.write_videos {
            config.write_mask_video = true;
            config.write_overlay_video = true;
            config.write_heatmap_video = true;
        }
        config.channel_weights = vec![1.0; colorspace.channel_count()];
        Ok(config)
    }
}

#[pyfunction]
fn run(config: &PyConfig) -> PyResult<()> {
    vrifa_cli::run_binding_config(config.to_config()?)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction(name = "convert_frame_to_colorspace")]
fn convert_frame_to_colorspace_py<'py>(
    py: Python<'py>,
    frame_bgr: PyReadonlyArray3<'py, u8>,
    colorspace: &str,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let colorspace =
        ColorSpace::parse(colorspace).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let frame = frame_bgr.as_array().to_owned();
    let converted = convert_frame_to_colorspace(&frame, colorspace)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(converted.into_pyarray(py))
}

#[pyfunction(name = "detect_front")]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    frame_converted,
    reference_converted,
    roi_mask,
    blur = "gaussian:9",
    pre_delta_blur = "none",
    morph_kernel = 13,
    min_area = 400,
    manual_threshold = None,
    percentile_threshold = None,
    threshold_offset = -30.0,
    channel_weights = None,
    morph_shape = "ellipse",
    morph_close_iterations = 1,
    morph_open_iterations = 1,
    darken_only = true,
    peak_brightness_map = None
))]
fn detect_front_py<'py>(
    py: Python<'py>,
    frame_converted: PyReadonlyArray3<'py, f32>,
    reference_converted: PyReadonlyArray3<'py, f32>,
    roi_mask: PyReadonlyArray2<'py, u8>,
    blur: &str,
    pre_delta_blur: &str,
    morph_kernel: usize,
    min_area: usize,
    manual_threshold: Option<f32>,
    percentile_threshold: Option<f32>,
    threshold_offset: f32,
    channel_weights: Option<Vec<f32>>,
    morph_shape: &str,
    morph_close_iterations: usize,
    morph_open_iterations: usize,
    darken_only: bool,
    peak_brightness_map: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray3<u8>>)> {
    let frame = frame_converted.as_array().to_owned();
    let reference = reference_converted.as_array().to_owned();
    let roi = roi_mask.as_array().to_owned();
    let peak = peak_brightness_map
        .as_ref()
        .map(|array| array.as_array().to_owned());
    let post_blur = vrifa_core::BlurSpec::parse(blur)
        .map_err(|err| PyValueError::new_err(format!("invalid blur: {err}")))?;
    let pre_delta_blur = vrifa_core::BlurSpec::parse(pre_delta_blur)
        .map_err(|err| PyValueError::new_err(format!("invalid pre_delta_blur: {err}")))?;
    let params = DetectFrontParams {
        post_blur,
        pre_delta_blur,
        morph_kernel,
        min_area,
        manual_threshold,
        percentile_threshold,
        threshold_offset,
        channel_weights: channel_weights.unwrap_or_else(|| vec![1.0; frame.dim().2]),
        morph_shape: MorphShape::parse(morph_shape),
        morph_close_iterations,
        morph_open_iterations,
        darken_only,
    };
    let (mask, heatmap) = detect_front_core(&frame, &reference, &roi, &params, peak.as_ref())
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok((mask.into_pyarray(py), heatmap.into_pyarray(py)))
}

#[pymodule]
fn vrifa_py(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<PyConfig>()?;
    module.add_function(wrap_pyfunction!(run, module)?)?;

    let core = PyModule::new(py, "core")?;
    core.add_function(wrap_pyfunction!(convert_frame_to_colorspace_py, &core)?)?;
    core.add_function(wrap_pyfunction!(detect_front_py, &core)?)?;
    module.add_submodule(&core)?;
    Ok(())
}
