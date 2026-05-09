use anyhow::{anyhow, bail, Context, Result};
use bytemuck::cast_slice;
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    },
    nvrtc::compile_ptx,
};
use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend, PeakImageBackend, RoiMargins};
use ndarray::{Array2, Array3};
use opencv::core::CV_32F;
use opencv::imgproc;
use opencv::prelude::*;
use std::any::Any;
use std::env;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;
use vrifa_core::colorspace::{convert_frame_to_colorspace, ColorSpace};
use vrifa_core::morphology::MorphShape;

const KERNELS: &str = include_str!("kernels/stage1.cu");

#[derive(Clone)]
pub struct CudaFrameBgr {
    data: CudaSlice<u8>,
    width: usize,
    height: usize,
}

#[derive(Clone)]
pub struct CudaFrameLab {
    data: CudaSlice<u32>,
    width: usize,
    height: usize,
}

#[derive(Clone)]
pub struct CudaPlaneF32 {
    data: CudaSlice<f32>,
    width: usize,
    height: usize,
}

#[derive(Clone)]
pub struct CudaMaskU8 {
    data: CudaSlice<u8>,
    width: usize,
    height: usize,
}

struct CudaBackendInner {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    colorspace: CudaFunction,
    extract_l: CudaFunction,
    peak: CudaFunction,
    roi: CudaFunction,
    delta: CudaFunction,
    blur_norm: CudaFunction,
    reduce_minmax: CudaFunction,
    normalize_u8: CudaFunction,
    threshold_binary: CudaFunction,
    dilate_binary: CudaFunction,
    erode_binary: CudaFunction,
    lab_lut: CudaSlice<u32>,
}

#[derive(Clone, Default)]
pub struct CudaBackend {
    inner: Option<Arc<CudaBackendInner>>,
    init_error: Option<String>,
}

impl CudaBackend {
    pub fn new() -> Self {
        match catch_unwind(AssertUnwindSafe(CudaBackendInner::new)) {
            Ok(Ok(inner)) => Self {
                inner: Some(Arc::new(inner)),
                init_error: None,
            },
            Ok(Err(err)) => Self {
                inner: None,
                init_error: Some(err.to_string()),
            },
            Err(payload) => Self {
                inner: None,
                init_error: Some(format!(
                    "CUDA runtime panicked during initialization: {}",
                    panic_message(payload)
                )),
            },
        }
    }

    pub fn init_error(&self) -> Option<&str> {
        self.init_error.as_deref()
    }

    fn inner(&self) -> Result<&CudaBackendInner> {
        self.inner.as_deref().ok_or_else(|| {
            anyhow!(
                "CUDA backend is unavailable{}",
                self.init_error
                    .as_deref()
                    .map(|value| format!(": {value}"))
                    .unwrap_or_default()
            )
        })
    }
}

impl CudaBackendInner {
    fn new() -> Result<Self> {
        let ordinal = cuda_device_ordinal()?;
        let context = CudaContext::new(ordinal)
            .with_context(|| format!("initializing CUDA context on device {ordinal}"))?;
        let stream = context.default_stream();
        let ptx = compile_ptx(KERNELS).context("compiling fast-vrifa CUDA kernels with NVRTC")?;
        let module = context
            .load_module(ptx)
            .context("loading fast-vrifa CUDA module")?;
        let colorspace = module
            .load_function("bgr_to_lab_lut")
            .context("loading CUDA colorspace kernel")?;
        let extract_l = module
            .load_function("extract_l_plane")
            .context("loading CUDA L-plane kernel")?;
        let peak = module
            .load_function("update_peak_brightness")
            .context("loading CUDA peak kernel")?;
        let roi = module
            .load_function("build_roi_mask")
            .context("loading CUDA ROI kernel")?;
        let delta = module
            .load_function("compute_delta_darken_only")
            .context("loading CUDA delta kernel")?;
        let blur_norm = module
            .load_function("gaussian_blur_f32")
            .context("loading CUDA blur kernel")?;
        let reduce_minmax = module
            .load_function("reduce_minmax_nonnegative")
            .context("loading CUDA min/max kernel")?;
        let normalize_u8 = module
            .load_function("normalize_minmax_u8")
            .context("loading CUDA normalize kernel")?;
        let threshold_binary = module
            .load_function("threshold_binary_u8")
            .context("loading CUDA threshold kernel")?;
        let dilate_binary = module
            .load_function("dilate_binary_u8")
            .context("loading CUDA dilation kernel")?;
        let erode_binary = module
            .load_function("erode_binary_u8")
            .context("loading CUDA erosion kernel")?;
        let host_lut = load_or_build_lab_lut()?;
        let lab_lut = stream
            .clone_htod(&host_lut)
            .context("uploading BGR->CIELAB lookup table to CUDA")?;
        Ok(Self {
            stream,
            _module: module,
            colorspace,
            extract_l,
            peak,
            roi,
            delta,
            blur_norm,
            reduce_minmax,
            normalize_u8,
            threshold_binary,
            dilate_binary,
            erode_binary,
            lab_lut,
        })
    }
}

impl Default for CudaBackendInner {
    fn default() -> Self {
        Self::new().expect("CUDA backend should not use infallible default construction")
    }
}

impl ImageBackend for CudaBackend {
    type DeviceFrameBgr = CudaFrameBgr;
    type DeviceFrameLab = CudaFrameLab;
    type DevicePlaneF32 = CudaPlaneF32;
    type DeviceMaskU8 = CudaMaskU8;

    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn label(&self) -> &'static str {
        if self.inner.is_some() {
            "cuda"
        } else {
            "cuda-unavailable"
        }
    }

    fn status(&self) -> BackendStatus {
        if self.inner.is_some() {
            BackendStatus::Ready
        } else {
            BackendStatus::Unavailable
        }
    }

    fn upload_frame_bgr(&self, frame_bgr: &Array3<u8>) -> Result<Self::DeviceFrameBgr> {
        let inner = self.inner()?;
        let (height, width, channels) = frame_bgr.dim();
        anyhow::ensure!(channels == 3, "expected a 3-channel BGR frame");
        let packed = frame_bgr
            .as_slice_memory_order()
            .ok_or_else(|| anyhow!("BGR frame must be contiguous"))?;
        let data = inner
            .stream
            .clone_htod(packed)
            .context("uploading BGR frame to CUDA")?;
        Ok(CudaFrameBgr {
            data,
            width,
            height,
        })
    }

    fn convert_bgr_to_lab(&self, frame_bgr: &Self::DeviceFrameBgr) -> Result<Self::DeviceFrameLab> {
        let inner = self.inner()?;
        let pixel_count = frame_bgr.width * frame_bgr.height;
        let mut output = inner
            .stream
            .alloc_zeros::<u32>(pixel_count)
            .context("allocating CUDA CIELAB frame")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.stream.launch_builder(&inner.colorspace);
        launch.arg(&frame_bgr.data);
        launch.arg(&inner.lab_lut);
        launch.arg(&mut output);
        launch.arg(&pixel_count_i32);
        unsafe { launch.launch(cfg) }.context("launching CUDA colorspace kernel")?;
        Ok(CudaFrameLab {
            data: output,
            width: frame_bgr.width,
            height: frame_bgr.height,
        })
    }

    fn download_frame_f32(&self, frame_lab: &Self::DeviceFrameLab) -> Result<Array3<f32>> {
        let inner = self.inner()?;
        let packed = inner
            .stream
            .clone_dtoh(&frame_lab.data)
            .context("downloading CUDA CIELAB frame")?;
        unpack_lab_pixels(&packed, frame_lab.height, frame_lab.width)
    }

    fn build_roi_mask(
        &self,
        shape: (usize, usize),
        margins: RoiMargins,
    ) -> Result<Self::DeviceMaskU8> {
        let inner = self.inner()?;
        let (height, width) = shape;
        let (top, bottom, left, right) = roi_bounds(shape, margins);
        let pixel_count = width * height;
        let mut output = inner
            .stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA ROI mask")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let width_i32 = width as i32;
        let height_i32 = height as i32;
        let top_i32 = top as i32;
        let bottom_i32 = bottom as i32;
        let left_i32 = left as i32;
        let right_i32 = right as i32;
        let mut launch = inner.stream.launch_builder(&inner.roi);
        launch.arg(&mut output);
        launch.arg(&width_i32);
        launch.arg(&height_i32);
        launch.arg(&top_i32);
        launch.arg(&bottom_i32);
        launch.arg(&left_i32);
        launch.arg(&right_i32);
        unsafe { launch.launch(cfg) }.context("launching CUDA ROI kernel")?;
        Ok(CudaMaskU8 {
            data: output,
            width,
            height,
        })
    }

    fn download_mask_u8(&self, mask: &Self::DeviceMaskU8) -> Result<Array2<u8>> {
        let inner = self.inner()?;
        let values = inner
            .stream
            .clone_dtoh(&mask.data)
            .context("downloading CUDA ROI mask")?;
        Array2::from_shape_vec((mask.height, mask.width), values)
            .context("reshaping downloaded ROI mask")
    }

    fn compute_delta_darken_only(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Array2<f32>,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32> {
        let inner = self.inner()?;
        anyhow::ensure!(
            reference_plane.dim() == (frame_lab.height, frame_lab.width),
            "reference plane shape does not match frame"
        );
        anyhow::ensure!(
            (roi_mask.height, roi_mask.width) == (frame_lab.height, frame_lab.width),
            "ROI mask shape does not match frame"
        );

        let reference = reference_plane
            .as_slice_memory_order()
            .ok_or_else(|| anyhow!("reference plane must be contiguous"))?;
        let reference_buffer = CudaPlaneF32 {
            data: inner
                .stream
                .clone_htod(reference)
                .context("uploading reference plane to CUDA")?,
            width: frame_lab.width,
            height: frame_lab.height,
        };
        self.compute_delta_darken_only_device(
            frame_lab,
            &reference_buffer,
            roi_mask,
            channel_weight,
        )
    }

    fn download_plane_f32(&self, plane: &Self::DevicePlaneF32) -> Result<Array2<f32>> {
        let inner = self.inner()?;
        let values = inner
            .stream
            .clone_dtoh(&plane.data)
            .context("downloading CUDA delta plane")?;
        Array2::from_shape_vec((plane.height, plane.width), values)
            .context("reshaping downloaded delta plane")
    }
}

impl PeakImageBackend for CudaBackend {
    fn extract_l_plane(&self, frame_lab: &Self::DeviceFrameLab) -> Result<Self::DevicePlaneF32> {
        let inner = self.inner()?;
        let pixel_count = frame_lab.width * frame_lab.height;
        let mut output = inner
            .stream
            .alloc_zeros::<f32>(pixel_count)
            .context("allocating CUDA L plane")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.stream.launch_builder(&inner.extract_l);
        launch.arg(&frame_lab.data);
        launch.arg(&mut output);
        launch.arg(&pixel_count_i32);
        unsafe { launch.launch(cfg) }.context("launching CUDA L-plane kernel")?;
        Ok(CudaPlaneF32 {
            data: output,
            width: frame_lab.width,
            height: frame_lab.height,
        })
    }

    fn update_peak_brightness_device(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        previous_peak: Option<&Self::DevicePlaneF32>,
    ) -> Result<Self::DevicePlaneF32> {
        let inner = self.inner()?;
        let pixel_count = frame_lab.width * frame_lab.height;
        let zero_buffer = if previous_peak.is_none() {
            Some(
                inner
                    .stream
                    .alloc_zeros::<f32>(pixel_count)
                    .context("allocating initial CUDA peak plane")?,
            )
        } else {
            None
        };
        if let Some(previous_peak) = previous_peak {
            anyhow::ensure!(
                (previous_peak.height, previous_peak.width) == (frame_lab.height, frame_lab.width),
                "previous peak shape does not match frame"
            );
        }
        let previous_view = previous_peak.map(|peak| peak.data.as_view());
        let mut output = inner
            .stream
            .alloc_zeros::<f32>(pixel_count)
            .context("allocating CUDA peak plane")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.stream.launch_builder(&inner.peak);
        launch.arg(&frame_lab.data);
        if let Some(previous_view) = previous_view.as_ref() {
            launch.arg(previous_view);
        } else if let Some(zero_buffer) = zero_buffer.as_ref() {
            launch.arg(zero_buffer);
        }
        launch.arg(&mut output);
        launch.arg(&pixel_count_i32);
        unsafe { launch.launch(cfg) }.context("launching CUDA peak kernel")?;
        Ok(CudaPlaneF32 {
            data: output,
            width: frame_lab.width,
            height: frame_lab.height,
        })
    }

    fn compute_delta_darken_only_device(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Self::DevicePlaneF32,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32> {
        let inner = self.inner()?;
        anyhow::ensure!(
            (reference_plane.height, reference_plane.width) == (frame_lab.height, frame_lab.width),
            "reference plane shape does not match frame"
        );
        anyhow::ensure!(
            (roi_mask.height, roi_mask.width) == (frame_lab.height, frame_lab.width),
            "ROI mask shape does not match frame"
        );

        let pixel_count = frame_lab.width * frame_lab.height;
        let mut output = inner
            .stream
            .alloc_zeros::<f32>(pixel_count)
            .context("allocating CUDA delta plane")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.stream.launch_builder(&inner.delta);
        launch.arg(&frame_lab.data);
        launch.arg(&reference_plane.data);
        launch.arg(&roi_mask.data);
        launch.arg(&mut output);
        launch.arg(&channel_weight);
        launch.arg(&pixel_count_i32);
        unsafe { launch.launch(cfg) }.context("launching CUDA delta kernel")?;
        Ok(CudaPlaneF32 {
            data: output,
            width: frame_lab.width,
            height: frame_lab.height,
        })
    }

    fn blur_and_normalize_delta(
        &self,
        delta: &Self::DevicePlaneF32,
        blur_kernel: usize,
        blur_enabled: bool,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let inner = self.inner()?;
        let pixel_count = delta.width * delta.height;
        let (source_width, source_height) = (delta.width, delta.height);
        let source = if blur_enabled {
            let kernel = canonical_blur_kernel_size(blur_kernel);
            let weights = gaussian_kernel_weights(kernel)?;
            let weights_buffer = inner
                .stream
                .clone_htod(&weights)
                .context("uploading CUDA gaussian weights")?;
            let mut output = inner
                .stream
                .alloc_zeros::<f32>(pixel_count)
                .context("allocating CUDA blurred delta plane")?;
            let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
            let width_i32 = source_width as i32;
            let height_i32 = source_height as i32;
            let kernel_i32 = kernel as i32;
            let pixel_count_i32 = pixel_count as i32;
            let mut launch = inner.stream.launch_builder(&inner.blur_norm);
            launch.arg(&delta.data);
            launch.arg(&weights_buffer);
            launch.arg(&mut output);
            launch.arg(&width_i32);
            launch.arg(&height_i32);
            launch.arg(&kernel_i32);
            launch.arg(&pixel_count_i32);
            unsafe { launch.launch(cfg) }.context("launching CUDA gaussian blur kernel")?;
            output
        } else {
            inner
                .stream
                .clone_dtod(&delta.data.as_view())
                .context("copying CUDA delta plane for normalization")?
        };

        let mut min_buffer = inner
            .stream
            .clone_htod(&[f32::INFINITY.to_bits()])
            .context("allocating CUDA min accumulator")?;
        let mut max_buffer = inner
            .stream
            .clone_htod(&[0u32])
            .context("allocating CUDA max accumulator")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut reduce = inner.stream.launch_builder(&inner.reduce_minmax);
        reduce.arg(&source);
        reduce.arg(&mut min_buffer);
        reduce.arg(&mut max_buffer);
        reduce.arg(&pixel_count_i32);
        unsafe { reduce.launch(cfg) }.context("launching CUDA delta min/max kernel")?;

        let min_bits = inner
            .stream
            .clone_dtoh(&min_buffer)
            .context("downloading CUDA delta min accumulator")?;
        let max_bits = inner
            .stream
            .clone_dtoh(&max_buffer)
            .context("downloading CUDA delta max accumulator")?;
        let min_value = f32::from_bits(min_bits[0]);
        let max_value = f32::from_bits(max_bits[0]);

        let mut output = inner
            .stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA normalized delta plane")?;
        let mut normalize = inner.stream.launch_builder(&inner.normalize_u8);
        normalize.arg(&source);
        normalize.arg(&mut output);
        normalize.arg(&min_value);
        normalize.arg(&max_value);
        normalize.arg(&pixel_count_i32);
        unsafe { normalize.launch(cfg) }.context("launching CUDA normalize kernel")?;

        Ok(Some(CudaMaskU8 {
            data: output,
            width: source_width,
            height: source_height,
        }))
    }

    fn threshold_and_morph_mask(
        &self,
        delta_norm: &Self::DeviceMaskU8,
        threshold_value: f32,
        morph_shape: MorphShape,
        morph_kernel: usize,
        morph_close_iterations: usize,
        morph_open_iterations: usize,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let inner = self.inner()?;
        let pixel_count = delta_norm.width * delta_norm.height;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;

        let mut current = inner
            .stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA threshold output")?;
        let mut threshold = inner.stream.launch_builder(&inner.threshold_binary);
        threshold.arg(&delta_norm.data);
        threshold.arg(&mut current);
        threshold.arg(&threshold_value);
        threshold.arg(&pixel_count_i32);
        unsafe { threshold.launch(cfg) }.context("launching CUDA threshold kernel")?;

        if morph_close_iterations == 0 && morph_open_iterations == 0 {
            return Ok(Some(CudaMaskU8 {
                data: current,
                width: delta_norm.width,
                height: delta_norm.height,
            }));
        }

        let kernel_size = canonical_morph_kernel_size(morph_kernel);
        let kernel_mask = structuring_element_mask(morph_shape, kernel_size)?;
        let kernel_buffer = inner
            .stream
            .clone_htod(&kernel_mask)
            .context("uploading CUDA morphology kernel mask")?;
        let width_i32 = delta_norm.width as i32;
        let height_i32 = delta_norm.height as i32;
        let kernel_i32 = kernel_size as i32;
        let mut scratch = inner
            .stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA morphology scratch buffer")?;

        for _ in 0..morph_close_iterations {
            launch_binary_morph(
                inner,
                &inner.dilate_binary,
                &current,
                &kernel_buffer,
                &mut scratch,
                width_i32,
                height_i32,
                kernel_i32,
                pixel_count_i32,
                cfg,
                "dilation",
            )?;
            std::mem::swap(&mut current, &mut scratch);
            launch_binary_morph(
                inner,
                &inner.erode_binary,
                &current,
                &kernel_buffer,
                &mut scratch,
                width_i32,
                height_i32,
                kernel_i32,
                pixel_count_i32,
                cfg,
                "erosion",
            )?;
            std::mem::swap(&mut current, &mut scratch);
        }
        for _ in 0..morph_open_iterations {
            launch_binary_morph(
                inner,
                &inner.erode_binary,
                &current,
                &kernel_buffer,
                &mut scratch,
                width_i32,
                height_i32,
                kernel_i32,
                pixel_count_i32,
                cfg,
                "erosion",
            )?;
            std::mem::swap(&mut current, &mut scratch);
            launch_binary_morph(
                inner,
                &inner.dilate_binary,
                &current,
                &kernel_buffer,
                &mut scratch,
                width_i32,
                height_i32,
                kernel_i32,
                pixel_count_i32,
                cfg,
                "dilation",
            )?;
            std::mem::swap(&mut current, &mut scratch);
        }

        Ok(Some(CudaMaskU8 {
            data: current,
            width: delta_norm.width,
            height: delta_norm.height,
        }))
    }
}

fn cuda_device_ordinal() -> Result<usize> {
    match env::var("FAST_VRIFA_CUDA_DEVICE") {
        Ok(raw) => raw
            .parse::<usize>()
            .with_context(|| format!("parsing FAST_VRIFA_CUDA_DEVICE={raw} as usize")),
        Err(env::VarError::NotPresent) => Ok(0),
        Err(err) => bail!("reading FAST_VRIFA_CUDA_DEVICE: {err}"),
    }
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn unpack_lab_pixels(packed: &[u32], height: usize, width: usize) -> Result<Array3<f32>> {
    let values = packed
        .iter()
        .flat_map(|value| {
            [
                (value & 0xff) as f32,
                ((value >> 8) & 0xff) as f32,
                ((value >> 16) & 0xff) as f32,
            ]
        })
        .collect::<Vec<_>>();
    Array3::from_shape_vec((height, width, 3), values).context("reshaping downloaded CIELAB frame")
}

fn roi_bounds(shape: (usize, usize), margins: RoiMargins) -> (usize, usize, usize, usize) {
    let (height, width) = shape;
    let top = (margins.top * height as f32) as usize;
    let mut bottom = height.saturating_sub((margins.bottom * height as f32) as usize);
    let left = (margins.left * width as f32) as usize;
    let mut right = width.saturating_sub((margins.right * width as f32) as usize);
    if bottom <= top {
        bottom = height.min(top + 1);
    }
    if right <= left {
        right = width.min(left + 1);
    }
    (top, bottom, left, right)
}

fn canonical_blur_kernel_size(blur_kernel: usize) -> usize {
    let mut kernel = blur_kernel;
    if kernel % 2 == 0 {
        kernel += 1;
    }
    kernel.max(1)
}

fn canonical_morph_kernel_size(morph_kernel: usize) -> usize {
    let kernel = morph_kernel + (1 - morph_kernel % 2);
    kernel.max(1)
}

fn gaussian_kernel_weights(kernel_size: usize) -> Result<Vec<f32>> {
    let kernel_size = canonical_blur_kernel_size(kernel_size);
    let kernel = imgproc::get_gaussian_kernel(kernel_size as i32, 0.0, CV_32F)
        .context("building Gaussian kernel for CUDA blur")?;
    Ok(kernel
        .data_typed::<f32>()
        .context("reading Gaussian kernel coefficients")?
        .iter()
        .copied()
        .collect())
}

fn structuring_element_mask(shape: MorphShape, kernel_size: usize) -> Result<Vec<u8>> {
    let kernel_size = canonical_morph_kernel_size(kernel_size);
    let kernel = imgproc::get_structuring_element_def(
        morph_shape_code(shape),
        opencv::core::Size::new(kernel_size as i32, kernel_size as i32),
    )
    .context("building morphology kernel for CUDA")?;
    Ok(kernel
        .data_typed::<u8>()
        .context("reading morphology kernel bytes")?
        .iter()
        .map(|value| u8::from(*value > 0))
        .collect())
}

fn morph_shape_code(shape: MorphShape) -> i32 {
    match shape {
        MorphShape::Ellipse => imgproc::MORPH_ELLIPSE,
        MorphShape::Rect => imgproc::MORPH_RECT,
        MorphShape::Cross => imgproc::MORPH_CROSS,
    }
}

fn launch_binary_morph(
    inner: &CudaBackendInner,
    kernel: &CudaFunction,
    input: &CudaSlice<u8>,
    kernel_mask: &CudaSlice<u8>,
    output: &mut CudaSlice<u8>,
    width: i32,
    height: i32,
    kernel_size: i32,
    pixel_count: i32,
    cfg: LaunchConfig,
    label: &str,
) -> Result<()> {
    let mut launch = inner.stream.launch_builder(kernel);
    launch.arg(input);
    launch.arg(kernel_mask);
    launch.arg(output);
    launch.arg(&width);
    launch.arg(&height);
    launch.arg(&kernel_size);
    launch.arg(&pixel_count);
    unsafe { launch.launch(cfg) }
        .with_context(|| format!("launching CUDA morphology {label} kernel"))?;
    Ok(())
}

fn lab_lut_cache_path() -> PathBuf {
    env::temp_dir().join("fast_vrifa_cuda_bgr2lab_u8_lut_v1.bin")
}

fn load_or_build_lab_lut() -> Result<Vec<u32>> {
    let expected_bytes = (1usize << 24) * std::mem::size_of::<u32>();
    let cache_path = lab_lut_cache_path();
    if let Ok(bytes) = fs::read(&cache_path) {
        if bytes.len() == expected_bytes {
            let values = cast_slice::<u8, u32>(&bytes).to_vec();
            return Ok(values);
        }
    }

    const CHUNK_SIZE: usize = 1 << 20;
    let mut lut = vec![0u32; 1 << 24];
    for start in (0..lut.len()).step_by(CHUNK_SIZE) {
        let len = (lut.len() - start).min(CHUNK_SIZE);
        let mut frame = Array3::<u8>::zeros((len, 1, 3));
        for offset in 0..len {
            let packed = (start + offset) as u32;
            frame[(offset, 0, 0)] = (packed & 0xff) as u8;
            frame[(offset, 0, 1)] = ((packed >> 8) & 0xff) as u8;
            frame[(offset, 0, 2)] = ((packed >> 16) & 0xff) as u8;
        }
        let converted = convert_frame_to_colorspace(&frame, ColorSpace::Cielab)
            .context("building the CUDA BGR->CIELAB lookup table")?;
        for offset in 0..len {
            lut[start + offset] = converted[(offset, 0, 0)] as u32
                | ((converted[(offset, 0, 1)] as u32) << 8)
                | ((converted[(offset, 0, 2)] as u32) << 16);
        }
    }

    let _ = fs::write(&cache_path, cast_slice(&lut));
    Ok(lut)
}

#[cfg(test)]
mod tests {
    use super::{roi_bounds, CudaBackend};
    use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend, PeakImageBackend, RoiMargins};
    use ndarray::array;

    #[test]
    fn cuda_backend_reports_runtime_availability() {
        let backend = CudaBackend::new();
        assert_eq!(backend.kind(), BackendKind::Cuda);
        assert!(matches!(
            backend.status(),
            BackendStatus::Ready | BackendStatus::Unavailable
        ));
        assert!(!backend.label().is_empty());
    }

    #[test]
    fn upload_requires_contiguous_bgr_layout() {
        let frame = array![[[1u8, 2u8, 3u8], [4u8, 5u8, 6u8]]];
        let backend = CudaBackend::new();
        if !matches!(backend.status(), BackendStatus::Ready) {
            return;
        }
        let uploaded = backend.upload_frame_bgr(&frame).unwrap();
        assert_eq!(uploaded.width, 2);
        assert_eq!(uploaded.height, 1);
    }

    #[test]
    fn roi_bounds_match_cpu_side_rounding() {
        let bounds = roi_bounds(
            (100, 200),
            RoiMargins {
                top: 0.1,
                bottom: 0.2,
                left: 0.05,
                right: 0.15,
            },
        );
        assert_eq!(bounds, (10, 80, 10, 170));
    }

    #[test]
    fn cuda_peak_path_matches_extracted_l_plane_when_available() {
        let backend = CudaBackend::new();
        if !matches!(backend.status(), BackendStatus::Ready) {
            return;
        }

        let frame = array![[[0u8, 0u8, 0u8], [255u8, 255u8, 255u8]]];
        let uploaded = backend.upload_frame_bgr(&frame).unwrap();
        let converted = backend.convert_bgr_to_lab(&uploaded).unwrap();
        let l_plane = backend.extract_l_plane(&converted).unwrap();
        let peak = backend
            .update_peak_brightness_device(&converted, Some(&l_plane))
            .unwrap();
        assert_eq!(
            backend.download_plane_f32(&peak).unwrap(),
            backend.download_plane_f32(&l_plane).unwrap()
        );
    }
}
