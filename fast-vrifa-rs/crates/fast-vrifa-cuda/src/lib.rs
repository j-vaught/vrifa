use anyhow::{anyhow, bail, Context, Result};
use bytemuck::cast_slice;
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, DevicePtr,
        LaunchConfig, PushKernelArg,
    },
    nvrtc::compile_ptx,
};
#[cfg(target_os = "linux")]
use cudarc::driver::DevicePtrMut;
use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend, PeakImageBackend, RoiMargins};
use ndarray::{Array2, Array3};
use opencv::core::CV_32F;
use opencv::imgproc;
use opencv::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use vrifa_core::colorspace::{convert_frame_to_colorspace, ColorSpace};
use vrifa_core::morphology::MorphShape;

mod npp;

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

#[derive(Clone)]
pub struct CudaLockState {
    counter: CudaSlice<u16>,
    locked: CudaSlice<u8>,
    width: usize,
    height: usize,
}

struct CudaLabBatch {
    data: CudaSlice<u32>,
    frames: usize,
}

pub struct CudaBatchDetectorOptions {
    pub channel_weight: f32,
    pub blur_kernel: usize,
    pub blur_enabled: bool,
    pub threshold_offset: f32,
    pub morph_shape: MorphShape,
    pub morph_kernel: usize,
    pub morph_close_iterations: usize,
    pub morph_open_iterations: usize,
    pub min_area: usize,
    pub lock_frames: usize,
    pub need_host_mask: bool,
    pub need_host_delta_norm: bool,
}

pub struct CudaBatchDetectorState {
    pub peak: Option<CudaPlaneF32>,
    pub lock: Option<CudaLockState>,
}

pub struct CudaBatchFrameOutput {
    pub mask: Option<Array2<u8>>,
    pub delta_norm: Option<Array2<u8>>,
}

struct CudaBackendInner {
    _context: Arc<CudaContext>,
    upload_stream: Arc<CudaStream>,
    compute_stream: Arc<CudaStream>,
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
    threshold_binary_device: CudaFunction,
    histogram_u8: CudaFunction,
    otsu_threshold: CudaFunction,
    dilate_binary: CudaFunction,
    erode_binary: CudaFunction,
    count_components: CudaFunction,
    filter_components: CudaFunction,
    lock_mask: CudaFunction,
    lab_lut: CudaSlice<u32>,
    npp: Option<Arc<npp::NppLibrary>>,
    gaussian_kernels: Mutex<HashMap<usize, CudaSlice<f32>>>,
    morph_kernels: Mutex<HashMap<(i32, usize), CudaSlice<u8>>>,
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

    pub fn create_batch_detector_state(
        &self,
        shape: (usize, usize),
        lock_frames: usize,
    ) -> Result<CudaBatchDetectorState> {
        let lock = if lock_frames > 0 {
            self.create_lock_state(shape)?
        } else {
            None
        };
        Ok(CudaBatchDetectorState { peak: None, lock })
    }

    pub fn process_peak_detector_batch(
        &self,
        frames: &[Array3<u8>],
        roi_mask: &CudaMaskU8,
        state: &mut CudaBatchDetectorState,
        options: &CudaBatchDetectorOptions,
    ) -> Result<Vec<CudaBatchFrameOutput>> {
        let inner = self.inner()?;
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let height = frames[0].dim().0;
        let width = frames[0].dim().1;
        anyhow::ensure!(
            (roi_mask.height, roi_mask.width) == (height, width),
            "ROI mask shape does not match batch frame shape"
        );
        for frame in frames.iter().skip(1) {
            anyhow::ensure!(
                frame.dim() == (height, width, 3),
                "all frames in a CUDA batch must have matching shape"
            );
        }

        let batch_lab = self.upload_and_convert_batch(frames)?;
        let frame_pixels = width * height;
        let mut outputs = Vec::with_capacity(batch_lab.frames);
        for batch_index in 0..batch_lab.frames {
            let offset = batch_index * frame_pixels;
            let frame_lab = batch_lab
                .data
                .try_slice(offset..offset + frame_pixels)
                .ok_or_else(|| anyhow!("creating CUDA batch LAB frame view"))?;
            state.peak = Some(update_peak_from_lab_view(
                inner,
                &frame_lab,
                state.peak.as_ref(),
                width,
                height,
            )?);
            let delta = compute_delta_from_lab_view(
                inner,
                &frame_lab,
                state
                    .peak
                    .as_ref()
                    .ok_or_else(|| anyhow!("CUDA peak state was not initialized"))?,
                roi_mask,
                options.channel_weight,
                width,
                height,
            )?;
            let delta_norm = self
                .blur_and_normalize_delta(&delta, options.blur_kernel, options.blur_enabled)?
                .ok_or_else(|| anyhow!("CUDA batch path requires device blur+normalize"))?;
            let mut mask = self
                .threshold_and_morph_mask_auto(
                    &delta_norm,
                    options.threshold_offset,
                    options.morph_shape,
                    options.morph_kernel,
                    options.morph_close_iterations,
                    options.morph_open_iterations,
                )?
                .ok_or_else(|| anyhow!("CUDA batch path requires device threshold+morph"))?;
            if options.min_area > 0 {
                mask = self
                    .filter_min_area_mask(&mask, options.min_area)?
                    .ok_or_else(|| anyhow!("CUDA batch path requires device min-area filtering"))?;
            }
            if options.lock_frames > 0 {
                let state_lock = state
                    .lock
                    .as_mut()
                    .ok_or_else(|| anyhow!("CUDA batch lock state was not initialized"))?;
                mask = self
                    .apply_locking_device(&mask, options.lock_frames, state_lock)?
                    .ok_or_else(|| anyhow!("CUDA batch path requires device locking"))?;
            }
            outputs.push(CudaBatchFrameOutput {
                mask: if options.need_host_mask {
                    Some(download_mask_ptr(inner, &mask.data, height, width, "downloading CUDA batch mask")?)
                } else {
                    None
                },
                delta_norm: if options.need_host_delta_norm {
                    Some(download_mask_ptr(
                        inner,
                        &delta_norm.data,
                        height,
                        width,
                        "downloading CUDA batch normalized delta",
                    )?)
                } else {
                    None
                },
            });
        }
        Ok(outputs)
    }

    fn upload_and_convert_batch(&self, frames: &[Array3<u8>]) -> Result<CudaLabBatch> {
        let inner = self.inner()?;
        let height = frames[0].dim().0;
        let width = frames[0].dim().1;
        let frame_bytes = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| anyhow!("CUDA batch frame byte size overflow"))?;
        let mut host_bgr = Vec::with_capacity(frame_bytes * frames.len());
        for frame in frames {
            let bytes = frame
                .as_slice_memory_order()
                .ok_or_else(|| anyhow!("CUDA batch input frame must be contiguous"))?;
            host_bgr.extend_from_slice(bytes);
        }

        let device_bgr = inner
            .upload_stream
            .clone_htod(&host_bgr)
            .context("uploading CUDA BGR batch")?;
        let pixel_count = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(frames.len()))
            .ok_or_else(|| anyhow!("CUDA batch pixel count overflow"))?;
        let mut output = inner
            .compute_stream
            .alloc_zeros::<u32>(pixel_count)
            .context("allocating CUDA CIELAB batch")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.compute_stream.launch_builder(&inner.colorspace);
        launch.arg(&device_bgr);
        launch.arg(&inner.lab_lut);
        launch.arg(&mut output);
        launch.arg(&pixel_count_i32);
        unsafe { launch.launch(cfg) }.context("launching CUDA batch colorspace kernel")?;
        Ok(CudaLabBatch {
            data: output,
            frames: frames.len(),
        })
    }
}

impl CudaBackendInner {
    fn new() -> Result<Self> {
        let ordinal = cuda_device_ordinal()?;
        let context = CudaContext::new(ordinal)
            .with_context(|| format!("initializing CUDA context on device {ordinal}"))?;
        let compute_stream = context.default_stream();
        let upload_stream = context
            .new_stream()
            .context("creating CUDA upload stream")?;
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
        let threshold_binary_device = module
            .load_function("threshold_binary_u8_device")
            .context("loading CUDA device-threshold kernel")?;
        let histogram_u8 = module
            .load_function("histogram_u8")
            .context("loading CUDA histogram kernel")?;
        let otsu_threshold = module
            .load_function("otsu_threshold_from_histogram")
            .context("loading CUDA Otsu kernel")?;
        let dilate_binary = module
            .load_function("dilate_binary_u8")
            .context("loading CUDA dilation kernel")?;
        let erode_binary = module
            .load_function("erode_binary_u8")
            .context("loading CUDA erosion kernel")?;
        let count_components = module
            .load_function("count_labeled_components_u32")
            .context("loading CUDA component-count kernel")?;
        let filter_components = module
            .load_function("filter_labeled_components_u8")
            .context("loading CUDA component-filter kernel")?;
        let lock_mask = module
            .load_function("apply_locking_u8")
            .context("loading CUDA lock-state kernel")?;
        let host_lut = load_or_build_lab_lut()?;
        let lab_lut = compute_stream
            .clone_htod(&host_lut)
            .context("uploading BGR->CIELAB lookup table to CUDA")?;
        let npp = npp::NppLibrary::load().ok();
        Ok(Self {
            _context: context,
            upload_stream,
            compute_stream,
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
            threshold_binary_device,
            histogram_u8,
            otsu_threshold,
            dilate_binary,
            erode_binary,
            count_components,
            filter_components,
            lock_mask,
            lab_lut,
            npp,
            gaussian_kernels: Mutex::new(HashMap::new()),
            morph_kernels: Mutex::new(HashMap::new()),
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
    type DeviceLockState = CudaLockState;

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
            .upload_stream
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
            .compute_stream
            .alloc_zeros::<u32>(pixel_count)
            .context("allocating CUDA CIELAB frame")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.compute_stream.launch_builder(&inner.colorspace);
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
            .compute_stream
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
            .compute_stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA ROI mask")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let width_i32 = width as i32;
        let height_i32 = height as i32;
        let top_i32 = top as i32;
        let bottom_i32 = bottom as i32;
        let left_i32 = left as i32;
        let right_i32 = right as i32;
        let mut launch = inner.compute_stream.launch_builder(&inner.roi);
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
            .compute_stream
            .clone_dtoh(&mask.data)
            .context("downloading CUDA ROI mask")?;
        Array2::from_shape_vec((mask.height, mask.width), values)
            .context("reshaping downloaded ROI mask")
    }

    fn upload_plane_f32(&self, plane: &Array2<f32>) -> Result<Self::DevicePlaneF32> {
        let inner = self.inner()?;
        let values = plane
            .as_slice_memory_order()
            .ok_or_else(|| anyhow!("reference plane must be contiguous"))?;
        Ok(CudaPlaneF32 {
            data: inner
                .compute_stream
                .clone_htod(values)
                .context("uploading reference plane to CUDA")?,
            width: plane.dim().1,
            height: plane.dim().0,
        })
    }

    fn compute_delta_darken_only(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Array2<f32>,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32> {
        anyhow::ensure!(
            reference_plane.dim() == (frame_lab.height, frame_lab.width),
            "reference plane shape does not match frame"
        );
        anyhow::ensure!(
            (roi_mask.height, roi_mask.width) == (frame_lab.height, frame_lab.width),
            "ROI mask shape does not match frame"
        );

        let reference_buffer = self.upload_plane_f32(reference_plane)?;
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
            .compute_stream
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
            .compute_stream
            .alloc_zeros::<f32>(pixel_count)
            .context("allocating CUDA L plane")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.compute_stream.launch_builder(&inner.extract_l);
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
                    .compute_stream
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
            .compute_stream
            .alloc_zeros::<f32>(pixel_count)
            .context("allocating CUDA peak plane")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.compute_stream.launch_builder(&inner.peak);
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
            .compute_stream
            .alloc_zeros::<f32>(pixel_count)
            .context("allocating CUDA delta plane")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut launch = inner.compute_stream.launch_builder(&inner.delta);
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
            let weights_buffer = cached_gaussian_kernel(inner, kernel)?;
            let mut output = inner
                .compute_stream
                .alloc_zeros::<f32>(pixel_count)
                .context("allocating CUDA blurred delta plane")?;
            let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
            let width_i32 = source_width as i32;
            let height_i32 = source_height as i32;
            let kernel_i32 = kernel as i32;
            let pixel_count_i32 = pixel_count as i32;
            let mut launch = inner.compute_stream.launch_builder(&inner.blur_norm);
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
                .compute_stream
                .clone_dtod(&delta.data.as_view())
                .context("copying CUDA delta plane for normalization")?
        };

        let mut min_buffer = inner
            .compute_stream
            .clone_htod(&[f32::INFINITY.to_bits()])
            .context("allocating CUDA min accumulator")?;
        let mut max_buffer = inner
            .compute_stream
            .clone_htod(&[0u32])
            .context("allocating CUDA max accumulator")?;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_i32 = pixel_count as i32;
        let mut reduce = inner.compute_stream.launch_builder(&inner.reduce_minmax);
        reduce.arg(&source);
        reduce.arg(&mut min_buffer);
        reduce.arg(&mut max_buffer);
        reduce.arg(&pixel_count_i32);
        unsafe { reduce.launch(cfg) }.context("launching CUDA delta min/max kernel")?;

        let min_bits = inner
            .compute_stream
            .clone_dtoh(&min_buffer)
            .context("downloading CUDA delta min accumulator")?;
        let max_bits = inner
            .compute_stream
            .clone_dtoh(&max_buffer)
            .context("downloading CUDA delta max accumulator")?;
        let min_value = f32::from_bits(min_bits[0]);
        let max_value = f32::from_bits(max_bits[0]);

        let mut output = inner
            .compute_stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA normalized delta plane")?;
        let mut normalize = inner.compute_stream.launch_builder(&inner.normalize_u8);
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
            .compute_stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA threshold output")?;
        let mut threshold = inner.compute_stream.launch_builder(&inner.threshold_binary);
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
        let kernel_buffer = cached_morph_kernel(inner, morph_shape, kernel_size)?;
        let width_i32 = delta_norm.width as i32;
        let height_i32 = delta_norm.height as i32;
        let kernel_i32 = kernel_size as i32;
        let mut scratch = inner
            .compute_stream
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

    fn threshold_and_morph_mask_auto(
        &self,
        delta_norm: &Self::DeviceMaskU8,
        threshold_offset: f32,
        morph_shape: MorphShape,
        morph_kernel: usize,
        morph_close_iterations: usize,
        morph_open_iterations: usize,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let inner = self.inner()?;
        let pixel_count = delta_norm.width * delta_norm.height;
        let histogram_cfg = LaunchConfig {
            grid_dim: ((pixel_count as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let pixel_count_i32 = pixel_count as i32;
        let mut histogram = inner
            .compute_stream
            .alloc_zeros::<u32>(256)
            .context("allocating CUDA Otsu histogram")?;
        let mut histogram_launch = inner.compute_stream.launch_builder(&inner.histogram_u8);
        histogram_launch.arg(&delta_norm.data);
        histogram_launch.arg(&mut histogram);
        histogram_launch.arg(&pixel_count_i32);
        unsafe { histogram_launch.launch(histogram_cfg) }
            .context("launching CUDA histogram kernel")?;

        let mut threshold = inner
            .compute_stream
            .alloc_zeros::<f32>(1)
            .context("allocating CUDA Otsu threshold")?;
        let otsu_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut otsu_launch = inner.compute_stream.launch_builder(&inner.otsu_threshold);
        otsu_launch.arg(&histogram);
        otsu_launch.arg(&mut threshold);
        otsu_launch.arg(&threshold_offset);
        unsafe { otsu_launch.launch(otsu_cfg) }.context("launching CUDA Otsu kernel")?;

        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let mut current = inner
            .compute_stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA threshold output")?;
        let mut threshold_launch = inner
            .compute_stream
            .launch_builder(&inner.threshold_binary_device);
        threshold_launch.arg(&delta_norm.data);
        threshold_launch.arg(&mut current);
        threshold_launch.arg(&threshold);
        threshold_launch.arg(&pixel_count_i32);
        unsafe { threshold_launch.launch(cfg) }
            .context("launching CUDA device-threshold kernel")?;

        if morph_close_iterations == 0 && morph_open_iterations == 0 {
            return Ok(Some(CudaMaskU8 {
                data: current,
                width: delta_norm.width,
                height: delta_norm.height,
            }));
        }

        let kernel_size = canonical_morph_kernel_size(morph_kernel);
        let kernel_buffer = cached_morph_kernel(inner, morph_shape, kernel_size)?;
        let width_i32 = delta_norm.width as i32;
        let height_i32 = delta_norm.height as i32;
        let kernel_i32 = kernel_size as i32;
        let mut scratch = inner
            .compute_stream
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

    fn filter_min_area_mask(
        &self,
        mask: &Self::DeviceMaskU8,
        min_area: usize,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (mask, min_area);
            return Ok(None);
        }

        #[cfg(target_os = "linux")]
        {
            if min_area == 0 {
                return Ok(Some(mask.clone()));
            }
            let inner = self.inner()?;
            let Some(npp) = inner.npp.as_ref() else {
                return Ok(None);
            };

            npp.set_stream(inner.compute_stream.cu_stream())
                .context("binding NPP to the CUDA stream")?;
            let stream_ctx = npp.stream_context().context("reading NPP stream context")?;
            let roi = npp::NppiSize {
                width: mask.width as i32,
                height: mask.height as i32,
            };
            let pixel_count = mask.width * mask.height;
            let labels_len = pixel_count;
            let mut labels = inner
                .compute_stream
                .alloc_zeros::<u32>(labels_len)
                .context("allocating CUDA label buffer")?;

            let mut label_buffer_size = 0i32;
            npp::status(
                unsafe { (npp.label_buffer_size)(roi, &mut label_buffer_size as *mut _) },
                "nppiLabelMarkersUFGetBufferSize_32u_C1R",
            )?;
            let mut label_buffer = inner
                .compute_stream
                .alloc_zeros::<u8>(label_buffer_size.max(0) as usize)
                .context("allocating CUDA label scratch buffer")?;

            {
                let (src_ptr, _src_record) = mask.data.device_ptr(&inner.compute_stream);
                let (labels_ptr, _labels_record) = labels.device_ptr_mut(&inner.compute_stream);
                let (label_buffer_ptr, _label_buffer_record) =
                    label_buffer.device_ptr_mut(&inner.compute_stream);
                npp::status(
                    unsafe {
                        (npp.label_markers)(
                            src_ptr as *mut u8,
                            mask.width as i32,
                            labels_ptr as *mut u32,
                            (mask.width * std::mem::size_of::<u32>()) as i32,
                            roi,
                            npp::NPPI_NORM_INF,
                            label_buffer_ptr as *mut u8,
                            stream_ctx,
                        )
                    },
                    "nppiLabelMarkersUF_8u32u_C1R_Ctx",
                )?;
            }

            let mut compress_buffer_size = 0i32;
            let starting_number = (mask.width * mask.height) as i32;
            npp::status(
                unsafe {
                    (npp.compress_buffer_size)(starting_number, &mut compress_buffer_size as *mut _)
                },
                "nppiCompressMarkerLabelsGetBufferSize_32u_C1R",
            )?;
            let mut compress_buffer = inner
                .compute_stream
                .alloc_zeros::<u8>(compress_buffer_size.max(0) as usize)
                .context("allocating CUDA label-compress scratch buffer")?;
            let mut new_number = 0i32;
            {
                let (labels_ptr, _labels_record) = labels.device_ptr_mut(&inner.compute_stream);
                let (compress_buffer_ptr, _compress_record) =
                    compress_buffer.device_ptr_mut(&inner.compute_stream);
                npp::status(
                    unsafe {
                        (npp.compress_markers)(
                            labels_ptr as *mut u32,
                            (mask.width * std::mem::size_of::<u32>()) as i32,
                            roi,
                            starting_number,
                            &mut new_number as *mut _,
                            compress_buffer_ptr as *mut u8,
                            stream_ctx,
                        )
                    },
                    "nppiCompressMarkerLabelsUF_32u_C1IR_Ctx",
                )?;
            }
            if new_number <= 0 {
                return Ok(Some(CudaMaskU8 {
                    data: inner
                        .compute_stream
                        .alloc_zeros::<u8>(pixel_count)
                        .context("allocating empty filtered mask")?,
                    width: mask.width,
                    height: mask.height,
                }));
            }

            let mut label_counts = inner
                .compute_stream
                .alloc_zeros::<u32>(new_number as usize + 1)
                .context("allocating CUDA component-count buffer")?;

            let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
            let pixel_count_i32 = pixel_count as i32;
            let mut count_launch = inner.compute_stream.launch_builder(&inner.count_components);
            count_launch.arg(&labels);
            count_launch.arg(&mut label_counts);
            count_launch.arg(&pixel_count_i32);
            unsafe { count_launch.launch(cfg) }.context("launching CUDA component-count kernel")?;

            let mut filtered = inner
                .compute_stream
                .alloc_zeros::<u8>(pixel_count)
                .context("allocating CUDA filtered mask")?;
            let min_area_u32 = min_area as u32;
            let mut launch = inner
                .compute_stream
                .launch_builder(&inner.filter_components);
            launch.arg(&mask.data);
            launch.arg(&labels);
            launch.arg(&label_counts);
            launch.arg(&mut filtered);
            launch.arg(&min_area_u32);
            launch.arg(&pixel_count_i32);
            unsafe { launch.launch(cfg) }.context("launching CUDA min-area filter kernel")?;

            Ok(Some(CudaMaskU8 {
                data: filtered,
                width: mask.width,
                height: mask.height,
            }))
        }
    }

    fn create_lock_state(&self, shape: (usize, usize)) -> Result<Option<Self::DeviceLockState>> {
        let inner = self.inner()?;
        let pixel_count = shape.0 * shape.1;
        Ok(Some(CudaLockState {
            counter: inner
                .compute_stream
                .alloc_zeros::<u16>(pixel_count)
                .context("allocating CUDA lock counter")?,
            locked: inner
                .compute_stream
                .alloc_zeros::<u8>(pixel_count)
                .context("allocating CUDA locked mask")?,
            width: shape.1,
            height: shape.0,
        }))
    }

    fn apply_locking_device(
        &self,
        mask: &Self::DeviceMaskU8,
        lock_frames: usize,
        state: &mut Self::DeviceLockState,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let inner = self.inner()?;
        anyhow::ensure!(
            (state.height, state.width) == (mask.height, mask.width),
            "device lock state shape does not match mask"
        );
        if lock_frames == 0 {
            return Ok(Some(mask.clone()));
        }
        let pixel_count = mask.width * mask.height;
        let pixel_count_i32 = pixel_count as i32;
        let lock_frames_u16 = lock_frames.min(u16::MAX as usize) as u16;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let mut output = inner
            .compute_stream
            .alloc_zeros::<u8>(pixel_count)
            .context("allocating CUDA lock output mask")?;
        let mut launch = inner.compute_stream.launch_builder(&inner.lock_mask);
        launch.arg(&mask.data);
        launch.arg(&mut state.counter);
        launch.arg(&mut state.locked);
        launch.arg(&mut output);
        launch.arg(&lock_frames_u16);
        launch.arg(&pixel_count_i32);
        unsafe { launch.launch(cfg) }.context("launching CUDA lock kernel")?;
        Ok(Some(CudaMaskU8 {
            data: output,
            width: mask.width,
            height: mask.height,
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

fn update_peak_from_lab_view(
    inner: &CudaBackendInner,
    frame_lab: &CudaView<'_, u32>,
    previous_peak: Option<&CudaPlaneF32>,
    width: usize,
    height: usize,
) -> Result<CudaPlaneF32> {
    let pixel_count = width * height;
    let zero_buffer = if previous_peak.is_none() {
        Some(
            inner
                .compute_stream
                .alloc_zeros::<f32>(pixel_count)
                .context("allocating initial CUDA peak plane")?,
        )
    } else {
        None
    };
    if let Some(previous_peak) = previous_peak {
        anyhow::ensure!(
            (previous_peak.height, previous_peak.width) == (height, width),
            "previous peak shape does not match frame"
        );
    }
    let previous_view = previous_peak.map(|peak| peak.data.as_view());
    let mut output = inner
        .compute_stream
        .alloc_zeros::<f32>(pixel_count)
        .context("allocating CUDA peak plane")?;
    let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
    let pixel_count_i32 = pixel_count as i32;
    let mut launch = inner.compute_stream.launch_builder(&inner.peak);
    launch.arg(frame_lab);
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
        width,
        height,
    })
}

fn compute_delta_from_lab_view(
    inner: &CudaBackendInner,
    frame_lab: &CudaView<'_, u32>,
    reference_plane: &CudaPlaneF32,
    roi_mask: &CudaMaskU8,
    channel_weight: f32,
    width: usize,
    height: usize,
) -> Result<CudaPlaneF32> {
    anyhow::ensure!(
        (reference_plane.height, reference_plane.width) == (height, width),
        "reference plane shape does not match frame"
    );
    anyhow::ensure!(
        (roi_mask.height, roi_mask.width) == (height, width),
        "ROI mask shape does not match frame"
    );
    let pixel_count = width * height;
    let mut output = inner
        .compute_stream
        .alloc_zeros::<f32>(pixel_count)
        .context("allocating CUDA delta plane")?;
    let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
    let pixel_count_i32 = pixel_count as i32;
    let mut launch = inner.compute_stream.launch_builder(&inner.delta);
    launch.arg(frame_lab);
    launch.arg(&reference_plane.data);
    launch.arg(&roi_mask.data);
    launch.arg(&mut output);
    launch.arg(&channel_weight);
    launch.arg(&pixel_count_i32);
    unsafe { launch.launch(cfg) }.context("launching CUDA delta kernel")?;
    Ok(CudaPlaneF32 {
        data: output,
        width,
        height,
    })
}

fn download_mask_ptr<Src>(inner: &CudaBackendInner, src: &Src, height: usize, width: usize, context: &str) -> Result<Array2<u8>>
where
    Src: DevicePtr<u8>,
{
    let values = inner
        .compute_stream
        .clone_dtoh(src)
        .with_context(|| context.to_string())?;
    Array2::from_shape_vec((height, width), values).context("reshaping downloaded CUDA mask")
}

fn cached_gaussian_kernel(inner: &CudaBackendInner, kernel_size: usize) -> Result<CudaSlice<f32>> {
    let kernel_size = canonical_blur_kernel_size(kernel_size);
    if let Some(buffer) = inner
        .gaussian_kernels
        .lock()
        .map_err(|_| anyhow!("locking Gaussian kernel cache"))?
        .get(&kernel_size)
        .cloned()
    {
        return Ok(buffer);
    }
    let weights = gaussian_kernel_weights(kernel_size)?;
    let buffer = inner
        .compute_stream
        .clone_htod(&weights)
        .context("uploading CUDA gaussian weights")?;
    inner
        .gaussian_kernels
        .lock()
        .map_err(|_| anyhow!("locking Gaussian kernel cache"))?
        .insert(kernel_size, buffer.clone());
    Ok(buffer)
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

fn cached_morph_kernel(
    inner: &CudaBackendInner,
    shape: MorphShape,
    kernel_size: usize,
) -> Result<CudaSlice<u8>> {
    let kernel_size = canonical_morph_kernel_size(kernel_size);
    let key = (morph_shape_code(shape), kernel_size);
    if let Some(buffer) = inner
        .morph_kernels
        .lock()
        .map_err(|_| anyhow!("locking morphology kernel cache"))?
        .get(&key)
        .cloned()
    {
        return Ok(buffer);
    }
    let kernel_mask = structuring_element_mask(shape, kernel_size)?;
    let buffer = inner
        .compute_stream
        .clone_htod(&kernel_mask)
        .context("uploading CUDA morphology kernel mask")?;
    inner
        .morph_kernels
        .lock()
        .map_err(|_| anyhow!("locking morphology kernel cache"))?
        .insert(key, buffer.clone());
    Ok(buffer)
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
    let mut launch = inner.compute_stream.launch_builder(kernel);
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
