use anyhow::{anyhow, bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use clap::Parser;
use fast_vrifa_core::{CpuBackend, ImageBackend, PeakImageBackend};
use indexmap::IndexMap;
use ndarray::{s, Array2, Array3};
use opencv::core::{self, Point, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;
#[cfg(feature = "cuda")]
use opencv::videoio;
use serde_yaml::Value;
use std::collections::VecDeque;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::mpsc::{channel, sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use vrifa_annotations::AnnotationFrame;
use vrifa_core::colorspace::{convert_frame_to_colorspace, ColorSpace};
use vrifa_core::blur::{self, BlurKind, BlurSpec};
use vrifa_core::contours::extract_bounding_boxes;
use vrifa_core::cvutil;
use vrifa_core::delta::compute_delta;
use vrifa_core::heatmap::apply_turbo_colormap;
use vrifa_core::lock::{apply_locking, LockState};
use vrifa_core::morphology::{MorphShape, MorphologyParams};
use vrifa_core::motion::estimate_translation;
use vrifa_core::overlay::create_overlay;
use vrifa_core::peak::update_peak_brightness_plane;
use vrifa_core::reference::{
    compute_dynamic_factor, select_dynamic_reference_index, DynamicReferenceParams,
};
use vrifa_core::registration::{fit_affine_warp, strong_gradient_mask, MotionModel};
use vrifa_core::roi::{build_roi_mask, resolve_roi_margins};
use vrifa_core::threshold::{self, ThresholdMode};
use vrifa_core::warp::{apply_warp, AffineWarp};
#[cfg(feature = "cuda")]
use vrifa_io::VideoMetadata;
use vrifa_io::{write_bgr_png, write_gray_png, VideoReader};

mod raw_video;
use raw_video::{
    finalize_raw_stream_to_mp4, AsyncRawVideoWriter, RawGrayFrameReader, RawPixelFormat,
    RawVideoArtifact,
};

#[cfg(feature = "cuda")]
use fast_vrifa_cuda::{CudaBackend, CudaBatchDetectorOptions};
#[cfg(feature = "wgpu")]
use fast_vrifa_wgpu::WgpuBackend;

pub use vrifa_cli::Config;
use vrifa_cli::{PeakOnShift, ReferenceMode};

#[cfg(feature = "cuda")]
const CUDA_BATCH_SIZE: usize = 32;
const OUTPUT_WORKERS: usize = 16;
const OUTPUT_QUEUE: usize = 96;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendMode {
    Delegate,
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(Clone, Debug)]
struct FastCliOptions {
    backend: BackendMode,
    ffmpeg_postprocess: bool,
    mask_only: bool,
    coco_bbox_only: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RunOptions {
    pub ffmpeg_postprocess: bool,
    pub mask_only: bool,
    pub coco_bbox_only: bool,
}

impl BackendMode {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "delegate" => Ok(Self::Delegate),
            "cpu" => Ok(Self::Cpu),
            "wgpu" => Ok(Self::Wgpu),
            "cuda" => Ok(Self::Cuda),
            other => bail!("--backend unsupported option '{other}'"),
        }
    }
}

impl FastCliOptions {
    fn effective_coco_bbox_only(&self) -> bool {
        self.coco_bbox_only || self.mask_only
    }

    fn effective_write_overlay_pngs(&self, config: &Config) -> bool {
        !self.mask_only && config.write_overlay_pngs
    }

    fn effective_write_heatmap_pngs(&self, config: &Config) -> bool {
        !self.mask_only && config.write_heatmap_pngs
    }

    fn effective_write_overlay_video(&self, config: &Config) -> bool {
        !self.mask_only && config.write_overlay_video
    }

    fn effective_write_heatmap_video(&self, config: &Config) -> bool {
        !self.mask_only && config.write_heatmap_video
    }

    fn has_fast_only_flags(&self) -> bool {
        self.mask_only || self.coco_bbox_only || self.ffmpeg_postprocess
    }
}

#[derive(Clone, Debug)]
struct MotionTraceRow {
    frame_index: usize,
    dx: f32,
    dy: f32,
    confidence: f32,
    cumulative_dx: f32,
    cumulative_dy: f32,
    per_frame_magnitude: f32,
    cumulative_magnitude: f32,
    recent_window_magnitude: f32,
    warp_dx: f32,
    warp_dy: f32,
    warp_error: f32,
    fit_applied: bool,
    fit_model: Option<MotionModel>,
    fit_score: Option<f32>,
    warp_active: bool,
}

const CAMERA_STABLE_RECENT_WINDOW: usize = 5;

#[derive(Clone, Debug)]
struct CameraStableState {
    prev_motion_frame: Option<Array3<f32>>,
    prev_registration_mask: Option<Array2<u8>>,
    reference_token: Option<usize>,
    cumulative_dx: f32,
    cumulative_dy: f32,
    recent_motion: VecDeque<(f32, f32)>,
    cached_warp: AffineWarp,
    warp_active: bool,
    shift_event_active: bool,
    shift_event_stable_frames: usize,
    motion_trace: Vec<MotionTraceRow>,
}

impl CameraStableState {
    fn new() -> Self {
        Self {
            prev_motion_frame: None,
            prev_registration_mask: None,
            reference_token: None,
            cumulative_dx: 0.0,
            cumulative_dy: 0.0,
            recent_motion: VecDeque::new(),
            cached_warp: AffineWarp::identity(),
            warp_active: false,
            shift_event_active: false,
            shift_event_stable_frames: 0,
            motion_trace: Vec::new(),
        }
    }

    fn reset(&mut self, reference_token: usize) {
        self.prev_motion_frame = None;
        self.prev_registration_mask = None;
        self.reference_token = Some(reference_token);
        self.cumulative_dx = 0.0;
        self.cumulative_dy = 0.0;
        self.recent_motion.clear();
        self.cached_warp = AffineWarp::identity();
        self.warp_active = false;
        self.shift_event_active = false;
        self.shift_event_stable_frames = 0;
    }
}

#[derive(Clone, Debug)]
enum PeakShiftAction {
    Keep,
    Reset,
    Warp(AffineWarp),
}

#[derive(Clone, Debug)]
struct CameraStableActions {
    warp: Option<AffineWarp>,
    peak_shift: PeakShiftAction,
}

fn build_registration_mask(prev_mask: Option<&Array2<u8>>) -> Option<Array2<u8>> {
    const MIN_DRY_FRACTION: f32 = 0.05;

    let prev_mask = prev_mask?;
    let (height, width) = prev_mask.dim();
    let total_pixels = height.saturating_mul(width);
    if total_pixels == 0 {
        return None;
    }

    let mut dry_pixels = 0usize;
    let dry_mask = prev_mask.mapv(|value| {
        if value == 0 {
            dry_pixels += 1;
            255
        } else {
            0
        }
    });
    let dry_fraction = dry_pixels as f32 / total_pixels as f32;
    (dry_fraction >= MIN_DRY_FRACTION).then_some(dry_mask)
}

fn build_static_registration_mask(
    prev_mask: Option<&Array2<u8>>,
    reference_for_frame: &Array3<f32>,
    roi_mask: &Array2<u8>,
) -> Result<Option<Array2<u8>>> {
    const STRONG_EDGE_PERCENTILE: f32 = 0.90;
    const BORDER_FRACTION: f32 = 0.08;
    const MIN_MASK_FRACTION: f32 = 0.02;

    let (height, width, _) = reference_for_frame.dim();
    let total_pixels = height.saturating_mul(width);
    if total_pixels == 0 {
        return Ok(None);
    }

    let dry_mask = build_registration_mask(prev_mask);
    let edge_mask = strong_gradient_mask(reference_for_frame, STRONG_EDGE_PERCENTILE)?;
    let border_mask = build_frame_border_mask((height, width), BORDER_FRACTION);

    let mut combined = Array2::<u8>::zeros((height, width));
    let mut combined_pixels = 0usize;
    for y in 0..height {
        for x in 0..width {
            let is_dry = dry_mask
                .as_ref()
                .map(|mask| mask[(y, x)] > 0)
                .unwrap_or(true);
            let outside_roi = roi_mask[(y, x)] == 0;
            let is_static = border_mask[(y, x)] > 0 || outside_roi || edge_mask[(y, x)] > 0;
            if is_dry && is_static {
                combined[(y, x)] = 255;
                combined_pixels += 1;
            }
        }
    }

    if combined_pixels as f32 / total_pixels as f32 >= MIN_MASK_FRACTION {
        return Ok(Some(combined));
    }
    Ok(dry_mask)
}

fn build_frame_border_mask(shape: (usize, usize), border_fraction: f32) -> Array2<u8> {
    let (height, width) = shape;
    let border_y = ((height as f32 * border_fraction).round() as usize).clamp(1, height.max(1));
    let border_x = ((width as f32 * border_fraction).round() as usize).clamp(1, width.max(1));
    let mut mask = Array2::<u8>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            if y < border_y
                || y >= height.saturating_sub(border_y)
                || x < border_x
                || x >= width.saturating_sub(border_x)
            {
                mask[(y, x)] = 255;
            }
        }
    }
    mask
}

fn peak_frame_plane(
    frame_converted: &Array3<f32>,
    pre_delta_blur: BlurSpec,
) -> Result<Array2<f32>> {
    let frame_l = frame_converted.slice(s![.., .., 0]).to_owned();
    if pre_delta_blur.is_no_op() {
        Ok(frame_l)
    } else {
        Ok(blur::blur_plane(&frame_l, pre_delta_blur)?)
    }
}

fn reference_token_for_frame(
    ref_mode: &ReferenceMode,
    frame_index: usize,
    reference_frame_index: usize,
) -> usize {
    match ref_mode {
        ReferenceMode::Running => frame_index,
        _ => reference_frame_index.max(1),
    }
}

fn prepare_camera_stable_actions(
    frame_index: usize,
    frame_converted: &Array3<f32>,
    reference_for_frame: &Array3<f32>,
    roi_mask: &Array2<u8>,
    reference_token: usize,
    config: &Config,
    state: &mut CameraStableState,
) -> Result<CameraStableActions> {
    if state.reference_token != Some(reference_token) {
        state.reset(reference_token);
    }

    let mut peak_shift = PeakShiftAction::Keep;

    if config.camera_stable {
        let (height, width, _) = frame_converted.dim();
        if let Some(prev_frame) = state.prev_motion_frame.as_ref() {
            let motion = estimate_translation(frame_converted, prev_frame)?;
            state.cumulative_dx -= motion.dx;
            state.cumulative_dy -= motion.dy;
            state.recent_motion.push_back((-motion.dx, -motion.dy));
            while state.recent_motion.len() > CAMERA_STABLE_RECENT_WINDOW {
                state.recent_motion.pop_front();
            }

            let per_frame_magnitude = motion.dx.hypot(motion.dy);
            let cumulative_magnitude = state.cumulative_dx.hypot(state.cumulative_dy);
            let (recent_window_dx, recent_window_dy) = state
                .recent_motion
                .iter()
                .fold((0.0f32, 0.0f32), |(sum_dx, sum_dy), (dx, dy)| {
                    (sum_dx + *dx, sum_dy + *dy)
                });
            let recent_window_magnitude = recent_window_dx.hypot(recent_window_dy);
            let (cached_warp_dx, cached_warp_dy) = if state.warp_active {
                state.cached_warp.center_displacement(width, height)
            } else {
                (0.0, 0.0)
            };
            let cached_translation_error = if state.warp_active {
                (state.cumulative_dx - cached_warp_dx).hypot(state.cumulative_dy - cached_warp_dy)
            } else {
                f32::INFINITY
            };
            let per_frame_trigger = per_frame_magnitude > config.motion_per_frame_threshold;
            if per_frame_trigger {
                state.shift_event_active = true;
                state.shift_event_stable_frames = 0;
            }
            if recent_window_magnitude > config.cumulative_motion_threshold {
                state.shift_event_active = true;
                state.shift_event_stable_frames = 0;
            }
            let drift_trigger =
                state.warp_active && cached_translation_error > config.cumulative_motion_threshold;
            let needs_fit = state.shift_event_active || drift_trigger;

            let mut fit_applied = false;
            let mut fit_model = None;
            let mut fit_score = None;
            if needs_fit {
                let registration_mask = build_static_registration_mask(
                    state.prev_registration_mask.as_ref(),
                    reference_for_frame,
                    roi_mask,
                )?;
                let init_warp = if state.warp_active {
                    let residual_dx = state.cumulative_dx - cached_warp_dx;
                    let residual_dy = state.cumulative_dy - cached_warp_dy;
                    AffineWarp::from_translation(residual_dx, residual_dy)
                        .compose(&state.cached_warp)
                } else {
                    AffineWarp::from_translation(state.cumulative_dx, state.cumulative_dy)
                };
                match fit_affine_warp(
                    frame_converted,
                    reference_for_frame,
                    &init_warp,
                    config.motion_model,
                    registration_mask.as_ref(),
                ) {
                    Ok(fit) => {
                        state.cached_warp = fit.warp;
                        let (fitted_dx, fitted_dy) =
                            state.cached_warp.center_displacement(width, height);
                        state.cumulative_dx = fitted_dx;
                        state.cumulative_dy = fitted_dy;
                        state.warp_active = !state.cached_warp.is_identityish(0.25);
                        fit_applied = true;
                        fit_model = Some(fit.model);
                        fit_score = Some(fit.score);
                        peak_shift = match config.peak_on_shift {
                            PeakOnShift::Reset => PeakShiftAction::Reset,
                            PeakOnShift::Warp => PeakShiftAction::Warp(state.cached_warp.clone()),
                        };
                    }
                    Err(err) => {
                        eprintln!(
                            "warning: camera-stable registration failed on frame {frame_index}: {err}"
                        );
                    }
                }
            }

            let (warp_dx, warp_dy, warp_error) = if state.warp_active {
                let (warp_dx, warp_dy) = state.cached_warp.center_displacement(width, height);
                let warp_error =
                    (state.cumulative_dx - warp_dx).hypot(state.cumulative_dy - warp_dy);
                (warp_dx, warp_dy, warp_error)
            } else {
                (0.0, 0.0, 0.0)
            };

            if state.shift_event_active {
                if per_frame_magnitude <= config.motion_per_frame_threshold {
                    state.shift_event_stable_frames += 1;
                    if state.shift_event_stable_frames >= 3 {
                        state.shift_event_active = false;
                        state.shift_event_stable_frames = 0;
                    }
                } else {
                    state.shift_event_stable_frames = 0;
                }
            }

            state.motion_trace.push(MotionTraceRow {
                frame_index,
                dx: motion.dx,
                dy: motion.dy,
                confidence: motion.confidence,
                cumulative_dx: state.cumulative_dx,
                cumulative_dy: state.cumulative_dy,
                per_frame_magnitude,
                cumulative_magnitude,
                recent_window_magnitude,
                warp_dx,
                warp_dy,
                warp_error,
                fit_applied,
                fit_model,
                fit_score,
                warp_active: state.warp_active,
            });
        } else {
            state.motion_trace.push(MotionTraceRow {
                frame_index,
                dx: 0.0,
                dy: 0.0,
                confidence: 0.0,
                cumulative_dx: 0.0,
                cumulative_dy: 0.0,
                per_frame_magnitude: 0.0,
                cumulative_magnitude: 0.0,
                recent_window_magnitude: 0.0,
                warp_dx: 0.0,
                warp_dy: 0.0,
                warp_error: 0.0,
                fit_applied: false,
                fit_model: None,
                fit_score: None,
                warp_active: false,
            });
        }
    }

    state.prev_motion_frame = Some(frame_converted.clone());
    Ok(CameraStableActions {
        warp: (config.camera_stable && state.warp_active).then(|| state.cached_warp.clone()),
        peak_shift,
    })
}

fn apply_peak_shift_host(
    previous_peak: Option<&Array2<f32>>,
    action: &PeakShiftAction,
) -> Result<Option<Array2<f32>>> {
    match action {
        PeakShiftAction::Keep => Ok(previous_peak.cloned()),
        PeakShiftAction::Reset => Ok(None),
        PeakShiftAction::Warp(matrix) => previous_peak
            .map(|peak| vrifa_core::warp::apply_warp_plane(peak, matrix))
            .transpose()
            .map_err(Into::into),
    }
}

fn apply_peak_shift_device<B: PeakImageBackend>(
    backend: &B,
    previous_peak: Option<&B::DevicePlaneF32>,
    action: &PeakShiftAction,
) -> Result<Option<B::DevicePlaneF32>> {
    match action {
        PeakShiftAction::Keep => previous_peak
            .map(|peak| backend.blur_plane_f32_device(peak, 1))
            .transpose(),
        PeakShiftAction::Reset => Ok(None),
        PeakShiftAction::Warp(matrix) => previous_peak
            .map(|peak| backend.warp_plane_f32_device(peak, matrix))
            .transpose(),
    }
}

fn write_motion_trace(path: &Path, rows: &[MotionTraceRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    writeln!(
        file,
        "frame,dx,dy,confidence,cumulative_dx,cumulative_dy,per_frame_magnitude,cumulative_magnitude,recent_window_magnitude,warp_dx,warp_dy,warp_error,fit_applied,fit_model,fit_score,warp_active"
    )?;
    for row in rows {
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{}",
            row.frame_index,
            row.dx,
            row.dy,
            row.confidence,
            row.cumulative_dx,
            row.cumulative_dy,
            row.per_frame_magnitude,
            row.cumulative_magnitude,
            row.recent_window_magnitude,
            row.warp_dx,
            row.warp_dy,
            row.warp_error,
            row.fit_applied,
            row.fit_model.map(MotionModel::name).unwrap_or(""),
            row.fit_score
                .map(|value| format!("{value:.6}"))
                .unwrap_or_default(),
            row.warp_active,
        )?;
    }
    Ok(())
}

struct ConvertedFrame<D> {
    device_lab: Option<D>,
    host: Option<Array3<f32>>,
}

struct DetectionOutputs {
    delta_norm: Array2<u8>,
    mask: Array2<u8>,
}

#[derive(Clone)]
struct OutputBundle {
    frame_index: usize,
    source_bgr: Arc<Array3<u8>>,
    mask: Arc<Array2<u8>>,
    delta_norm: Option<Arc<Array2<u8>>>,
}

#[derive(Clone)]
struct OutputWorkerContext {
    mask_dir: Option<PathBuf>,
    overlay_dir: Option<PathBuf>,
    heatmap_dir: Option<PathBuf>,
    coco_images_dir: Option<PathBuf>,
    write_mask_pngs: bool,
    write_overlay_pngs: bool,
    write_heatmap_pngs: bool,
    annotations_enabled: bool,
    stream_coco_images: bool,
    store_frame_for_export: bool,
    annotation_segmentation_tolerance: f32,
    annotation_segmentation_max_edge_length: f32,
}

struct OutputWorkerPool {
    sender: SyncSender<OutputBundle>,
    record_receiver: Receiver<AnnotationFrame>,
    handles: Vec<thread::JoinHandle<Result<()>>>,
}

impl OutputWorkerPool {
    fn new(context: OutputWorkerContext) -> Self {
        let (sender, receiver) = sync_channel::<OutputBundle>(OUTPUT_QUEUE);
        let receiver = Arc::new(Mutex::new(receiver));
        let (record_sender, record_receiver) = channel::<AnnotationFrame>();
        let mut handles = Vec::with_capacity(OUTPUT_WORKERS);
        for _ in 0..OUTPUT_WORKERS {
            let receiver = Arc::clone(&receiver);
            let record_sender = record_sender.clone();
            let context = context.clone();
            handles.push(thread::spawn(move || -> Result<()> {
                loop {
                    let bundle = {
                        let receiver = receiver
                            .lock()
                            .map_err(|_| anyhow!("output worker queue poisoned"))?;
                        match receiver.recv() {
                            Ok(bundle) => bundle,
                            Err(_) => break,
                        }
                    };
                    if let Some(record) = process_output_bundle(&context, bundle)? {
                        record_sender
                            .send(record)
                            .map_err(|err| anyhow!("annotation result queue stopped: {err}"))?;
                    }
                }
                Ok(())
            }));
        }
        drop(record_sender);
        Self {
            sender,
            record_receiver,
            handles,
        }
    }

    fn submit(&self, bundle: OutputBundle) -> Result<()> {
        self.sender
            .send(bundle)
            .map_err(|err| anyhow!("output worker queue stopped: {err}"))
    }

    fn close(self) -> Result<Vec<AnnotationFrame>> {
        drop(self.sender);
        for handle in self.handles {
            handle
                .join()
                .map_err(|_| anyhow!("output worker thread panicked"))??;
        }
        let mut records = self.record_receiver.into_iter().collect::<Vec<_>>();
        records.sort_by_key(|record| record.frame_index);
        Ok(records)
    }
}

fn build_output_worker_context(
    config: &Config,
    options: &FastCliOptions,
    output_dir: &Path,
    stream_coco_images: bool,
) -> OutputWorkerContext {
    let coco_only = config.annotation_formats.len() == 1 && config.annotation_formats[0] == "coco";
    let coco_bbox_only = options.effective_coco_bbox_only();
    let has_non_coco_formats = config
        .annotation_formats
        .iter()
        .any(|format| format != "coco");
    let store_frame_for_export = !config.annotation_formats.is_empty()
        && (has_non_coco_formats || (coco_only && !coco_bbox_only && !stream_coco_images));
    OutputWorkerContext {
        mask_dir: config.write_mask_pngs.then(|| output_dir.join("masks")),
        overlay_dir: options
            .effective_write_overlay_pngs(config)
            .then(|| output_dir.join("overlays")),
        heatmap_dir: options
            .effective_write_heatmap_pngs(config)
            .then(|| output_dir.join("heatmap")),
        coco_images_dir: (stream_coco_images && !coco_bbox_only)
            .then(|| output_dir.join("formatCOCO").join("images").join("default")),
        write_mask_pngs: config.write_mask_pngs,
        write_overlay_pngs: options.effective_write_overlay_pngs(config),
        write_heatmap_pngs: options.effective_write_heatmap_pngs(config),
        annotations_enabled: !config.annotation_formats.is_empty(),
        stream_coco_images: stream_coco_images && !coco_bbox_only,
        store_frame_for_export,
        annotation_segmentation_tolerance: config.annotation_segmentation_tolerance,
        annotation_segmentation_max_edge_length: config.annotation_segmentation_max_edge_length,
    }
}

fn process_output_bundle(
    context: &OutputWorkerContext,
    bundle: OutputBundle,
) -> Result<Option<AnnotationFrame>> {
    let basename = format!("frame_{:06}.png", bundle.frame_index);
    if context.write_mask_pngs {
        write_gray_png(
            context
                .mask_dir
                .as_ref()
                .ok_or_else(|| anyhow!("mask output directory was not configured"))?
                .join(&basename),
            &bundle.mask,
        )?;
    }
    if context.write_overlay_pngs {
        let overlay = create_overlay(&bundle.source_bgr, &bundle.mask)?;
        write_bgr_png(
            context
                .overlay_dir
                .as_ref()
                .ok_or_else(|| anyhow!("overlay output directory was not configured"))?
                .join(&basename),
            &overlay,
        )?;
    }
    if context.write_heatmap_pngs {
        let delta_norm = bundle
            .delta_norm
            .as_ref()
            .ok_or_else(|| anyhow!("heatmap output requires delta_norm"))?;
        let heatmap = apply_turbo_colormap(delta_norm)?;
        write_bgr_png(
            context
                .heatmap_dir
                .as_ref()
                .ok_or_else(|| anyhow!("heatmap output directory was not configured"))?
                .join(&basename),
            &heatmap,
        )?;
    }
    if !context.annotations_enabled {
        return Ok(None);
    }

    let boxes = extract_bounding_boxes(
        &bundle.mask,
        context.annotation_segmentation_tolerance,
        context.annotation_segmentation_max_edge_length,
    )?;
    let frame_bgr = if context.stream_coco_images {
        write_bgr_png(
            context
                .coco_images_dir
                .as_ref()
                .ok_or_else(|| anyhow!("COCO image directory was not configured"))?
                .join(&basename),
            &bundle.source_bgr,
        )?;
        None
    } else if context.store_frame_for_export {
        Some((*bundle.source_bgr).clone())
    } else {
        None
    };

    Ok(Some(AnnotationFrame {
        frame_index: bundle.frame_index,
        frame_bgr,
        boxes,
    }))
}

pub fn run() -> Result<()> {
    let raw_args: Vec<OsString> = env::args_os().collect();
    let (options, stripped_args) = parse_fast_args(raw_args)?;
    if contains_debug_dump_flags(&stripped_args) {
        let status = forward_to_reference(stripped_args.iter().skip(1))?;
        return handle_reference_status(status);
    }
    if matches!(options.backend, BackendMode::Delegate) && !options.has_fast_only_flags() {
        let status = forward_to_reference(stripped_args.iter().skip(1))?;
        return handle_reference_status(status);
    }

    let cli_args = vrifa_cli::CliArgs::try_parse_from(stripped_args)
        .context("parsing fast-vrifa CLI arguments")?;
    let config = Config::try_from(cli_args)?;

    run_with_options(config, &options)
}

pub fn run_config(config: Config) -> Result<()> {
    vrifa_cli::run_binding_config(config).context("delegating bound config to reference vrifa")
}

pub fn run_with_backend_name(config: Config, backend: &str) -> Result<()> {
    run_with_backend(config, BackendMode::parse(backend)?)
}

pub fn run_with_backend(config: Config, backend: BackendMode) -> Result<()> {
    run_with_backend_options(config, backend, RunOptions::default())
}

pub fn run_with_backend_options(
    config: Config,
    backend: BackendMode,
    options: RunOptions,
) -> Result<()> {
    let options = FastCliOptions {
        backend,
        ffmpeg_postprocess: options.ffmpeg_postprocess,
        mask_only: options.mask_only,
        coco_bbox_only: options.coco_bbox_only,
    };
    run_with_options(config, &options)
}

fn run_with_options(config: Config, options: &FastCliOptions) -> Result<()> {
    match options.backend {
        BackendMode::Delegate => run_cpu_backend(config, options),
        BackendMode::Cpu => run_cpu_backend(config, options),
        BackendMode::Wgpu => run_wgpu_backend(config, options),
        BackendMode::Cuda => run_cuda_backend(config, options),
    }
}

pub fn delegated_backend_label() -> &'static str {
    "delegated-cpu"
}

pub fn reference_binary_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("VRIFA_BIN") {
        candidates.push(PathBuf::from(path));
    }

    let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_root.join("../../..");
    candidates.push(repo_root.join("vrifa-rs/target/release/vrifa"));
    candidates.push(repo_root.join("vrifa-rs/target/debug/vrifa"));
    candidates.push(PathBuf::from("vrifa"));
    candidates
}

pub fn locate_reference_binary() -> Result<PathBuf> {
    let candidates = reference_binary_candidates();
    for candidate in &candidates {
        if candidate.is_file() || candidate == Path::new("vrifa") {
            return Ok(candidate.clone());
        }
    }
    bail!("unable to locate the locked vrifa binary; build vrifa-rs first or set VRIFA_BIN")
}

pub fn forward_to_reference<I, S>(args: I) -> Result<ExitStatus>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let reference = locate_reference_binary()?;
    Command::new(&reference)
        .args(args)
        .status()
        .with_context(|| format!("launching delegated binary {}", reference.display()))
}

fn handle_reference_status(status: ExitStatus) -> Result<()> {
    if status.success() {
        return Ok(());
    }
    if let Some(code) = status.code() {
        bail!("reference vrifa exited with status code {code}");
    }
    bail!("reference vrifa terminated by signal");
}

fn parse_fast_args(raw_args: Vec<OsString>) -> Result<(FastCliOptions, Vec<OsString>)> {
    let mut args = raw_args.into_iter();
    let program = args.next().unwrap_or_else(|| OsString::from("fast-vrifa"));
    let mut stripped = vec![program];
    let mut options = FastCliOptions {
        backend: BackendMode::Delegate,
        ffmpeg_postprocess: false,
        mask_only: false,
        coco_bbox_only: false,
    };

    while let Some(arg) = args.next() {
        if let Some(text) = arg.to_str() {
            if text == "--backend" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--backend requires a value"))?;
                options.backend = BackendMode::parse(
                    value
                        .to_str()
                        .ok_or_else(|| anyhow!("--backend value must be valid UTF-8"))?,
                )?;
                continue;
            }
            if let Some(value) = text.strip_prefix("--backend=") {
                options.backend = BackendMode::parse(value)?;
                continue;
            }
            if text == "--ffmpeg-postprocess" {
                options.ffmpeg_postprocess = true;
                continue;
            }
            if text == "--no-ffmpeg-postprocess" {
                options.ffmpeg_postprocess = false;
                continue;
            }
            if let Some(value) = text.strip_prefix("--ffmpeg-postprocess=") {
                options.ffmpeg_postprocess = parse_fast_bool_flag(value, "--ffmpeg-postprocess")?;
                continue;
            }
            if text == "--mask-only" {
                options.mask_only = true;
                continue;
            }
            if text == "--coco-bbox-only" {
                options.coco_bbox_only = true;
                continue;
            }
            if let Some(value) = text.strip_prefix("--mask-only=") {
                options.mask_only = parse_fast_bool_flag(value, "--mask-only")?;
                continue;
            }
            if let Some(value) = text.strip_prefix("--coco-bbox-only=") {
                options.coco_bbox_only = parse_fast_bool_flag(value, "--coco-bbox-only")?;
                continue;
            }
        }
        stripped.push(arg);
    }

    Ok((options, stripped))
}

fn contains_debug_dump_flags(args: &[OsString]) -> bool {
    args.iter().any(|arg| {
        let Some(text) = arg.to_str() else {
            return false;
        };
        text == "--debug-dump-frames"
            || text == "--debug-dump-dir"
            || text.starts_with("--debug-dump-frames=")
            || text.starts_with("--debug-dump-dir=")
    })
}

fn parse_fast_bool_flag(raw: &str, flag: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" => Ok(true),
        "false" | "no" | "0" => Ok(false),
        other => bail!("{flag} expects true/false/yes/no/1/0, got '{other}'"),
    }
}

fn resolve_configured_roi_mask(config: &Config, shape: (usize, usize)) -> Array2<u8> {
    if let Some(path) = config.roi_mask.as_deref() {
        if let Ok(mask) = load_roi_mask_png(path, shape) {
            return mask;
        }
    }
    build_roi_mask(
        shape,
        resolve_roi_margins(
            config.roi_margin,
            config.roi_margin_top,
            config.roi_margin_bottom,
            config.roi_margin_left,
            config.roi_margin_right,
        ),
    )
}

fn load_roi_mask_png(path: &Path, shape: (usize, usize)) -> Result<Array2<u8>> {
    let img = opencv::imgcodecs::imread(
        &path.to_string_lossy(),
        opencv::imgcodecs::IMREAD_GRAYSCALE,
    )?;
    if img.empty() {
        bail!("could not read ROI mask PNG at {}", path.display());
    }
    let (height, width) = shape;
    let mat = if (img.rows() as usize, img.cols() as usize) == shape {
        img
    } else {
        let mut resized = core::Mat::default();
        opencv::imgproc::resize(
            &img,
            &mut resized,
            core::Size::new(width as i32, height as i32),
            0.0,
            0.0,
            opencv::imgproc::INTER_NEAREST,
        )?;
        resized
    };
    let arr = cvutil::mat_to_array2_u8(&mat)?;
    Ok(arr.mapv(|v| if v > 127 { 1 } else { 0 }))
}

fn run_cpu_backend(config: Config, options: &FastCliOptions) -> Result<()> {
    let backend = CpuBackend;
    run_hybrid_pipeline(config, &backend, options)
}

fn run_wgpu_backend(config: Config, options: &FastCliOptions) -> Result<()> {
    #[cfg(feature = "wgpu")]
    {
        let backend = WgpuBackend::new().context("initializing wgpu backend")?;
        return run_hybrid_pipeline(config, &backend, options);
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (config, options);
        bail!("--backend wgpu requires building fast-vrifa with --features wgpu");
    }
}

fn run_cuda_backend(config: Config, options: &FastCliOptions) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let backend = CudaBackend::new();
        if !matches!(backend.status(), fast_vrifa_core::BackendStatus::Ready) {
            let detail = backend
                .init_error()
                .unwrap_or("CUDA runtime initialization failed");
            bail!("--backend cuda is unavailable on this machine: {detail}");
        }
        if can_use_cuda_batched_peak_fast_path(&config) {
            return run_cuda_batched_peak_pipeline(config, &backend, options);
        }
        return run_hybrid_pipeline(config, &backend, options);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (config, options);
        bail!("--backend cuda requires building fast-vrifa with --features cuda");
    }
}

#[cfg(feature = "cuda")]
fn can_use_cuda_batched_peak_fast_path(config: &Config) -> bool {
    matches!(config.colorspace, ColorSpace::Cielab)
        && config.darken_only
        && config.peak_reference
        && !config.camera_stable
        && config.pre_delta_blur.is_no_op()
        && matches!(config.ref_mode, ReferenceMode::First)
        && matches!(config.threshold_mode, ThresholdMode::Otsu)
        && matches!(config.post_blur.kind, BlurKind::Gaussian | BlurKind::None)
        && config.roi_mask.is_none()
}

/// Decompose a BlurSpec into the legacy (kernel_size, blur_enabled)
/// pair the CUDA backend's BatchDetectorOptions still expects. CUDA only
/// implements Gaussian; other kinds bail and the harness routes the
/// trial to vrifa-rs.
fn legacy_blur_kernel(spec: BlurSpec) -> Result<(usize, bool)> {
    match spec.kind {
        BlurKind::None => Ok((1, false)),
        BlurKind::Gaussian => Ok((spec.size.max(1), true)),
        other => bail!(
            "fast-vrifa CUDA path implements only Gaussian/none blur; got {other:?}. \
             Re-run this trial with the CPU vrifa-rs binary."
        ),
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_batched_peak_pipeline(
    config: Config,
    backend: &CudaBackend,
    options: &FastCliOptions,
) -> Result<()> {
    fs::create_dir_all(&config.output_dir)?;
    let capture = videoio::VideoCapture::from_file_def(&config.video_path.to_string_lossy())
        .with_context(|| format!("opening video {}", config.video_path.display()))?;
    if !capture.is_opened()? {
        bail!("unable to open video: {}", config.video_path.display());
    }
    let total_frames = capture.get(videoio::CAP_PROP_FRAME_COUNT)? as usize;
    let fps = capture.get(videoio::CAP_PROP_FPS)?;
    let metadata = VideoMetadata {
        total_frames: (total_frames > 0).then_some(total_frames),
        fps: if fps > 0.0 { fps } else { 30.0 },
        width: capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as usize,
        height: capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as usize,
    };

    let roi_mask = resolve_configured_roi_mask(&config, (metadata.height, metadata.width));
    let device_roi_mask = backend.upload_mask_u8(&roi_mask)?;
    let roi_pixels = roi_mask.iter().filter(|value| **value > 0).count();

    let write_heatmap_pngs = options.effective_write_heatmap_pngs(&config);
    let write_overlay_video = options.effective_write_overlay_video(&config);
    let write_heatmap_video = options.effective_write_heatmap_video(&config);

    let video_dir = config.output_dir.join("videos");
    let raw_stream_dir = config.output_dir.join(".streams");
    let expected_video_frames = metadata
        .total_frames
        .map(|total_frames| total_frames / config.frame_step.max(1));
    let need_mask_stream = config.write_mask_video || write_overlay_video;
    let need_delta_norm_stream = write_heatmap_video;
    let mut mask_writer = None;
    let mut delta_norm_writer = None;
    if need_mask_stream || need_delta_norm_stream {
        fs::create_dir_all(&raw_stream_dir)?;
        if need_mask_stream {
            mask_writer = Some(AsyncRawVideoWriter::open(
                raw_stream_dir.join("mask.raw"),
                metadata.fps,
                metadata.width,
                metadata.height,
                RawPixelFormat::Gray8,
                expected_video_frames,
                32,
            )?);
        }
        if need_delta_norm_stream {
            delta_norm_writer = Some(AsyncRawVideoWriter::open(
                raw_stream_dir.join("delta_norm.raw"),
                metadata.fps,
                metadata.width,
                metadata.height,
                RawPixelFormat::Gray8,
                expected_video_frames,
                32,
            )?);
        }
    }

    let stream_coco_images = metadata
        .total_frames
        .map(|frames| frames > 300)
        .unwrap_or(true)
        && config.annotation_mode == "all"
        && config.annotation_formats.len() == 1
        && config.annotation_formats[0] == "coco";
    let output_context =
        build_output_worker_context(&config, options, &config.output_dir, stream_coco_images);
    if let Some(dir) = output_context.mask_dir.as_ref() {
        fs::create_dir_all(dir)?;
    }
    if let Some(dir) = output_context.overlay_dir.as_ref() {
        fs::create_dir_all(dir)?;
    }
    if let Some(dir) = output_context.heatmap_dir.as_ref() {
        fs::create_dir_all(dir)?;
    }
    if let Some(dir) = output_context.coco_images_dir.as_ref() {
        fs::create_dir_all(dir)?;
    }
    let needs_output_workers = output_context.write_mask_pngs
        || output_context.write_overlay_pngs
        || output_context.write_heatmap_pngs
        || output_context.annotations_enabled;
    let output_pool = needs_output_workers.then(|| OutputWorkerPool::new(output_context.clone()));

    let need_host_mask = needs_output_workers || config.write_mask_video || write_overlay_video;
    let need_host_delta_norm = write_heatmap_pngs || write_heatmap_video;
    let (blur_kernel, blur_enabled) = legacy_blur_kernel(config.post_blur)?;
    let batch_options = CudaBatchDetectorOptions {
        channel_weight: config.channel_weights[0],
        blur_kernel,
        blur_enabled,
        threshold_offset: config.threshold_offset,
        morph_shape: config.morph_shape,
        morph_kernel: config.morph_kernel,
        morph_close_iterations: config.morph_close_iterations,
        morph_open_iterations: config.morph_open_iterations,
        min_area: config.min_area,
        lock_frames: config.lock_frames,
        need_host_mask,
        need_host_delta_norm,
    };
    let mut detector_state =
        backend.create_batch_detector_state(roi_mask.dim(), config.lock_frames)?;

    let mut processed = 0usize;
    let mut processing_time_accum = 0.0f64;
    let run_start = Instant::now();
    let mut processed_records = Vec::new();
    let (decode_tx, decode_rx) = sync_channel::<(usize, Array3<u8>)>(CUDA_BATCH_SIZE.max(2));
    let video_path = config.video_path.clone();
    let frame_step = config.frame_step;
    let decode_handle = thread::spawn(move || -> Result<()> {
        let mut reader = VideoReader::open(&video_path)?;
        while let Some((frame_index, frame_bgr)) = reader.read_next()? {
            if frame_index % frame_step != 0 {
                continue;
            }
            decode_tx
                .send((frame_index, frame_bgr))
                .map_err(|err| anyhow!("decode queue stopped: {err}"))?;
        }
        Ok(())
    });

    let mut output_pool = output_pool;
    while let Ok((frame_index, frame_bgr)) = decode_rx.recv() {
        let compute_start = Instant::now();
        let device_bgr = backend.upload_frame_bgr(&frame_bgr)?;
        let device_lab = backend.convert_bgr_to_lab(&device_bgr)?;
        detector_state.peak =
            Some(backend.update_peak_brightness_device(&device_lab, detector_state.peak.as_ref())?);
        let delta = backend.compute_delta_darken_only_device(
            &device_lab,
            detector_state
                .peak
                .as_ref()
                .ok_or_else(|| anyhow!("CUDA peak state was not initialized"))?,
            &device_roi_mask,
            batch_options.channel_weight,
        )?;
        let delta_norm = backend
            .blur_and_normalize_delta(
                &delta,
                batch_options.blur_kernel,
                batch_options.blur_enabled,
            )?
            .ok_or_else(|| anyhow!("CUDA streamed path requires device blur+normalize"))?;
        let mut mask = backend
            .threshold_and_morph_mask_auto(
                &delta_norm,
                batch_options.threshold_offset,
                batch_options.morph_shape,
                batch_options.morph_kernel,
                batch_options.morph_close_iterations,
                batch_options.morph_open_iterations,
            )?
            .ok_or_else(|| anyhow!("CUDA streamed path requires device threshold+morph"))?;
        if batch_options.min_area > 0 {
            mask = backend
                .filter_min_area_mask(&mask, batch_options.min_area)?
                .ok_or_else(|| anyhow!("CUDA streamed path requires device min-area filtering"))?;
        }
        if batch_options.lock_frames > 0 {
            let state_lock = detector_state
                .lock
                .as_mut()
                .ok_or_else(|| anyhow!("CUDA streamed lock state was not initialized"))?;
            mask = backend
                .apply_locking_device(&mask, batch_options.lock_frames, state_lock)?
                .ok_or_else(|| anyhow!("CUDA streamed path requires device locking"))?;
        }

        let host_mask = need_host_mask
            .then(|| backend.download_mask_u8(&mask))
            .transpose()?;
        let host_delta_norm = need_host_delta_norm
            .then(|| backend.download_mask_u8(&delta_norm))
            .transpose()?;
        let source_bgr = Arc::new(frame_bgr);
        let mask = host_mask.map(Arc::new);
        let delta_norm = host_delta_norm.map(Arc::new);
        if let Some(pool) = output_pool.as_ref() {
            pool.submit(OutputBundle {
                frame_index,
                source_bgr: Arc::clone(&source_bgr),
                mask: Arc::clone(
                    mask.as_ref()
                        .ok_or_else(|| anyhow!("output generation requires a host mask"))?,
                ),
                delta_norm: delta_norm.as_ref().map(Arc::clone),
            })?;
        }
        if let Some(writer) = mask_writer.as_mut() {
            writer.write_gray(
                mask.as_ref()
                    .ok_or_else(|| anyhow!("mask video staging requires a host mask"))?
                    .clone(),
            )?;
        }
        if let Some(writer) = delta_norm_writer.as_mut() {
            writer.write_gray(
                delta_norm
                    .as_ref()
                    .ok_or_else(|| anyhow!("heatmap video staging requires host delta_norm"))?
                    .clone(),
            )?;
        }

        processed += 1;
        processing_time_accum += compute_start.elapsed().as_secs_f64();
    }
    decode_handle
        .join()
        .map_err(|_| anyhow!("decode thread panicked"))??;

    let mask_artifact = if let Some(writer) = mask_writer.take() {
        Some(writer.close()?)
    } else {
        None
    };
    let delta_norm_artifact = if let Some(writer) = delta_norm_writer.take() {
        Some(writer.close()?)
    } else {
        None
    };
    if let Some(pool) = output_pool.take() {
        processed_records = pool.close()?;
    }

    if options.ffmpeg_postprocess {
        fs::create_dir_all(&video_dir)?;
        let ffmpeg_bin = env::var_os("FFMPEG_BIN").unwrap_or_else(|| OsString::from("ffmpeg"));
        if config.write_mask_video {
            let artifact = mask_artifact
                .as_ref()
                .ok_or_else(|| anyhow!("mask video reconstruction requires a mask stream"))?;
            finalize_raw_stream_to_mp4(&ffmpeg_bin, artifact, video_dir.join("mask.mp4"))?;
        }
        if write_overlay_video {
            let mask_artifact = mask_artifact
                .as_ref()
                .ok_or_else(|| anyhow!("overlay video reconstruction requires a mask stream"))?;
            postprocess_overlay_video(
                &config,
                metadata.fps,
                metadata.width,
                metadata.height,
                mask_artifact,
                &raw_stream_dir,
                &ffmpeg_bin,
                video_dir.join("overlay.mp4"),
            )?;
        }
        if write_heatmap_video {
            if let Some(delta_norm_artifact) = delta_norm_artifact.as_ref() {
                postprocess_heatmap_video(
                    metadata.fps,
                    metadata.width,
                    metadata.height,
                    delta_norm_artifact,
                    &raw_stream_dir,
                    &ffmpeg_bin,
                    video_dir.join("heatmap.mp4"),
                )?;
            }
        }
    }

    if !config.annotation_formats.is_empty() {
        let selection = vrifa_core::sampling::choose_annotation_indices(
            processed_records.len(),
            &config.annotation_mode,
            config.annotation_count,
            config.annotation_stride,
        );
        vrifa_annotations::export_annotation_outputs(
            &config.output_dir,
            &processed_records,
            &selection,
            metadata.width,
            metadata.height,
            &config.annotation_formats,
        )?;
    }

    let avg_compute_time = if processed > 0 {
        processing_time_accum / processed as f64
    } else {
        0.0
    };
    let run_total_time = run_start.elapsed().as_secs_f64();
    let roi_fraction = roi_pixels as f64 / (metadata.width * metadata.height) as f64;
    let video_duration = metadata
        .total_frames
        .map(|frames| frames as f64 / metadata.fps);
    write_run_summary(
        &config,
        metadata.total_frames,
        processed,
        metadata.fps,
        video_duration,
        roi_pixels,
        roi_fraction,
        avg_compute_time,
        run_total_time,
        None,
        None,
        None,
        None,
    )?;
    println!(
        "Processed {processed} frames. Outputs saved to {}. Summary written to {}",
        config.output_dir.display(),
        config.output_dir.join("run_summary.yaml").display()
    );
    Ok(())
}

fn run_hybrid_pipeline<B>(config: Config, backend: &B, options: &FastCliOptions) -> Result<()>
where
    B: PeakImageBackend,
{
    fs::create_dir_all(&config.output_dir)?;
    let mut reader = VideoReader::open(&config.video_path)?;
    let metadata = reader.metadata();
    let Some((_, first_frame_bgr)) = reader.read_next()? else {
        bail!("failed to read reference frame");
    };
    let first_frame =
        convert_frame_with_backend(backend, &first_frame_bgr, config.colorspace, true)?;
    let first_frame_converted = first_frame
        .host
        .clone()
        .ok_or_else(|| anyhow!("first-frame conversion unexpectedly omitted host data"))?;

    let absolute_index = match config.ref_mode {
        ReferenceMode::Absolute(index) => Some(index),
        _ => None,
    };
    let reference_values_needed =
        config.camera_stable || !(config.peak_reference && config.darken_only);
    let device_delta_eligible =
        config.darken_only && matches!(config.colorspace, ColorSpace::Cielab);
    let use_device_plane_path =
        device_delta_eligible && (config.camera_stable || !config.pre_delta_blur.is_no_op());
    let absolute_reference = if reference_values_needed {
        if let Some(index) = absolute_index {
            if let Some(total) = metadata.total_frames {
                if index >= total {
                    bail!("requested absolute frame index exceeds video length");
                }
            }
            let frame = reader
                .read_frame_at_zero_based(index)
                .with_context(|| format!("reading absolute reference frame {index}"))?
                .ok_or_else(|| anyhow!("unable to read absolute reference frame {index}"))?;
            Some(
                convert_frame_with_backend(backend, &frame, config.colorspace, true)?
                    .host
                    .ok_or_else(|| {
                        anyhow!("absolute reference conversion unexpectedly omitted host data")
                    })?,
            )
        } else {
            Some(first_frame_converted.clone())
        }
    } else {
        None
    };
    let first_reference_device = if device_delta_eligible && reference_values_needed {
        Some(backend.upload_plane_f32(&first_frame_converted.slice(s![.., .., 0]).to_owned())?)
    } else {
        None
    };
    let absolute_reference_device = if device_delta_eligible {
        absolute_reference
            .as_ref()
            .map(|reference| backend.upload_plane_f32(&reference.slice(s![.., .., 0]).to_owned()))
            .transpose()?
    } else {
        None
    };

    let mut running_reference = reference_values_needed.then_some(first_frame_converted.clone());
    let mut prev_buffer: Option<VecDeque<Array3<f32>>> = if reference_values_needed {
        match config.ref_mode {
            ReferenceMode::Prev(offset) => Some(VecDeque::with_capacity(offset)),
            _ => None,
        }
    } else {
        None
    };
    let mut dynamic_reader =
        if reference_values_needed && matches!(config.ref_mode, ReferenceMode::Dynamic) {
            Some(VideoReader::open(&config.video_path)?)
        } else {
            None
        };
    let mut dynamic_cache: IndexMap<usize, Array3<f32>> = IndexMap::new();
    let mut dynamic_measurements: Vec<(f32, f32)> = Vec::new();
    let mut dynamic_factor: Option<f32> = None;
    let mut dynamic_lag_log = Vec::new();
    let mut dynamic_first_lag: Option<usize> = None;
    let mut dynamic_last_lag: Option<usize> = None;

    let roi_mask = resolve_configured_roi_mask(
        &config,
        (first_frame_converted.dim().0, first_frame_converted.dim().1),
    );
    let device_roi_mask = backend.upload_mask_u8(&roi_mask)?;
    let roi_pixels = roi_mask.iter().filter(|value| **value > 0).count();
    let mut lock_state = (config.lock_frames > 0).then(|| LockState::new(roi_mask.dim()));
    let mut peak_brightness_map = if config.peak_reference && !device_delta_eligible {
        Some(peak_frame_plane(
            &first_frame_converted,
            config.pre_delta_blur,
        )?)
    } else {
        None
    };
    let mut peak_brightness_device = if config.peak_reference && device_delta_eligible {
        let first_l = first_frame
            .device_lab
            .as_ref()
            .map(|frame| backend.extract_l_plane(frame))
            .transpose()?
            .ok_or_else(|| anyhow!("device peak path requires a device CIELAB frame"))?;
        Some(
            if use_device_plane_path && !config.pre_delta_blur.is_no_op() {
                backend.blur_plane_f32_device(&first_l, config.pre_delta_blur.size.max(1))?
            } else {
                first_l
            },
        )
    } else {
        None
    };
    let mut camera_stable_state = CameraStableState::new();

    let write_heatmap_pngs = options.effective_write_heatmap_pngs(&config);
    let write_overlay_video = options.effective_write_overlay_video(&config);
    let write_heatmap_video = options.effective_write_heatmap_video(&config);

    let video_dir = config.output_dir.join("videos");
    let raw_stream_dir = config.output_dir.join(".streams");
    let expected_video_frames = metadata
        .total_frames
        .map(|total_frames| total_frames / config.frame_step.max(1));
    let need_mask_stream = config.write_mask_video || write_overlay_video;
    let need_delta_norm_stream = write_heatmap_video;
    let mut mask_writer = None;
    let mut delta_norm_writer = None;
    if need_mask_stream || need_delta_norm_stream {
        fs::create_dir_all(&raw_stream_dir)?;
        if need_mask_stream {
            mask_writer = Some(AsyncRawVideoWriter::open(
                raw_stream_dir.join("mask.raw"),
                metadata.fps,
                metadata.width,
                metadata.height,
                RawPixelFormat::Gray8,
                expected_video_frames,
                32,
            )?);
        }
        if need_delta_norm_stream {
            delta_norm_writer = Some(AsyncRawVideoWriter::open(
                raw_stream_dir.join("delta_norm.raw"),
                metadata.fps,
                metadata.width,
                metadata.height,
                RawPixelFormat::Gray8,
                expected_video_frames,
                32,
            )?);
        }
    }
    let stream_coco_images = metadata
        .total_frames
        .map(|frames| frames > 300)
        .unwrap_or(true)
        && config.annotation_mode == "all"
        && config.annotation_formats.len() == 1
        && config.annotation_formats[0] == "coco";
    let output_context =
        build_output_worker_context(&config, options, &config.output_dir, stream_coco_images);
    if let Some(dir) = output_context.mask_dir.as_ref() {
        fs::create_dir_all(dir)?;
    }
    if let Some(dir) = output_context.overlay_dir.as_ref() {
        fs::create_dir_all(dir)?;
    }
    if let Some(dir) = output_context.heatmap_dir.as_ref() {
        fs::create_dir_all(dir)?;
    }
    if let Some(dir) = output_context.coco_images_dir.as_ref() {
        fs::create_dir_all(dir)?;
    }
    let needs_output_workers = output_context.write_mask_pngs
        || output_context.write_overlay_pngs
        || output_context.write_heatmap_pngs
        || output_context.annotations_enabled;
    let output_pool = needs_output_workers.then(|| OutputWorkerPool::new(output_context.clone()));
    let supports_device_auto_threshold =
        matches!(config.threshold_mode, ThresholdMode::Otsu);
    let mut device_lock_state = if config.lock_frames > 0 {
        backend.create_lock_state(roi_mask.dim())?
    } else {
        None
    };
    let need_host_mask = needs_output_workers
        || config.write_mask_video
        || write_overlay_video
        || matches!(config.ref_mode, ReferenceMode::Dynamic)
        || (config.lock_frames > 0 && device_lock_state.is_none());
    let need_host_delta_norm =
        write_heatmap_pngs || write_heatmap_video || !supports_device_auto_threshold;

    reader.seek_zero()?;
    let mut processed = 0usize;
    let mut processing_time_accum = 0.0f64;
    let run_start = Instant::now();
    let mut processed_records = Vec::new();
    let mut output_pool = output_pool;
    let need_host_current = config.camera_stable
        || !device_delta_eligible
        || (reference_values_needed
            && matches!(
                config.ref_mode,
                ReferenceMode::Running | ReferenceMode::Prev(_) | ReferenceMode::Dynamic
            ));

    while let Some((frame_index, frame_bgr)) = reader.read_next()? {
        let current =
            convert_frame_with_backend(backend, &frame_bgr, config.colorspace, need_host_current)?;
        let frame_converted = current.host.as_ref();
        let mut reference_frame_index = 1usize;
        let reference_for_frame = if reference_values_needed {
            Some(match config.ref_mode {
                ReferenceMode::First => first_frame_converted.clone(),
                ReferenceMode::Absolute(_) => {
                    reference_frame_index = absolute_index.filter(|index| *index > 0).unwrap_or(1);
                    absolute_reference
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(|| first_frame_converted.clone())
                }
                ReferenceMode::Running => running_reference
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| first_frame_converted.clone()),
                ReferenceMode::Prev(offset) => {
                    if let Some(buffer) = &prev_buffer {
                        if buffer.len() >= offset {
                            buffer
                                .front()
                                .cloned()
                                .unwrap_or_else(|| first_frame_converted.clone())
                        } else {
                            first_frame_converted.clone()
                        }
                    } else {
                        first_frame_converted.clone()
                    }
                }
                ReferenceMode::Dynamic => {
                    let params = DynamicReferenceParams {
                        factor: dynamic_factor,
                        target_fraction: config.dynamic_target_fraction,
                        lag_scale: config.dynamic_lag_scale,
                        linear_mode: config.dynamic_lag_linear,
                        linear_start: config.dynamic_lag_linear_start,
                        linear_max: config.dynamic_lag_linear_max,
                        total_frames: metadata.total_frames,
                    };
                    let ref_index = select_dynamic_reference_index(
                        frame_index,
                        metadata.fps as f32,
                        roi_pixels,
                        &params,
                    );
                    reference_frame_index = ref_index;
                    fetch_reference_converted(
                        ref_index,
                        dynamic_reader.as_mut(),
                        &mut dynamic_cache,
                        config.dynamic_ref_cache_size,
                        &first_frame_converted,
                        config.colorspace,
                        backend,
                    )?
                }
            })
        } else {
            if matches!(config.ref_mode, ReferenceMode::Dynamic) {
                let params = DynamicReferenceParams {
                    factor: dynamic_factor,
                    target_fraction: config.dynamic_target_fraction,
                    lag_scale: config.dynamic_lag_scale,
                    linear_mode: config.dynamic_lag_linear,
                    linear_start: config.dynamic_lag_linear_start,
                    linear_max: config.dynamic_lag_linear_max,
                    total_frames: metadata.total_frames,
                };
                reference_frame_index = select_dynamic_reference_index(
                    frame_index,
                    metadata.fps as f32,
                    roi_pixels,
                    &params,
                );
            } else if matches!(config.ref_mode, ReferenceMode::Absolute(_)) {
                reference_frame_index = absolute_index.filter(|index| *index > 0).unwrap_or(1);
            }
            None
        };

        if frame_index % config.frame_step == 0 {
            let compute_start = Instant::now();
            let morph_params = MorphologyParams {
                post_blur: config.post_blur,
                morph_kernel: config.morph_kernel,
                min_area: config.min_area,
                threshold_mode: config.threshold_mode,
                threshold_offset: config.threshold_offset,
                morph_shape: config.morph_shape,
                morph_close_iterations: config.morph_close_iterations,
                morph_open_iterations: config.morph_open_iterations,
            };
            let camera_actions = if config.camera_stable {
                prepare_camera_stable_actions(
                    frame_index,
                    frame_converted.ok_or_else(|| {
                        anyhow!("camera-stable path requires a converted host frame")
                    })?,
                    reference_for_frame.as_ref().ok_or_else(|| {
                        anyhow!("camera-stable path requires a host reference frame")
                    })?,
                    &roi_mask,
                    reference_token_for_frame(&config.ref_mode, frame_index, reference_frame_index),
                    &config,
                    &mut camera_stable_state,
                )?
            } else {
                CameraStableActions {
                    warp: None,
                    peak_shift: PeakShiftAction::Keep,
                }
            };

            let mut host_mask: Option<Array2<u8>> = None;
            let mut host_delta_norm: Option<Array2<u8>> = None;
            if let Some(device_lab) = current.device_lab.as_ref() {
                if device_delta_eligible && use_device_plane_path {
                    let mut frame_l = backend.extract_l_plane(device_lab)?;
                    if !config.pre_delta_blur.is_no_op() {
                        frame_l = backend
                            .blur_plane_f32_device(&frame_l, config.pre_delta_blur.size.max(1))?;
                    }
                    if config.peak_reference {
                        let peak_base = apply_peak_shift_device(
                            backend,
                            peak_brightness_device.as_ref(),
                            &camera_actions.peak_shift,
                        )?;
                        peak_brightness_device =
                            Some(backend.update_peak_brightness_plane_device(
                                &frame_l,
                                peak_base.as_ref(),
                            )?);
                    }

                    let device_delta = if config.peak_reference {
                        backend.compute_delta_darken_only_planes_device(
                            &frame_l,
                            peak_brightness_device.as_ref().ok_or_else(|| {
                                anyhow!("peak reference was enabled without a device peak map")
                            })?,
                            &device_roi_mask,
                            config.channel_weights[0],
                        )?
                    } else {
                        let reference_host = reference_for_frame
                            .as_ref()
                            .or((!reference_values_needed).then_some(&first_frame_converted))
                            .ok_or_else(|| anyhow!("missing reference frame for device delta"))?;
                        let mut reference_plane = backend
                            .upload_plane_f32(&reference_host.slice(s![.., .., 0]).to_owned())?;
                        if let Some(warp) = camera_actions.warp.as_ref() {
                            reference_plane =
                                backend.warp_plane_f32_device(&reference_plane, warp)?;
                        }
                        if !config.pre_delta_blur.is_no_op() {
                            reference_plane = backend.blur_plane_f32_device(
                                &reference_plane,
                                config.pre_delta_blur.size.max(1),
                            )?;
                        }
                        backend.compute_delta_darken_only_planes_device(
                            &frame_l,
                            &reference_plane,
                            &device_roi_mask,
                            config.channel_weights[0],
                        )?
                    };
                    let (post_blur_kernel_legacy, post_blur_enabled_legacy) =
                        legacy_blur_kernel(morph_params.post_blur)?;
                    if let Some(device_delta_norm) = backend.blur_and_normalize_delta(
                        &device_delta,
                        post_blur_kernel_legacy,
                        post_blur_enabled_legacy,
                    )? {
                        let mut device_mask = if supports_device_auto_threshold {
                            backend.threshold_and_morph_mask_auto(
                                &device_delta_norm,
                                morph_params.threshold_offset,
                                morph_params.morph_shape,
                                morph_params.morph_kernel,
                                morph_params.morph_close_iterations,
                                morph_params.morph_open_iterations,
                            )?
                        } else {
                            None
                        };

                        if device_mask.is_none() {
                            let delta_norm = backend.download_mask_u8(&device_delta_norm)?;
                            let threshold_value = threshold::choose_threshold(
                                &delta_norm,
                                &roi_mask,
                                morph_params.threshold_mode,
                                morph_params.threshold_offset,
                            )?;
                            host_delta_norm = Some(delta_norm.clone());
                            if let Some(mask) = backend.threshold_and_morph_mask(
                                &device_delta_norm,
                                threshold_value,
                                morph_params.morph_shape,
                                morph_params.morph_kernel,
                                morph_params.morph_close_iterations,
                                morph_params.morph_open_iterations,
                            )? {
                                device_mask = Some(mask);
                            } else {
                                let unlocked = detect_mask_from_delta_norm_threshold(
                                    &delta_norm,
                                    threshold_value,
                                    &morph_params,
                                )?;
                                host_mask = Some(apply_locking(
                                    &unlocked,
                                    config.lock_frames,
                                    lock_state.as_mut(),
                                )?);
                            }
                        }

                        if let Some(mask) = device_mask.take() {
                            let mask = if morph_params.min_area > 0 {
                                if let Some(filtered_mask) =
                                    backend.filter_min_area_mask(&mask, morph_params.min_area)?
                                {
                                    filtered_mask
                                } else {
                                    let filtered = filter_min_area_array(
                                        &backend.download_mask_u8(&mask)?,
                                        morph_params.min_area,
                                    )?;
                                    host_mask = Some(apply_locking(
                                        &filtered,
                                        config.lock_frames,
                                        lock_state.as_mut(),
                                    )?);
                                    mask
                                }
                            } else {
                                mask
                            };

                            if host_mask.is_none() {
                                let mask = if config.lock_frames > 0 {
                                    if let Some(state) = device_lock_state.as_mut() {
                                        if let Some(locked_mask) = backend.apply_locking_device(
                                            &mask,
                                            config.lock_frames,
                                            state,
                                        )? {
                                            locked_mask
                                        } else {
                                            host_mask = Some(apply_locking(
                                                &backend.download_mask_u8(&mask)?,
                                                config.lock_frames,
                                                lock_state.as_mut(),
                                            )?);
                                            mask
                                        }
                                    } else {
                                        host_mask = Some(apply_locking(
                                            &backend.download_mask_u8(&mask)?,
                                            config.lock_frames,
                                            lock_state.as_mut(),
                                        )?);
                                        mask
                                    }
                                } else {
                                    mask
                                };

                                if need_host_mask && host_mask.is_none() {
                                    host_mask = Some(backend.download_mask_u8(&mask)?);
                                }
                            }
                        }

                        if need_host_delta_norm && host_delta_norm.is_none() {
                            host_delta_norm = Some(backend.download_mask_u8(&device_delta_norm)?);
                        }
                    } else {
                        let delta = backend.download_plane_f32(&device_delta)?;
                        let detect = detect_mask_and_delta_norm(&delta, &roi_mask, &morph_params)?;
                        host_mask = Some(apply_locking(
                            &detect.mask,
                            config.lock_frames,
                            lock_state.as_mut(),
                        )?);
                        host_delta_norm = Some(detect.delta_norm);
                    }
                } else if device_delta_eligible {
                    if config.peak_reference {
                        peak_brightness_device = Some(backend.update_peak_brightness_device(
                            device_lab,
                            peak_brightness_device.as_ref(),
                        )?);
                    }
                    let device_delta = if config.peak_reference {
                        backend.compute_delta_darken_only_device(
                            device_lab,
                            peak_brightness_device.as_ref().ok_or_else(|| {
                                anyhow!("peak reference was enabled without a device peak map")
                            })?,
                            &device_roi_mask,
                            config.channel_weights[0],
                        )?
                    } else if let Some(device_reference_plane) = match config.ref_mode {
                        ReferenceMode::First => first_reference_device.as_ref(),
                        ReferenceMode::Absolute(_) => absolute_reference_device
                            .as_ref()
                            .or(first_reference_device.as_ref()),
                        _ => None,
                    } {
                        backend.compute_delta_darken_only_device(
                            device_lab,
                            device_reference_plane,
                            &device_roi_mask,
                            config.channel_weights[0],
                        )?
                    } else {
                        let reference_plane = reference_for_frame
                            .as_ref()
                            .ok_or_else(|| anyhow!("missing reference frame for device delta"))?
                            .slice(s![.., .., 0])
                            .to_owned();
                        backend.compute_delta_darken_only(
                            device_lab,
                            &reference_plane,
                            &device_roi_mask,
                            config.channel_weights[0],
                        )?
                    };
                    let (post_blur_kernel_legacy, post_blur_enabled_legacy) =
                        legacy_blur_kernel(morph_params.post_blur)?;
                    if let Some(device_delta_norm) = backend.blur_and_normalize_delta(
                        &device_delta,
                        post_blur_kernel_legacy,
                        post_blur_enabled_legacy,
                    )? {
                        let mut device_mask = if supports_device_auto_threshold {
                            backend.threshold_and_morph_mask_auto(
                                &device_delta_norm,
                                morph_params.threshold_offset,
                                morph_params.morph_shape,
                                morph_params.morph_kernel,
                                morph_params.morph_close_iterations,
                                morph_params.morph_open_iterations,
                            )?
                        } else {
                            None
                        };

                        if device_mask.is_none() {
                            let delta_norm = backend.download_mask_u8(&device_delta_norm)?;
                            let threshold_value = threshold::choose_threshold(
                                &delta_norm,
                                &roi_mask,
                                morph_params.threshold_mode,
                                morph_params.threshold_offset,
                            )?;
                            host_delta_norm = Some(delta_norm.clone());
                            if let Some(mask) = backend.threshold_and_morph_mask(
                                &device_delta_norm,
                                threshold_value,
                                morph_params.morph_shape,
                                morph_params.morph_kernel,
                                morph_params.morph_close_iterations,
                                morph_params.morph_open_iterations,
                            )? {
                                device_mask = Some(mask);
                            } else {
                                let unlocked = detect_mask_from_delta_norm_threshold(
                                    &delta_norm,
                                    threshold_value,
                                    &morph_params,
                                )?;
                                host_mask = Some(apply_locking(
                                    &unlocked,
                                    config.lock_frames,
                                    lock_state.as_mut(),
                                )?);
                            }
                        }

                        if let Some(mask) = device_mask.take() {
                            let mask = if morph_params.min_area > 0 {
                                if let Some(filtered_mask) =
                                    backend.filter_min_area_mask(&mask, morph_params.min_area)?
                                {
                                    filtered_mask
                                } else {
                                    let filtered = filter_min_area_array(
                                        &backend.download_mask_u8(&mask)?,
                                        morph_params.min_area,
                                    )?;
                                    host_mask = Some(apply_locking(
                                        &filtered,
                                        config.lock_frames,
                                        lock_state.as_mut(),
                                    )?);
                                    mask
                                }
                            } else {
                                mask
                            };

                            if host_mask.is_none() {
                                let mask = if config.lock_frames > 0 {
                                    if let Some(state) = device_lock_state.as_mut() {
                                        if let Some(locked_mask) = backend.apply_locking_device(
                                            &mask,
                                            config.lock_frames,
                                            state,
                                        )? {
                                            locked_mask
                                        } else {
                                            host_mask = Some(apply_locking(
                                                &backend.download_mask_u8(&mask)?,
                                                config.lock_frames,
                                                lock_state.as_mut(),
                                            )?);
                                            mask
                                        }
                                    } else {
                                        host_mask = Some(apply_locking(
                                            &backend.download_mask_u8(&mask)?,
                                            config.lock_frames,
                                            lock_state.as_mut(),
                                        )?);
                                        mask
                                    }
                                } else {
                                    mask
                                };

                                if need_host_mask && host_mask.is_none() {
                                    host_mask = Some(backend.download_mask_u8(&mask)?);
                                }
                            }
                        }

                        if need_host_delta_norm && host_delta_norm.is_none() {
                            host_delta_norm = Some(backend.download_mask_u8(&device_delta_norm)?);
                        }
                    } else {
                        let delta = backend.download_plane_f32(&device_delta)?;
                        let detect = detect_mask_and_delta_norm(&delta, &roi_mask, &morph_params)?;
                        host_mask = Some(apply_locking(
                            &detect.mask,
                            config.lock_frames,
                            lock_state.as_mut(),
                        )?);
                        host_delta_norm = Some(detect.delta_norm);
                    }
                } else {
                    let frame_host = frame_converted.ok_or_else(|| {
                        anyhow!("host delta path requires a converted host frame")
                    })?;
                    let host_reference = reference_for_frame
                        .as_ref()
                        .or((!reference_values_needed).then_some(&first_frame_converted))
                        .ok_or_else(|| anyhow!("missing reference frame for host delta"))?;
                    let aligned_reference = if let Some(warp) = camera_actions.warp.as_ref() {
                        apply_warp(host_reference, warp)?
                    } else {
                        host_reference.clone()
                    };
                    if config.peak_reference {
                        let peak_base = apply_peak_shift_host(
                            peak_brightness_map.as_ref(),
                            &camera_actions.peak_shift,
                        )?;
                        let peak_frame =
                            peak_frame_plane(frame_host, config.pre_delta_blur)?;
                        peak_brightness_map = Some(update_peak_brightness_plane(
                            &peak_frame,
                            peak_base.as_ref(),
                        )?);
                    }
                    let frame_for_delta = if !config.pre_delta_blur.is_no_op() {
                        blur::blur_frame(frame_host, config.pre_delta_blur)?
                    } else {
                        frame_host.clone()
                    };
                    let reference_for_delta = if !config.pre_delta_blur.is_no_op() {
                        blur::blur_frame(&aligned_reference, config.pre_delta_blur)?
                    } else {
                        aligned_reference
                    };
                    let delta = compute_delta(
                        &frame_for_delta,
                        &reference_for_delta,
                        &roi_mask,
                        &config.channel_weights,
                        config.darken_only,
                        peak_brightness_map
                            .as_ref()
                            .filter(|_| config.peak_reference),
                    )?;
                    let detect = detect_mask_and_delta_norm(&delta, &roi_mask, &morph_params)?;
                    host_mask = Some(apply_locking(
                        &detect.mask,
                        config.lock_frames,
                        lock_state.as_mut(),
                    )?);
                    host_delta_norm = Some(detect.delta_norm);
                }
            } else {
                let frame_host = frame_converted
                    .ok_or_else(|| anyhow!("host delta path requires a converted host frame"))?;
                let host_reference = reference_for_frame
                    .as_ref()
                    .or((!reference_values_needed).then_some(&first_frame_converted))
                    .ok_or_else(|| anyhow!("missing reference frame for host delta"))?;
                let aligned_reference = if let Some(warp) = camera_actions.warp.as_ref() {
                    apply_warp(host_reference, warp)?
                } else {
                    host_reference.clone()
                };
                if config.peak_reference {
                    let peak_base = apply_peak_shift_host(
                        peak_brightness_map.as_ref(),
                        &camera_actions.peak_shift,
                    )?;
                    let peak_frame = peak_frame_plane(frame_host, config.pre_delta_blur)?;
                    peak_brightness_map = Some(update_peak_brightness_plane(
                        &peak_frame,
                        peak_base.as_ref(),
                    )?);
                }
                let frame_for_delta = if !config.pre_delta_blur.is_no_op() {
                    blur::blur_frame(frame_host, config.pre_delta_blur)?
                } else {
                    frame_host.clone()
                };
                let reference_for_delta = if !config.pre_delta_blur.is_no_op() {
                    blur::blur_frame(&aligned_reference, config.pre_delta_blur)?
                } else {
                    aligned_reference
                };
                let delta = compute_delta(
                    &frame_for_delta,
                    &reference_for_delta,
                    &roi_mask,
                    &config.channel_weights,
                    config.darken_only,
                    peak_brightness_map
                        .as_ref()
                        .filter(|_| config.peak_reference),
                )?;
                let detect = detect_mask_and_delta_norm(&delta, &roi_mask, &morph_params)?;
                host_mask = Some(apply_locking(
                    &detect.mask,
                    config.lock_frames,
                    lock_state.as_mut(),
                )?);
                host_delta_norm = Some(detect.delta_norm);
            }
            if config.camera_stable {
                if let Some(mask) = host_mask.as_ref() {
                    camera_stable_state.prev_registration_mask = Some(mask.clone());
                }
            }
            let mask = host_mask.map(Arc::new);
            let delta_norm = host_delta_norm.map(Arc::new);
            let source_bgr = Arc::new(frame_bgr);
            if let Some(pool) = output_pool.as_ref() {
                pool.submit(OutputBundle {
                    frame_index,
                    source_bgr: Arc::clone(&source_bgr),
                    mask: Arc::clone(
                        mask.as_ref()
                            .ok_or_else(|| anyhow!("output generation requires a host mask"))?,
                    ),
                    delta_norm: delta_norm.as_ref().map(Arc::clone),
                })?;
            }
            if let Some(writer) = mask_writer.as_mut() {
                writer.write_gray(
                    mask.as_ref()
                        .ok_or_else(|| anyhow!("mask video staging requires a host mask"))?
                        .clone(),
                )?;
            }
            if let Some(writer) = delta_norm_writer.as_mut() {
                writer.write_gray(
                    delta_norm
                        .as_ref()
                        .ok_or_else(|| anyhow!("heatmap video staging requires host delta_norm"))?
                        .clone(),
                )?;
            }

            if matches!(config.ref_mode, ReferenceMode::Dynamic) {
                let lag = frame_index.saturating_sub(reference_frame_index);
                dynamic_first_lag.get_or_insert(lag);
                dynamic_last_lag = Some(lag);
                dynamic_lag_log.push((frame_index, lag));
                let mask_area = mask
                    .as_ref()
                    .ok_or_else(|| anyhow!("dynamic reference mode requires a host mask"))?
                    .iter()
                    .filter(|value| **value > 0)
                    .count();
                if dynamic_factor.is_none()
                    && metadata.fps > 0.0
                    && frame_index > 1
                    && mask_area > 0
                {
                    let time_seconds = (frame_index - 1) as f32 / metadata.fps as f32;
                    dynamic_measurements.push((time_seconds, mask_area as f32));
                    if dynamic_measurements.len() >= config.dynamic_calibration_frames {
                        dynamic_factor = compute_dynamic_factor(&dynamic_measurements);
                    }
                }
            }

            processed += 1;
            processing_time_accum += compute_start.elapsed().as_secs_f64();
        }

        if let Some(buffer) = prev_buffer.as_mut() {
            if let ReferenceMode::Prev(offset) = config.ref_mode {
                if buffer.len() == offset {
                    buffer.pop_front();
                }
                buffer.push_back(
                    frame_converted
                        .ok_or_else(|| {
                            anyhow!("prev reference mode requires a converted host frame")
                        })?
                        .clone(),
                );
            }
        }
        if matches!(config.ref_mode, ReferenceMode::Running) && running_reference.is_some() {
            let alpha = config.ref_running_alpha;
            let frame_converted = frame_converted
                .ok_or_else(|| anyhow!("running reference mode requires a converted host frame"))?;
            for (running, current) in running_reference
                .as_mut()
                .ok_or_else(|| anyhow!("running reference storage was not initialized"))?
                .iter_mut()
                .zip(frame_converted.iter())
            {
                *running = (1.0 - alpha) * *running + alpha * *current;
            }
        }
    }

    if config.camera_stable && !camera_stable_state.motion_trace.is_empty() {
        write_motion_trace(
            &config.output_dir.join("motion_trace.csv"),
            &camera_stable_state.motion_trace,
        )?;
    }

    let mask_artifact = if let Some(writer) = mask_writer.take() {
        Some(writer.close()?)
    } else {
        None
    };
    let delta_norm_artifact = if let Some(writer) = delta_norm_writer.take() {
        Some(writer.close()?)
    } else {
        None
    };
    if let Some(pool) = output_pool.take() {
        processed_records = pool.close()?;
    }
    if options.ffmpeg_postprocess {
        fs::create_dir_all(&video_dir)?;
        let ffmpeg_bin = env::var_os("FFMPEG_BIN").unwrap_or_else(|| OsString::from("ffmpeg"));
        if config.write_mask_video {
            let artifact = mask_artifact
                .as_ref()
                .ok_or_else(|| anyhow!("mask video reconstruction requires a mask stream"))?;
            finalize_raw_stream_to_mp4(&ffmpeg_bin, artifact, video_dir.join("mask.mp4"))?;
        }
        if write_overlay_video {
            let artifact = mask_artifact
                .as_ref()
                .ok_or_else(|| anyhow!("overlay video reconstruction requires a mask stream"))?;
            postprocess_overlay_video(
                &config,
                metadata.fps,
                metadata.width,
                metadata.height,
                artifact,
                &raw_stream_dir,
                &ffmpeg_bin,
                video_dir.join("overlay.mp4"),
            )?;
        }
        if write_heatmap_video {
            if let Some(artifact) = delta_norm_artifact.as_ref() {
                postprocess_heatmap_video(
                    metadata.fps,
                    metadata.width,
                    metadata.height,
                    artifact,
                    &raw_stream_dir,
                    &ffmpeg_bin,
                    video_dir.join("heatmap.mp4"),
                )?;
            }
        }
        if let Some(artifact) = mask_artifact.as_ref() {
            let _ = fs::remove_file(&artifact.path);
        }
        if let Some(artifact) = delta_norm_artifact.as_ref() {
            let _ = fs::remove_file(&artifact.path);
        }
    }

    if !config.annotation_formats.is_empty() {
        let selection = vrifa_core::sampling::choose_annotation_indices(
            processed_records.len(),
            &config.annotation_mode,
            config.annotation_count,
            config.annotation_stride,
        );
        vrifa_annotations::export_annotation_outputs(
            &config.output_dir,
            &processed_records,
            &selection,
            metadata.width,
            metadata.height,
            &config.annotation_formats,
        )?;
    }

    let run_total_time = run_start.elapsed().as_secs_f64();
    let roi_fraction = if !roi_mask.is_empty() {
        roi_pixels as f64 / roi_mask.len() as f64
    } else {
        0.0
    };
    let avg_compute_time = if processed > 0 {
        processing_time_accum / processed as f64
    } else {
        0.0
    };
    let video_duration = metadata
        .total_frames
        .map(|frames| frames as f64 / metadata.fps);
    write_run_summary(
        &config,
        metadata.total_frames,
        processed,
        metadata.fps,
        video_duration,
        roi_pixels,
        roi_fraction,
        avg_compute_time,
        run_total_time,
        dynamic_factor,
        dynamic_first_lag,
        dynamic_last_lag,
        absolute_index,
    )?;

    if let Some(path) = &config.dynamic_lag_log {
        if !dynamic_lag_log.is_empty() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(path)?;
            writeln!(file, "frame,lag")?;
            for (frame, lag) in dynamic_lag_log {
                writeln!(file, "{frame},{lag}")?;
            }
        }
    }

    println!(
        "Processed {processed} frames. Outputs saved to {}. Summary written to {}",
        config.output_dir.display(),
        config.output_dir.join("run_summary.yaml").display()
    );
    Ok(())
}

fn postprocess_overlay_video(
    config: &Config,
    fps: f64,
    width: usize,
    height: usize,
    mask_artifact: &RawVideoArtifact,
    raw_stream_dir: &Path,
    ffmpeg_bin: &OsStr,
    output_path: PathBuf,
) -> Result<()> {
    if mask_artifact.frames_written == 0 {
        return Ok(());
    }
    let overlay_raw_path = raw_stream_dir.join("overlay_post.raw");
    let overlay_writer = AsyncRawVideoWriter::open(
        &overlay_raw_path,
        fps,
        width,
        height,
        RawPixelFormat::Bgr24,
        Some(mask_artifact.frames_written),
        8,
    )?;
    let mut source_reader = VideoReader::open(&config.video_path)?;
    let mut mask_reader = RawGrayFrameReader::open(mask_artifact)?;

    while let Some((frame_index, frame_bgr)) = source_reader.read_next()? {
        if frame_index % config.frame_step != 0 {
            continue;
        }
        let mask = mask_reader.read_next()?.ok_or_else(|| {
            anyhow!("mask raw stream ended before overlay video reconstruction completed")
        })?;
        overlay_writer.write_bgr(Arc::new(create_overlay(&frame_bgr, &mask)?))?;
    }

    if mask_reader.read_next()?.is_some() {
        bail!("mask raw stream contains extra frames after overlay reconstruction");
    }

    let artifact = overlay_writer.close()?;
    finalize_raw_stream_to_mp4(ffmpeg_bin, &artifact, &output_path)?;
    let _ = fs::remove_file(&artifact.path);
    Ok(())
}

fn postprocess_heatmap_video(
    fps: f64,
    width: usize,
    height: usize,
    delta_norm_artifact: &RawVideoArtifact,
    raw_stream_dir: &Path,
    ffmpeg_bin: &OsStr,
    output_path: PathBuf,
) -> Result<()> {
    if delta_norm_artifact.frames_written == 0 {
        return Ok(());
    }
    let heatmap_raw_path = raw_stream_dir.join("heatmap_post.raw");
    let heatmap_writer = AsyncRawVideoWriter::open(
        &heatmap_raw_path,
        fps,
        width,
        height,
        RawPixelFormat::Bgr24,
        Some(delta_norm_artifact.frames_written),
        8,
    )?;
    let mut delta_norm_reader = RawGrayFrameReader::open(delta_norm_artifact)?;
    while let Some(delta_norm) = delta_norm_reader.read_next()? {
        heatmap_writer.write_bgr(Arc::new(apply_turbo_colormap(&delta_norm)?))?;
    }
    let artifact = heatmap_writer.close()?;
    finalize_raw_stream_to_mp4(ffmpeg_bin, &artifact, &output_path)?;
    let _ = fs::remove_file(&artifact.path);
    Ok(())
}

fn detect_mask_and_delta_norm(
    delta: &Array2<f32>,
    roi_mask: &Array2<u8>,
    params: &MorphologyParams,
) -> Result<DetectionOutputs> {
    let blurred = blur::blur_plane(delta, params.post_blur)?;
    let delta_blur = cvutil::array2_f32_to_mat(&blurred)?;

    let delta_norm = normalize_minmax_to_u8(&delta_blur)?;
    Ok(DetectionOutputs {
        mask: detect_mask_from_delta_norm(&delta_norm, roi_mask, params)?,
        delta_norm,
    })
}

fn detect_mask_from_delta_norm(
    delta_norm: &Array2<u8>,
    roi_mask: &Array2<u8>,
    params: &MorphologyParams,
) -> Result<Array2<u8>> {
    let threshold_value = threshold::choose_threshold(
        delta_norm,
        roi_mask,
        params.threshold_mode,
        params.threshold_offset,
    )?;
    detect_mask_from_delta_norm_threshold(delta_norm, threshold_value, params)
}

fn detect_mask_from_delta_norm_threshold(
    delta_norm: &Array2<u8>,
    threshold_value: f32,
    params: &MorphologyParams,
) -> Result<Array2<u8>> {
    let mut binary = threshold_to_mat(delta_norm, threshold_value)?;
    apply_morphology_in_place(&mut binary, params)?;
    if params.min_area > 0 {
        binary = filter_min_area(&binary, params.min_area)?;
    }
    Ok(cvutil::mat_to_array2_u8(&binary)?)
}

fn threshold_to_mat(delta_norm: &Array2<u8>, threshold_value: f32) -> Result<opencv::core::Mat> {
    let norm_mat = cvutil::array2_u8_to_mat(&delta_norm)?;
    let mut binary = opencv::core::Mat::default();
    imgproc::threshold(
        &norm_mat,
        &mut binary,
        threshold_value as f64,
        255.0,
        imgproc::THRESH_BINARY,
    )?;
    Ok(binary)
}

fn apply_morphology_in_place(
    binary: &mut opencv::core::Mat,
    params: &MorphologyParams,
) -> Result<()> {
    let mut kernel_size = params.morph_kernel + (1 - params.morph_kernel % 2);
    if kernel_size == 0 {
        kernel_size = 1;
    }
    let kernel = imgproc::get_structuring_element_def(
        morph_shape_code(params.morph_shape),
        Size::new(kernel_size as i32, kernel_size as i32),
    )?;

    for _ in 0..params.morph_close_iterations {
        let mut next = opencv::core::Mat::default();
        imgproc::morphology_ex(
            binary,
            &mut next,
            imgproc::MORPH_CLOSE,
            &kernel,
            Point::new(-1, -1),
            1,
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;
        *binary = next;
    }
    for _ in 0..params.morph_open_iterations {
        let mut next = opencv::core::Mat::default();
        imgproc::morphology_ex(
            binary,
            &mut next,
            imgproc::MORPH_OPEN,
            &kernel,
            Point::new(-1, -1),
            1,
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;
        *binary = next;
    }
    Ok(())
}

fn filter_min_area_array(binary: &Array2<u8>, min_area: usize) -> Result<Array2<u8>> {
    let binary = cvutil::array2_u8_to_mat(binary)?;
    Ok(cvutil::mat_to_array2_u8(&filter_min_area(
        &binary, min_area,
    )?)?)
}

fn normalize_minmax_to_u8(delta: &opencv::core::Mat) -> Result<Array2<u8>> {
    let mut normalized = opencv::core::Mat::default();
    core::normalize(
        delta,
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

fn morph_shape_code(shape: MorphShape) -> i32 {
    match shape {
        MorphShape::Ellipse => imgproc::MORPH_ELLIPSE,
        MorphShape::Rect => imgproc::MORPH_RECT,
        MorphShape::Cross => imgproc::MORPH_CROSS,
    }
}

fn convert_frame_with_backend<B>(
    backend: &B,
    frame_bgr: &Array3<u8>,
    colorspace: ColorSpace,
    download_host: bool,
) -> Result<ConvertedFrame<B::DeviceFrameLab>>
where
    B: ImageBackend,
{
    if matches!(colorspace, ColorSpace::Cielab) {
        let uploaded = backend.upload_frame_bgr(frame_bgr)?;
        let device_lab = backend.convert_bgr_to_lab(&uploaded)?;
        let host = if download_host {
            Some(backend.download_frame_f32(&device_lab)?)
        } else {
            None
        };
        Ok(ConvertedFrame {
            device_lab: Some(device_lab),
            host,
        })
    } else {
        Ok(ConvertedFrame {
            device_lab: None,
            host: Some(
                convert_frame_to_colorspace(frame_bgr, colorspace)?.mapv(|value| value as f32),
            ),
        })
    }
}

fn fetch_reference_converted<B>(
    index: usize,
    reader: Option<&mut VideoReader>,
    cache: &mut IndexMap<usize, Array3<f32>>,
    cache_capacity: usize,
    first_frame_converted: &Array3<f32>,
    colorspace: ColorSpace,
    backend: &B,
) -> Result<Array3<f32>>
where
    B: ImageBackend,
{
    if index <= 1 {
        return Ok(first_frame_converted.clone());
    }
    if let Some(value) = cache.shift_remove(&index) {
        cache.insert(index, value.clone());
        return Ok(value);
    }
    let Some(reader) = reader else {
        return Ok(first_frame_converted.clone());
    };
    let Some(frame) = reader.read_frame_at(index)? else {
        return Ok(first_frame_converted.clone());
    };
    let converted = convert_frame_with_backend(backend, &frame, colorspace, true)?
        .host
        .ok_or_else(|| anyhow!("reference conversion unexpectedly omitted host data"))?;
    cache.insert(index, converted.clone());
    while cache.len() > cache_capacity {
        cache.shift_remove_index(0);
    }
    Ok(converted)
}

#[allow(clippy::too_many_arguments)]
fn write_run_summary(
    config: &Config,
    total_frames: Option<usize>,
    processed: usize,
    fps: f64,
    video_duration: Option<f64>,
    roi_pixels: usize,
    roi_fraction: f64,
    avg_compute_time: f64,
    run_total_time: f64,
    dynamic_factor: Option<f32>,
    dynamic_first_lag: Option<usize>,
    dynamic_last_lag: Option<usize>,
    absolute_index: Option<usize>,
) -> Result<()> {
    let mut summary: IndexMap<String, Value> = IndexMap::new();
    macro_rules! put {
        ($key:literal, $value:expr) => {
            summary.insert($key.to_string(), serde_yaml::to_value($value)?);
        };
    }

    put!(
        "run_timestamp",
        Utc::now().to_rfc3339_opts(SecondsFormat::Micros, true)
    );
    put!(
        "video_path",
        config.video_path.to_string_lossy().to_string()
    );
    put!(
        "output_dir",
        config.output_dir.to_string_lossy().to_string()
    );
    put!("frame_step", config.frame_step);
    put!("total_frames", total_frames);
    put!("processed_frames", processed);
    put!("video_fps", fps);
    put!("video_duration_seconds", video_duration);
    put!("reference_mode", reference_mode_name(&config.ref_mode));
    put!("reference_offset", reference_mode_offset(&config.ref_mode));
    put!("absolute_reference_index", absolute_index);
    put!(
        "ref_running_alpha",
        matches!(config.ref_mode, ReferenceMode::Running)
            .then_some(yaml_f32(config.ref_running_alpha))
    );
    put!(
        "dynamic_calibration_frames",
        matches!(config.ref_mode, ReferenceMode::Dynamic)
            .then_some(config.dynamic_calibration_frames)
    );
    put!(
        "dynamic_target_fraction",
        matches!(config.ref_mode, ReferenceMode::Dynamic)
            .then_some(yaml_f32(config.dynamic_target_fraction))
    );
    put!(
        "dynamic_lag_scale",
        matches!(config.ref_mode, ReferenceMode::Dynamic)
            .then_some(yaml_f32(config.dynamic_lag_scale))
    );
    put!(
        "dynamic_lag_linear",
        matches!(config.ref_mode, ReferenceMode::Dynamic).then_some(config.dynamic_lag_linear)
    );
    put!(
        "dynamic_lag_linear_max",
        (matches!(config.ref_mode, ReferenceMode::Dynamic) && config.dynamic_lag_linear)
            .then_some(config.dynamic_lag_linear_max)
    );
    put!(
        "dynamic_lag_linear_start",
        (matches!(config.ref_mode, ReferenceMode::Dynamic) && config.dynamic_lag_linear)
            .then_some(config.dynamic_lag_linear_start)
    );
    put!("dynamic_reference_factor", dynamic_factor.map(yaml_f32));
    put!("dynamic_reference_start_lag", dynamic_first_lag);
    put!("dynamic_reference_end_lag", dynamic_last_lag);
    put!(
        "annotation_formats",
        (!config.annotation_formats.is_empty()).then_some(&config.annotation_formats)
    );
    put!("annotation_mode", &config.annotation_mode);
    put!(
        "annotation_count",
        (config.annotation_mode == "count").then_some(config.annotation_count)
    );
    put!(
        "annotation_stride",
        (config.annotation_mode == "stride").then_some(config.annotation_stride)
    );
    put!(
        "annotation_segmentation_tolerance",
        (!config.annotation_formats.is_empty())
            .then_some(yaml_f32(config.annotation_segmentation_tolerance))
    );
    put!(
        "annotation_segmentation_max_edge_length",
        (!config.annotation_formats.is_empty())
            .then_some(yaml_f32(config.annotation_segmentation_max_edge_length))
    );
    put!("colorspace", config.colorspace.canonical_name());
    put!(
        "channel_weights",
        config
            .channel_weights
            .iter()
            .copied()
            .map(yaml_f32)
            .collect::<Vec<_>>()
    );
    put!("lock_frames", config.lock_frames);
    put!("roi_margin", yaml_f32(config.roi_margin));
    put!("roi_margin_top", config.roi_margin_top.map(yaml_f32));
    put!("roi_margin_bottom", config.roi_margin_bottom.map(yaml_f32));
    put!("roi_margin_left", config.roi_margin_left.map(yaml_f32));
    put!("roi_margin_right", config.roi_margin_right.map(yaml_f32));
    put!("pre_delta_blur", config.pre_delta_blur.to_string());
    put!("blur", config.post_blur.to_string());
    put!("morph_kernel", config.morph_kernel);
    put!("morph_shape", config.morph_shape.name());
    put!("morph_close_iterations", config.morph_close_iterations);
    put!("morph_open_iterations", config.morph_open_iterations);
    put!("min_area", config.min_area);
    put!("threshold", config.threshold_mode.to_string());
    put!("threshold_offset", yaml_f32(config.threshold_offset));
    put!("darken_only", config.darken_only);
    put!("peak_reference", config.peak_reference);
    put!("camera_stable", config.camera_stable);
    put!(
        "motion_per_frame_threshold",
        config
            .camera_stable
            .then_some(yaml_f32(config.motion_per_frame_threshold))
    );
    put!(
        "cumulative_motion_threshold",
        config
            .camera_stable
            .then_some(yaml_f32(config.cumulative_motion_threshold))
    );
    put!(
        "motion_model",
        config.camera_stable.then_some(config.motion_model.name())
    );
    put!(
        "peak_on_shift",
        config
            .camera_stable
            .then_some(peak_on_shift_name(config.peak_on_shift))
    );
    put!("write_mask_pngs", config.write_mask_pngs);
    put!("write_overlay_pngs", config.write_overlay_pngs);
    put!("write_heatmap_pngs", config.write_heatmap_pngs);
    put!("write_videos", config.write_videos);
    put!("write_mask_video", config.write_mask_video);
    put!("write_overlay_video", config.write_overlay_video);
    put!("write_heatmap_video", config.write_heatmap_video);
    put!("roi_pixel_count", roi_pixels);
    put!("roi_fraction", roi_fraction);
    put!("average_compute_time_seconds", avg_compute_time);
    put!("total_run_time_seconds", run_total_time);

    let summary_path = config.output_dir.join("run_summary.yaml");
    let file = File::create(&summary_path)
        .with_context(|| format!("creating {}", summary_path.display()))?;
    serde_yaml::to_writer(file, &summary)
        .with_context(|| format!("writing {}", summary_path.display()))?;
    Ok(())
}

fn reference_mode_name(mode: &ReferenceMode) -> &'static str {
    match mode {
        ReferenceMode::First => "first",
        ReferenceMode::Running => "running",
        ReferenceMode::Prev(_) => "prev",
        ReferenceMode::Absolute(_) => "absolute",
        ReferenceMode::Dynamic => "dynamic",
    }
}

fn peak_on_shift_name(mode: PeakOnShift) -> &'static str {
    match mode {
        PeakOnShift::Reset => "reset",
        PeakOnShift::Warp => "warp",
    }
}

fn reference_mode_offset(mode: &ReferenceMode) -> Option<usize> {
    match mode {
        ReferenceMode::Prev(value) | ReferenceMode::Absolute(value) => Some(*value),
        _ => None,
    }
}

fn yaml_f32(value: f32) -> f64 {
    ((value as f64) * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::{
        delegated_backend_label, parse_fast_args, reference_binary_candidates, BackendMode,
    };
    use std::ffi::OsString;

    #[test]
    fn delegated_backend_is_reported() {
        assert_eq!(delegated_backend_label(), "delegated-cpu");
    }

    #[test]
    fn reference_binary_search_order_is_seeded() {
        let candidates = reference_binary_candidates();
        assert!(candidates.len() >= 3);
    }

    #[test]
    fn backend_flag_is_removed_before_forwarding() {
        let args = vec![
            OsString::from("fast-vrifa"),
            OsString::from("--backend"),
            OsString::from("wgpu"),
            OsString::from("--video-path"),
            OsString::from("data/input_1.mp4"),
        ];
        let (options, stripped) = parse_fast_args(args).unwrap();
        assert_eq!(options.backend, BackendMode::Wgpu);
        assert_eq!(stripped.len(), 3);
        assert_eq!(stripped[1], OsString::from("--video-path"));
    }

    #[test]
    fn ffmpeg_postprocess_flag_is_removed_before_forwarding() {
        let args = vec![
            OsString::from("fast-vrifa"),
            OsString::from("--backend=cuda"),
            OsString::from("--ffmpeg-postprocess"),
            OsString::from("--video-path"),
            OsString::from("data/input_2.mp4"),
        ];
        let (options, stripped) = parse_fast_args(args).unwrap();
        assert_eq!(options.backend, BackendMode::Cuda);
        assert!(options.ffmpeg_postprocess);
        assert_eq!(stripped.len(), 3);
        assert_eq!(stripped[1], OsString::from("--video-path"));
    }

    #[test]
    fn mask_only_and_bbox_only_flags_are_removed_before_forwarding() {
        let args = vec![
            OsString::from("fast-vrifa"),
            OsString::from("--mask-only"),
            OsString::from("--coco-bbox-only=true"),
            OsString::from("--video-path"),
            OsString::from("data/input_2.mp4"),
        ];
        let (options, stripped) = parse_fast_args(args).unwrap();
        assert!(options.mask_only);
        assert!(options.coco_bbox_only);
        assert_eq!(stripped.len(), 3);
        assert_eq!(stripped[1], OsString::from("--video-path"));
    }

    #[test]
    fn backend_parser_accepts_cpu_and_cuda() {
        assert_eq!(BackendMode::parse("cpu").unwrap(), BackendMode::Cpu);
        assert_eq!(BackendMode::parse("cuda").unwrap(), BackendMode::Cuda);
    }
}
