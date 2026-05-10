use anyhow::{anyhow, bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use clap::{ArgAction, Parser};
use indexmap::IndexMap;
use ndarray::{Array2, Array3};
use ndarray_npy::write_npy;
use serde_yaml::Value;
use std::collections::{BTreeSet, VecDeque};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use vrifa_stable_annotations::AnnotationFrame;
use vrifa_stable_core::colorspace::{convert_frame_to_colorspace, ColorSpace};
use vrifa_stable_core::contours::extract_bounding_boxes;
use vrifa_stable_core::delta::blur_plane;
use vrifa_stable_core::lock::{apply_locking, LockState};
use vrifa_stable_core::morphology::MorphShape;
use vrifa_stable_core::motion::estimate_translation;
use vrifa_stable_core::overlay::create_overlay;
use vrifa_stable_core::peak::update_peak_brightness_plane;
use vrifa_stable_core::reference::{select_dynamic_reference_index, DynamicReferenceParams};
use vrifa_stable_core::registration::{fit_affine_warp, strong_gradient_mask, MotionModel};
use vrifa_stable_core::roi::{build_roi_mask, resolve_roi_margins};
use vrifa_stable_core::warp::{apply_warp, apply_warp_plane, AffineWarp};
use vrifa_stable_core::{detect_front, detect_front_debug, DetectFrontDebug, DetectFrontParams};
use vrifa_stable_io::{AsyncPngWriter, AsyncVideoWriter, VideoReader};

#[derive(Parser, Debug)]
#[command(
    name = "vrifa-stable",
    about = "VARTM Resin Infusion Flow-Front Assessment Algorithm"
)]
pub struct CliArgs {
    #[arg(long, default_value = "private_assets/input_video.mp4")]
    video_path: PathBuf,
    #[arg(long, default_value = "flow_front_outputs")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    frame_step: usize,
    #[arg(long, default_value_t = 0.15)]
    roi_margin: f32,
    #[arg(long)]
    roi_margin_top: Option<f32>,
    #[arg(long)]
    roi_margin_bottom: Option<f32>,
    #[arg(long)]
    roi_margin_left: Option<f32>,
    #[arg(long)]
    roi_margin_right: Option<f32>,
    #[arg(long, default_value_t = 9)]
    blur_kernel: usize,
    #[arg(long, default_value_t = 0)]
    pre_delta_blur_kernel: usize,
    #[arg(long, action = ArgAction::SetTrue)]
    skip_blur: bool,
    #[arg(long, default_value_t = 13)]
    morph_kernel: usize,
    #[arg(long, default_value = "ellipse")]
    morph_shape: String,
    #[arg(long, default_value_t = 1)]
    morph_close_iterations: usize,
    #[arg(long, default_value_t = 1)]
    morph_open_iterations: usize,
    #[arg(long, default_value_t = 400)]
    min_area: usize,
    #[arg(long)]
    contrast_threshold: Option<f32>,
    #[arg(long)]
    contrast_percentile: Option<f32>,
    #[arg(long, default_value_t = -30.0)]
    threshold_offset: f32,
    #[arg(long, default_value_t = true, action = ArgAction::SetTrue)]
    darken_only: bool,
    #[arg(long = "no-darken-only", action = ArgAction::SetTrue)]
    no_darken_only: bool,
    #[arg(long, default_value_t = true, action = ArgAction::SetTrue)]
    peak_reference: bool,
    #[arg(long = "no-peak-reference", action = ArgAction::SetTrue)]
    no_peak_reference: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    camera_stable: bool,
    #[arg(long, default_value_t = 1.5)]
    motion_per_frame_threshold: f32,
    #[arg(long, default_value_t = 3.0)]
    cumulative_motion_threshold: f32,
    #[arg(long, default_value = "affine")]
    motion_model: String,
    #[arg(long, default_value = "reset")]
    peak_on_shift: String,
    #[arg(long, action = ArgAction::SetTrue)]
    write_videos: bool,
    #[arg(long, default_value = "false")]
    write_mask_pngs: String,
    #[arg(long, default_value = "false")]
    write_overlay_pngs: String,
    #[arg(long, default_value = "false")]
    write_heatmap_pngs: String,
    #[arg(long)]
    write_mask_video: Option<String>,
    #[arg(long)]
    write_overlay_video: Option<String>,
    #[arg(long)]
    write_heatmap_video: Option<String>,
    #[arg(long, default_value_t = 3)]
    lock_frames: usize,
    #[arg(long, default_value = "CIELAB")]
    colorspace: String,
    #[arg(long)]
    channel_weights: Option<String>,
    #[arg(long, num_args = 1..=2, default_values_t = vec!["first".to_string()])]
    ref_mode: Vec<String>,
    #[arg(long, default_value_t = 0.05)]
    ref_running_alpha: f32,
    #[arg(long, default_value_t = 10)]
    dynamic_calibration_frames: usize,
    #[arg(long, default_value_t = 0.2)]
    dynamic_target_fraction: f32,
    #[arg(long, default_value_t = 32)]
    dynamic_ref_cache_size: usize,
    #[arg(long, default_value = "coco")]
    annotation_formats: String,
    #[arg(long, default_value = "all")]
    annotation_mode: String,
    #[arg(long, default_value_t = 0)]
    annotation_count: usize,
    #[arg(long, default_value_t = 1)]
    annotation_stride: usize,
    #[arg(long, default_value_t = 0.0)]
    annotation_segmentation_tolerance: f32,
    #[arg(long, default_value_t = 0.0)]
    annotation_segmentation_max_edge_length: f32,
    #[arg(long, default_value_t = 1.0)]
    dynamic_lag_scale: f32,
    #[arg(long, action = ArgAction::SetTrue)]
    dynamic_lag_linear: bool,
    #[arg(long, default_value_t = 60)]
    dynamic_lag_linear_max: usize,
    #[arg(long, default_value_t = 0)]
    dynamic_lag_linear_start: usize,
    #[arg(long)]
    dynamic_lag_log: Option<PathBuf>,
    #[arg(long)]
    debug_dump_frames: Option<String>,
    #[arg(long)]
    debug_dump_dir: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub enum ReferenceMode {
    First,
    Running,
    Prev(usize),
    Absolute(usize),
    Dynamic,
}

impl ReferenceMode {
    fn name(&self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Running => "running",
            Self::Prev(_) => "prev",
            Self::Absolute(_) => "absolute",
            Self::Dynamic => "dynamic",
        }
    }

    fn offset(&self) -> Option<usize> {
        match self {
            Self::Prev(value) | Self::Absolute(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeakOnShift {
    Reset,
    Warp,
}

impl PeakOnShift {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "reset" => Ok(Self::Reset),
            "warp" => Ok(Self::Warp),
            other => bail!("--peak-on-shift expects reset or warp, got '{other}'"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Reset => "reset",
            Self::Warp => "warp",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub video_path: PathBuf,
    pub output_dir: PathBuf,
    pub frame_step: usize,
    pub roi_margin: f32,
    pub roi_margin_top: Option<f32>,
    pub roi_margin_bottom: Option<f32>,
    pub roi_margin_left: Option<f32>,
    pub roi_margin_right: Option<f32>,
    pub pre_delta_blur_kernel: usize,
    pub blur_kernel: usize,
    pub skip_blur: bool,
    pub morph_kernel: usize,
    pub morph_shape: MorphShape,
    pub morph_close_iterations: usize,
    pub morph_open_iterations: usize,
    pub min_area: usize,
    pub contrast_threshold: Option<f32>,
    pub contrast_percentile: Option<f32>,
    pub threshold_offset: f32,
    pub darken_only: bool,
    pub peak_reference: bool,
    pub camera_stable: bool,
    pub motion_per_frame_threshold: f32,
    pub cumulative_motion_threshold: f32,
    pub motion_model: MotionModel,
    pub peak_on_shift: PeakOnShift,
    pub write_videos: bool,
    pub write_mask_pngs: bool,
    pub write_overlay_pngs: bool,
    pub write_heatmap_pngs: bool,
    pub write_mask_video: bool,
    pub write_overlay_video: bool,
    pub write_heatmap_video: bool,
    pub lock_frames: usize,
    pub colorspace: ColorSpace,
    pub channel_weights: Vec<f32>,
    pub ref_mode: ReferenceMode,
    pub ref_running_alpha: f32,
    pub dynamic_calibration_frames: usize,
    pub dynamic_target_fraction: f32,
    pub dynamic_ref_cache_size: usize,
    pub annotation_formats: Vec<String>,
    pub annotation_mode: String,
    pub annotation_count: usize,
    pub annotation_stride: usize,
    pub annotation_segmentation_tolerance: f32,
    pub annotation_segmentation_max_edge_length: f32,
    pub dynamic_lag_scale: f32,
    pub dynamic_lag_linear: bool,
    pub dynamic_lag_linear_max: usize,
    pub dynamic_lag_linear_start: usize,
    pub dynamic_lag_log: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        let colorspace = ColorSpace::Cielab;
        Self {
            video_path: PathBuf::from("private_assets/input_video.mp4"),
            output_dir: PathBuf::from("flow_front_outputs"),
            frame_step: 1,
            roi_margin: 0.15,
            roi_margin_top: None,
            roi_margin_bottom: None,
            roi_margin_left: None,
            roi_margin_right: None,
            pre_delta_blur_kernel: 0,
            blur_kernel: 9,
            skip_blur: false,
            morph_kernel: 13,
            morph_shape: MorphShape::Ellipse,
            morph_close_iterations: 1,
            morph_open_iterations: 1,
            min_area: 400,
            contrast_threshold: None,
            contrast_percentile: None,
            threshold_offset: -30.0,
            darken_only: true,
            peak_reference: true,
            camera_stable: false,
            motion_per_frame_threshold: 1.5,
            cumulative_motion_threshold: 3.0,
            motion_model: MotionModel::Affine,
            peak_on_shift: PeakOnShift::Reset,
            write_videos: false,
            write_mask_pngs: false,
            write_overlay_pngs: false,
            write_heatmap_pngs: false,
            write_mask_video: false,
            write_overlay_video: true,
            write_heatmap_video: false,
            lock_frames: 3,
            colorspace,
            channel_weights: vec![1.0; colorspace.channel_count()],
            ref_mode: ReferenceMode::First,
            ref_running_alpha: 0.05,
            dynamic_calibration_frames: 10,
            dynamic_target_fraction: 0.2,
            dynamic_ref_cache_size: 32,
            annotation_formats: vec!["coco".to_string()],
            annotation_mode: "all".to_string(),
            annotation_count: 0,
            annotation_stride: 1,
            annotation_segmentation_tolerance: 0.0,
            annotation_segmentation_max_edge_length: 0.0,
            dynamic_lag_scale: 1.0,
            dynamic_lag_linear: false,
            dynamic_lag_linear_max: 60,
            dynamic_lag_linear_start: 0,
            dynamic_lag_log: None,
        }
    }
}

impl TryFrom<CliArgs> for Config {
    type Error = anyhow::Error;

    fn try_from(args: CliArgs) -> Result<Self> {
        let colorspace = ColorSpace::parse(&args.colorspace)?;
        let mut write_mask_video =
            parse_bool_optional(args.write_mask_video.as_deref(), "--write-mask-video")?;
        let mut write_overlay_video =
            parse_bool_optional(args.write_overlay_video.as_deref(), "--write-overlay-video")?;
        let mut write_heatmap_video =
            parse_bool_optional(args.write_heatmap_video.as_deref(), "--write-heatmap-video")?;
        if args.write_videos {
            write_mask_video.get_or_insert(true);
            write_overlay_video.get_or_insert(true);
            write_heatmap_video.get_or_insert(true);
        }

        let annotation_formats = parse_annotation_formats(&args.annotation_formats)?;
        if !matches!(args.annotation_mode.as_str(), "all" | "count" | "stride") {
            bail!("--annotation-mode expects all, count, or stride");
        }
        if !annotation_formats.is_empty()
            && args.annotation_mode == "count"
            && args.annotation_count == 0
        {
            bail!("--annotation-count must be > 0 when --annotation-mode count is selected");
        }
        if !annotation_formats.is_empty()
            && args.annotation_mode == "stride"
            && args.annotation_stride < 1
        {
            bail!("--annotation-stride must be >= 1 when --annotation-mode stride is selected");
        }
        if args.dynamic_lag_linear && args.dynamic_lag_linear_start > args.dynamic_lag_linear_max {
            bail!("--dynamic-lag-linear-start cannot exceed --dynamic-lag-linear-max");
        }
        if args.frame_step == 0 {
            bail!("--frame-step must be >= 1");
        }

        let channel_weights =
            parse_channel_weights(args.channel_weights.as_deref(), colorspace.channel_count())?;
        let ref_mode = parse_ref_mode(&args.ref_mode)?;
        let motion_model = MotionModel::parse(&args.motion_model)?;
        let peak_on_shift = PeakOnShift::parse(&args.peak_on_shift)?;
        Ok(Self {
            video_path: args.video_path,
            output_dir: args.output_dir,
            frame_step: args.frame_step,
            roi_margin: args.roi_margin,
            roi_margin_top: args.roi_margin_top,
            roi_margin_bottom: args.roi_margin_bottom,
            roi_margin_left: args.roi_margin_left,
            roi_margin_right: args.roi_margin_right,
            pre_delta_blur_kernel: args.pre_delta_blur_kernel,
            blur_kernel: args.blur_kernel,
            skip_blur: args.skip_blur,
            morph_kernel: args.morph_kernel,
            morph_shape: MorphShape::parse(&args.morph_shape),
            morph_close_iterations: args.morph_close_iterations,
            morph_open_iterations: args.morph_open_iterations,
            min_area: args.min_area,
            contrast_threshold: args.contrast_threshold,
            contrast_percentile: args.contrast_percentile,
            threshold_offset: args.threshold_offset,
            darken_only: args.darken_only && !args.no_darken_only,
            peak_reference: args.peak_reference && !args.no_peak_reference,
            camera_stable: args.camera_stable,
            motion_per_frame_threshold: args.motion_per_frame_threshold.max(0.0),
            cumulative_motion_threshold: args.cumulative_motion_threshold.max(0.0),
            motion_model,
            peak_on_shift,
            write_videos: args.write_videos,
            write_mask_pngs: parse_bool(&args.write_mask_pngs, "--write-mask-pngs")?,
            write_overlay_pngs: parse_bool(&args.write_overlay_pngs, "--write-overlay-pngs")?,
            write_heatmap_pngs: parse_bool(&args.write_heatmap_pngs, "--write-heatmap-pngs")?,
            write_mask_video: write_mask_video.unwrap_or(false),
            write_overlay_video: write_overlay_video.unwrap_or(true),
            write_heatmap_video: write_heatmap_video.unwrap_or(false),
            lock_frames: args.lock_frames,
            colorspace,
            channel_weights,
            ref_mode,
            ref_running_alpha: args.ref_running_alpha.clamp(0.0, 1.0),
            dynamic_calibration_frames: args.dynamic_calibration_frames.max(1),
            dynamic_target_fraction: args.dynamic_target_fraction.clamp(0.0, 1.0),
            dynamic_ref_cache_size: args.dynamic_ref_cache_size.max(1),
            annotation_formats,
            annotation_mode: args.annotation_mode,
            annotation_count: args.annotation_count,
            annotation_stride: args.annotation_stride,
            annotation_segmentation_tolerance: args.annotation_segmentation_tolerance.max(0.0),
            annotation_segmentation_max_edge_length: args
                .annotation_segmentation_max_edge_length
                .max(0.0),
            dynamic_lag_scale: args.dynamic_lag_scale.max(0.0),
            dynamic_lag_linear: args.dynamic_lag_linear,
            dynamic_lag_linear_max: args.dynamic_lag_linear_max,
            dynamic_lag_linear_start: args.dynamic_lag_linear_start,
            dynamic_lag_log: args.dynamic_lag_log,
        })
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
struct CameraStableOutput {
    aligned_reference: Array3<f32>,
    peak_map: Option<Array2<f32>>,
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
    pre_delta_blur_kernel: usize,
) -> Result<Array2<f32>> {
    let frame_l = frame_converted.slice(ndarray::s![.., .., 0]).to_owned();
    if pre_delta_blur_kernel > 1 {
        blur_plane(&frame_l, pre_delta_blur_kernel).map_err(Into::into)
    } else {
        Ok(frame_l)
    }
}

pub fn run() -> Result<()> {
    let args = CliArgs::parse();
    let debug_dump_frames = parse_frame_list(args.debug_dump_frames.as_deref())?;
    let debug_dump_dir = args.debug_dump_dir.clone();
    let config = Config::try_from(args)?;
    if let Some(frames) = debug_dump_frames {
        let output_dir = debug_dump_dir.unwrap_or_else(|| config.output_dir.clone());
        return dump_debug_stages(config, &frames, &output_dir);
    }
    run_config(config)
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

fn prepare_camera_stable(
    frame_index: usize,
    frame_converted: &Array3<f32>,
    reference_for_frame: &Array3<f32>,
    roi_mask: &Array2<u8>,
    reference_token: usize,
    peak_state: Option<&Array2<f32>>,
    config: &Config,
    state: &mut CameraStableState,
) -> Result<CameraStableOutput> {
    if state.reference_token != Some(reference_token) {
        state.reset(reference_token);
    }

    let mut peak_base = peak_state.cloned();
    let mut fit_applied = false;
    let mut fit_model = None;
    let mut fit_score = None;

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
                        if config.peak_reference {
                            peak_base = match config.peak_on_shift {
                                PeakOnShift::Reset => None,
                                PeakOnShift::Warp => peak_state
                                    .map(|peak| apply_warp_plane(peak, &state.cached_warp))
                                    .transpose()?,
                            };
                        }
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

    let aligned_reference = if config.camera_stable && state.warp_active {
        apply_warp(reference_for_frame, &state.cached_warp)?
    } else {
        reference_for_frame.clone()
    };
    let peak_map = if config.peak_reference {
        let peak_frame = peak_frame_plane(frame_converted, config.pre_delta_blur_kernel)?;
        Some(update_peak_brightness_plane(
            &peak_frame,
            peak_base.as_ref(),
        )?)
    } else {
        None
    };
    state.prev_motion_frame = Some(frame_converted.clone());

    Ok(CameraStableOutput {
        aligned_reference,
        peak_map,
    })
}

pub fn run_config(config: Config) -> Result<()> {
    fs::create_dir_all(&config.output_dir)?;
    let mut reader = VideoReader::open(&config.video_path)?;
    let metadata = reader.metadata();
    let Some((_, first_frame_bgr)) = reader.read_next()? else {
        bail!("failed to read reference frame");
    };
    let first_frame_converted = convert_to_f32(&first_frame_bgr, config.colorspace)?;

    let absolute_index = match config.ref_mode {
        ReferenceMode::Absolute(index) => Some(index),
        _ => None,
    };
    let absolute_reference = if let Some(index) = absolute_index {
        if let Some(total) = metadata.total_frames {
            if index >= total {
                bail!("requested absolute frame index exceeds video length");
            }
        }
        let frame = reader
            .read_frame_at_zero_based(index)
            .with_context(|| format!("reading absolute reference frame {index}"))?
            .ok_or_else(|| anyhow!("unable to read absolute reference frame {index}"))?;
        convert_to_f32(&frame, config.colorspace)?
    } else {
        first_frame_converted.clone()
    };

    let mut running_reference = first_frame_converted.clone();
    let mut prev_buffer: Option<VecDeque<Array3<f32>>> = match config.ref_mode {
        ReferenceMode::Prev(offset) => Some(VecDeque::with_capacity(offset)),
        _ => None,
    };
    let mut dynamic_reader = if matches!(config.ref_mode, ReferenceMode::Dynamic) {
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

    let roi_margins = resolve_roi_margins(
        config.roi_margin,
        config.roi_margin_top,
        config.roi_margin_bottom,
        config.roi_margin_left,
        config.roi_margin_right,
    );
    let roi_mask = build_roi_mask(
        (first_frame_converted.dim().0, first_frame_converted.dim().1),
        roi_margins,
    );
    let roi_pixels = roi_mask.iter().filter(|value| **value > 0).count();
    let mut lock_state = (config.lock_frames > 0).then(|| LockState::new(roi_mask.dim()));
    let mut peak_brightness_map = if config.peak_reference {
        Some(peak_frame_plane(
            &first_frame_converted,
            config.pre_delta_blur_kernel,
        )?)
    } else {
        None
    };
    let mut camera_stable_state = CameraStableState::new();

    let mask_dir = config.output_dir.join("masks");
    let overlay_dir = config.output_dir.join("overlays");
    let heatmap_dir = config.output_dir.join("heatmap");
    if config.write_mask_pngs {
        fs::create_dir_all(&mask_dir)?;
    }
    if config.write_overlay_pngs {
        fs::create_dir_all(&overlay_dir)?;
    }
    if config.write_heatmap_pngs {
        fs::create_dir_all(&heatmap_dir)?;
    }
    let mut mask_png_writer = config
        .write_mask_pngs
        .then(|| AsyncPngWriter::open(false))
        .transpose()?;
    let mut overlay_png_writer = config
        .write_overlay_pngs
        .then(|| AsyncPngWriter::open(true))
        .transpose()?;
    let mut heatmap_png_writer = config
        .write_heatmap_pngs
        .then(|| AsyncPngWriter::open(true))
        .transpose()?;

    let video_dir = config.output_dir.join("videos");
    let mut mask_writer = None;
    let mut overlay_writer = None;
    let mut heat_writer = None;
    if config.write_mask_video || config.write_overlay_video || config.write_heatmap_video {
        fs::create_dir_all(&video_dir)?;
        if config.write_mask_video {
            mask_writer = Some(AsyncVideoWriter::open(
                video_dir.join("mask.mp4"),
                metadata.fps,
                metadata.width,
                metadata.height,
                false,
            )?);
        }
        if config.write_overlay_video {
            overlay_writer = Some(AsyncVideoWriter::open(
                video_dir.join("overlay.mp4"),
                metadata.fps,
                metadata.width,
                metadata.height,
                true,
            )?);
        }
        if config.write_heatmap_video {
            heat_writer = Some(AsyncVideoWriter::open(
                video_dir.join("heatmap.mp4"),
                metadata.fps,
                metadata.width,
                metadata.height,
                true,
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
    let coco_images_dir = config
        .output_dir
        .join("formatCOCO")
        .join("images")
        .join("default");
    let mut coco_image_writer = if stream_coco_images {
        fs::create_dir_all(&coco_images_dir)?;
        Some(AsyncPngWriter::open_with_workers(true, 8, 64)?)
    } else {
        None
    };

    reader.seek_zero()?;
    let mut processed = 0usize;
    let mut processing_time_accum = 0.0f64;
    let run_start = Instant::now();
    let mut processed_records = Vec::new();

    while let Some((frame_index, frame_bgr)) = reader.read_next()? {
        let frame_converted = convert_to_f32(&frame_bgr, config.colorspace)?;
        let mut reference_frame_index = 1usize;
        let reference_for_frame = match config.ref_mode {
            ReferenceMode::First => first_frame_converted.clone(),
            ReferenceMode::Absolute(_) => {
                reference_frame_index = absolute_index.filter(|index| *index > 0).unwrap_or(1);
                absolute_reference.clone()
            }
            ReferenceMode::Running => running_reference.clone(),
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
                )?
            }
        };

        if frame_index % config.frame_step == 0 {
            let compute_start = Instant::now();
            let camera_stable = prepare_camera_stable(
                frame_index,
                &frame_converted,
                &reference_for_frame,
                &roi_mask,
                reference_token_for_frame(&config.ref_mode, frame_index, reference_frame_index),
                peak_brightness_map.as_ref(),
                &config,
                &mut camera_stable_state,
            )?;
            peak_brightness_map = camera_stable.peak_map;
            let params = DetectFrontParams {
                pre_delta_blur_kernel: config.pre_delta_blur_kernel,
                blur_kernel: config.blur_kernel,
                morph_kernel: config.morph_kernel,
                min_area: config.min_area,
                manual_threshold: config.contrast_threshold,
                percentile_threshold: config.contrast_percentile,
                threshold_offset: config.threshold_offset,
                channel_weights: config.channel_weights.clone(),
                blur_enabled: !config.skip_blur,
                morph_shape: config.morph_shape,
                morph_close_iterations: config.morph_close_iterations,
                morph_open_iterations: config.morph_open_iterations,
                darken_only: config.darken_only,
            };
            let (mask_raw, heatmap) = detect_front(
                &frame_converted,
                &camera_stable.aligned_reference,
                &roi_mask,
                &params,
                peak_brightness_map.as_ref(),
            )?;
            let mask = Arc::new(apply_locking(
                &mask_raw,
                config.lock_frames,
                lock_state.as_mut(),
            )?);
            camera_stable_state.prev_registration_mask = Some((*mask).clone());
            let overlay = Arc::new(create_overlay(&frame_bgr, &mask)?);
            let heatmap = Arc::new(heatmap);

            if !config.annotation_formats.is_empty() {
                let boxes = extract_bounding_boxes(
                    &mask,
                    config.annotation_segmentation_tolerance,
                    config.annotation_segmentation_max_edge_length,
                )?;
                let frame_bgr = if let Some(writer) = coco_image_writer.as_mut() {
                    writer.write_bgr(
                        coco_images_dir.join(format!("frame_{frame_index:06}.png")),
                        frame_bgr,
                    )?;
                    None
                } else {
                    Some(frame_bgr)
                };
                processed_records.push(AnnotationFrame {
                    frame_index,
                    frame_bgr,
                    boxes,
                });
            }

            let basename = format!("frame_{frame_index:06}.png");
            if let Some(writer) = mask_png_writer.as_mut() {
                writer.write_gray(mask_dir.join(&basename), (*mask).clone())?;
            }
            if let Some(writer) = overlay_png_writer.as_mut() {
                writer.write_bgr(overlay_dir.join(&basename), (*overlay).clone())?;
            }
            if let Some(writer) = heatmap_png_writer.as_mut() {
                writer.write_bgr(heatmap_dir.join(&basename), (*heatmap).clone())?;
            }
            if let Some(writer) = mask_writer.as_mut() {
                writer.write_gray(mask.clone())?;
            }
            if let Some(writer) = overlay_writer.as_mut() {
                writer.write_bgr(overlay.clone())?;
            }
            if let Some(writer) = heat_writer.as_mut() {
                writer.write_bgr(heatmap.clone())?;
            }

            if matches!(config.ref_mode, ReferenceMode::Dynamic) {
                let lag = frame_index.saturating_sub(reference_frame_index);
                dynamic_first_lag.get_or_insert(lag);
                dynamic_last_lag = Some(lag);
                dynamic_lag_log.push((frame_index, lag));
                let mask_area = mask.iter().filter(|value| **value > 0).count();
                if dynamic_factor.is_none()
                    && metadata.fps > 0.0
                    && frame_index > 1
                    && mask_area > 0
                {
                    let time_seconds = (frame_index - 1) as f32 / metadata.fps as f32;
                    dynamic_measurements.push((time_seconds, mask_area as f32));
                    if dynamic_measurements.len() >= config.dynamic_calibration_frames {
                        dynamic_factor = vrifa_stable_core::reference::compute_dynamic_factor(
                            &dynamic_measurements,
                        );
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
                buffer.push_back(frame_converted.clone());
            }
        }
        if matches!(config.ref_mode, ReferenceMode::Running) {
            let alpha = config.ref_running_alpha;
            for (running, current) in running_reference.iter_mut().zip(frame_converted.iter()) {
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

    if let Some(writer) = mask_writer.take() {
        writer.close()?;
    }
    if let Some(writer) = overlay_writer.take() {
        writer.close()?;
    }
    if let Some(writer) = heat_writer.take() {
        writer.close()?;
    }
    if let Some(writer) = coco_image_writer.take() {
        writer.close()?;
    }
    if let Some(writer) = mask_png_writer.take() {
        writer.close()?;
    }
    if let Some(writer) = overlay_png_writer.take() {
        writer.close()?;
    }
    if let Some(writer) = heatmap_png_writer.take() {
        writer.close()?;
    }

    if !config.annotation_formats.is_empty() {
        let selection = vrifa_stable_core::sampling::choose_annotation_indices(
            processed_records.len(),
            &config.annotation_mode,
            config.annotation_count,
            config.annotation_stride,
        );
        vrifa_stable_annotations::export_annotation_outputs(
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

#[derive(Debug)]
struct StageDumpFrame {
    frame_converted: Array3<f32>,
    reference_for_frame: Array3<f32>,
    aligned_reference: Array3<f32>,
    detect: DetectFrontDebug,
    mask: ndarray::Array2<u8>,
    overlay: Array3<u8>,
}

fn dump_debug_stages(config: Config, frames: &[usize], output_dir: &Path) -> Result<()> {
    let targets: BTreeSet<usize> = frames.iter().copied().collect();
    if targets.is_empty() {
        bail!("--debug-dump-frames requires at least one positive frame index");
    }
    fs::create_dir_all(output_dir)?;

    let mut reader = VideoReader::open(&config.video_path)?;
    let metadata = reader.metadata();
    let Some((_, first_frame_bgr)) = reader.read_next()? else {
        bail!("failed to read reference frame");
    };
    let first_frame_converted = convert_to_f32(&first_frame_bgr, config.colorspace)?;

    let absolute_index = match config.ref_mode {
        ReferenceMode::Absolute(index) => Some(index),
        _ => None,
    };
    let absolute_reference = if let Some(index) = absolute_index {
        if let Some(total) = metadata.total_frames {
            if index >= total {
                bail!("requested absolute frame index exceeds video length");
            }
        }
        let frame = reader
            .read_frame_at_zero_based(index)
            .with_context(|| format!("reading absolute reference frame {index}"))?
            .ok_or_else(|| anyhow!("unable to read absolute reference frame {index}"))?;
        convert_to_f32(&frame, config.colorspace)?
    } else {
        first_frame_converted.clone()
    };

    let mut running_reference = first_frame_converted.clone();
    let mut prev_buffer: Option<VecDeque<Array3<f32>>> = match config.ref_mode {
        ReferenceMode::Prev(offset) => Some(VecDeque::with_capacity(offset)),
        _ => None,
    };
    let mut dynamic_reader = if matches!(config.ref_mode, ReferenceMode::Dynamic) {
        Some(VideoReader::open(&config.video_path)?)
    } else {
        None
    };
    let mut dynamic_cache: IndexMap<usize, Array3<f32>> = IndexMap::new();
    let mut dynamic_measurements: Vec<(f32, f32)> = Vec::new();
    let mut dynamic_factor: Option<f32> = None;

    let roi_margins = resolve_roi_margins(
        config.roi_margin,
        config.roi_margin_top,
        config.roi_margin_bottom,
        config.roi_margin_left,
        config.roi_margin_right,
    );
    let roi_mask = build_roi_mask(
        (first_frame_converted.dim().0, first_frame_converted.dim().1),
        roi_margins,
    );
    let roi_pixels = roi_mask.iter().filter(|value| **value > 0).count();
    let mut lock_state = (config.lock_frames > 0).then(|| LockState::new(roi_mask.dim()));
    let mut peak_brightness_map = if config.peak_reference {
        Some(peak_frame_plane(
            &first_frame_converted,
            config.pre_delta_blur_kernel,
        )?)
    } else {
        None
    };
    let mut camera_stable_state = CameraStableState::new();

    let params = DetectFrontParams {
        pre_delta_blur_kernel: config.pre_delta_blur_kernel,
        blur_kernel: config.blur_kernel,
        morph_kernel: config.morph_kernel,
        min_area: config.min_area,
        manual_threshold: config.contrast_threshold,
        percentile_threshold: config.contrast_percentile,
        threshold_offset: config.threshold_offset,
        channel_weights: config.channel_weights.clone(),
        blur_enabled: !config.skip_blur,
        morph_shape: config.morph_shape,
        morph_close_iterations: config.morph_close_iterations,
        morph_open_iterations: config.morph_open_iterations,
        darken_only: config.darken_only,
    };

    reader.seek_zero()?;
    let mut dumped = BTreeSet::new();

    while let Some((frame_index, frame_bgr)) = reader.read_next()? {
        let frame_converted = convert_to_f32(&frame_bgr, config.colorspace)?;
        let mut reference_frame_index = 1usize;
        let reference_for_frame = match config.ref_mode {
            ReferenceMode::First => first_frame_converted.clone(),
            ReferenceMode::Absolute(_) => {
                reference_frame_index = absolute_index.filter(|index| *index > 0).unwrap_or(1);
                absolute_reference.clone()
            }
            ReferenceMode::Running => running_reference.clone(),
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
                let dynamic_params = DynamicReferenceParams {
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
                    &dynamic_params,
                );
                reference_frame_index = ref_index;
                fetch_reference_converted(
                    ref_index,
                    dynamic_reader.as_mut(),
                    &mut dynamic_cache,
                    config.dynamic_ref_cache_size,
                    &first_frame_converted,
                    config.colorspace,
                )?
            }
        };

        if frame_index % config.frame_step == 0 {
            let camera_stable = prepare_camera_stable(
                frame_index,
                &frame_converted,
                &reference_for_frame,
                &roi_mask,
                reference_token_for_frame(&config.ref_mode, frame_index, reference_frame_index),
                peak_brightness_map.as_ref(),
                &config,
                &mut camera_stable_state,
            )?;
            peak_brightness_map = camera_stable.peak_map;

            let detect = detect_front_debug(
                &frame_converted,
                &camera_stable.aligned_reference,
                &roi_mask,
                &params,
                peak_brightness_map.as_ref(),
            )?;
            let mask = apply_locking(&detect.mask, config.lock_frames, lock_state.as_mut())?;
            camera_stable_state.prev_registration_mask = Some(mask.clone());

            if targets.contains(&frame_index) {
                let overlay = create_overlay(&frame_bgr, &mask)?;
                write_stage_dump_frame(
                    output_dir,
                    frame_index,
                    &StageDumpFrame {
                        frame_converted: frame_converted.clone(),
                        reference_for_frame: reference_for_frame.clone(),
                        aligned_reference: camera_stable.aligned_reference.clone(),
                        detect: detect.clone(),
                        mask,
                        overlay,
                    },
                )?;
                dumped.insert(frame_index);
                if dumped.len() == targets.len() {
                    break;
                }
            }

            if matches!(config.ref_mode, ReferenceMode::Dynamic) {
                let mask_area = detect.mask.iter().filter(|value| **value > 0).count();
                if dynamic_factor.is_none()
                    && metadata.fps > 0.0
                    && frame_index > 1
                    && mask_area > 0
                {
                    let time_seconds = (frame_index - 1) as f32 / metadata.fps as f32;
                    dynamic_measurements.push((time_seconds, mask_area as f32));
                    if dynamic_measurements.len() >= config.dynamic_calibration_frames {
                        dynamic_factor = vrifa_stable_core::reference::compute_dynamic_factor(
                            &dynamic_measurements,
                        );
                    }
                }
            }
        }

        if let Some(buffer) = prev_buffer.as_mut() {
            if let ReferenceMode::Prev(offset) = config.ref_mode {
                if buffer.len() == offset {
                    buffer.pop_front();
                }
                buffer.push_back(frame_converted.clone());
            }
        }
        if matches!(config.ref_mode, ReferenceMode::Running) {
            let alpha = config.ref_running_alpha;
            for (running, current) in running_reference.iter_mut().zip(frame_converted.iter()) {
                *running = (1.0 - alpha) * *running + alpha * *current;
            }
        }
    }

    if dumped != targets {
        let missing: Vec<usize> = targets.difference(&dumped).copied().collect();
        bail!("failed to dump requested frame(s): {missing:?}");
    }

    println!(
        "Dumped stage intermediates for frame(s) {:?} to {}",
        frames,
        output_dir.display()
    );
    Ok(())
}

fn write_stage_dump_frame(
    output_dir: &Path,
    frame_index: usize,
    dump: &StageDumpFrame,
) -> Result<()> {
    let frame_dir = output_dir.join(format!("frame_{frame_index:06}"));
    fs::create_dir_all(&frame_dir)?;
    save_npy(
        &frame_dir.join("frame_converted.npy"),
        &dump.frame_converted,
    )?;
    save_npy(
        &frame_dir.join("reference_for_frame.npy"),
        &dump.reference_for_frame,
    )?;
    save_npy(
        &frame_dir.join("aligned_reference.npy"),
        &dump.aligned_reference,
    )?;
    save_npy(&frame_dir.join("delta.npy"), &dump.detect.delta)?;
    save_npy(&frame_dir.join("delta_blur.npy"), &dump.detect.delta_blur)?;
    save_npy(&frame_dir.join("delta_norm.npy"), &dump.detect.delta_norm)?;
    save_npy(&frame_dir.join("binary.npy"), &dump.detect.binary)?;
    save_npy(&frame_dir.join("mask.npy"), &dump.mask)?;
    save_npy(&frame_dir.join("overlay.npy"), &dump.overlay)?;
    save_npy(&frame_dir.join("heatmap.npy"), &dump.detect.heatmap)?;
    Ok(())
}

fn save_npy<T>(path: &Path, array: &T) -> Result<()>
where
    T: ndarray_npy::WriteNpyExt,
{
    write_npy(path, array).with_context(|| format!("writing {}", path.display()))
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

fn convert_to_f32(frame_bgr: &Array3<u8>, colorspace: ColorSpace) -> Result<Array3<f32>> {
    Ok(convert_frame_to_colorspace(frame_bgr, colorspace)?.mapv(|value| value as f32))
}

fn fetch_reference_converted(
    index: usize,
    reader: Option<&mut VideoReader>,
    cache: &mut IndexMap<usize, Array3<f32>>,
    cache_capacity: usize,
    first_frame_converted: &Array3<f32>,
    colorspace: ColorSpace,
) -> Result<Array3<f32>> {
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
    let converted = convert_to_f32(&frame, colorspace)?;
    cache.insert(index, converted.clone());
    while cache.len() > cache_capacity {
        cache.shift_remove_index(0);
    }
    Ok(converted)
}

fn parse_bool(text: &str, flag: &str) -> Result<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{flag} expects true/false but got '{text}'"),
    }
}

fn parse_frame_list(value: Option<&str>) -> Result<Option<Vec<usize>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let mut frames = Vec::new();
    for segment in value
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
    {
        let frame = segment
            .parse::<usize>()
            .with_context(|| format!("invalid frame index '{segment}'"))?;
        if frame == 0 {
            bail!("frame indices are 1-based and must be >= 1");
        }
        frames.push(frame);
    }
    if frames.is_empty() {
        bail!("--debug-dump-frames requires at least one frame index");
    }
    Ok(Some(frames))
}

fn parse_bool_optional(value: Option<&str>, flag: &str) -> Result<Option<bool>> {
    value.map(|text| parse_bool(text, flag)).transpose()
}

fn parse_channel_weights(raw: Option<&str>, num_channels: usize) -> Result<Vec<f32>> {
    let Some(raw) = raw else {
        return Ok(vec![1.0; num_channels]);
    };
    let mut values = Vec::new();
    for segment in raw
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
    {
        values.push(
            segment
                .parse::<f32>()
                .with_context(|| "--channel-weights contains non-numeric values")?,
        );
    }
    if values.is_empty() {
        bail!("--channel-weights requires at least one numeric weight");
    }
    if values.len() == 1 && num_channels > 1 {
        values = vec![values[0]; num_channels];
    }
    if values.len() != num_channels {
        bail!(
            "--channel-weights expects {num_channels} weight(s), got {}",
            values.len()
        );
    }
    Ok(values)
}

fn parse_ref_mode(raw: &[String]) -> Result<ReferenceMode> {
    if raw.is_empty() {
        bail!("--ref-mode requires a mode name");
    }
    match raw[0].to_ascii_lowercase().as_str() {
        "first" => {
            ensure_no_extra(raw, "first")?;
            Ok(ReferenceMode::First)
        }
        "running" => {
            ensure_no_extra(raw, "running")?;
            Ok(ReferenceMode::Running)
        }
        "dynamic" => {
            ensure_no_extra(raw, "dynamic")?;
            Ok(ReferenceMode::Dynamic)
        }
        "prev" => {
            let value = parse_ref_offset(raw, "prev")?;
            if value < 1 {
                bail!("--ref-mode prev requires a positive count (>= 1)");
            }
            Ok(ReferenceMode::Prev(value))
        }
        "absolute" => Ok(ReferenceMode::Absolute(parse_ref_offset(raw, "absolute")?)),
        other => bail!("--ref-mode unsupported option '{other}'"),
    }
}

fn ensure_no_extra(raw: &[String], mode: &str) -> Result<()> {
    if raw.len() != 1 {
        bail!("--ref-mode '{mode}' does not accept extra arguments");
    }
    Ok(())
}

fn parse_ref_offset(raw: &[String], mode: &str) -> Result<usize> {
    if raw.len() != 2 {
        bail!("--ref-mode '{mode}' requires a frame count");
    }
    Ok(raw[1].parse()?)
}

fn parse_annotation_formats(raw: &str) -> Result<Vec<String>> {
    let mut formats = Vec::new();
    for segment in raw
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
    {
        let lower = segment.to_ascii_lowercase();
        if !matches!(lower.as_str(), "coco" | "yolov5" | "darknet") {
            bail!("--annotation-formats expects coco, yolov5, and/or darknet");
        }
        formats.push(lower);
    }
    Ok(formats)
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
    put!("reference_mode", config.ref_mode.name());
    put!("reference_offset", config.ref_mode.offset());
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
    put!("pre_delta_blur_kernel", config.pre_delta_blur_kernel);
    put!("blur_kernel", config.blur_kernel);
    put!("skip_blur", config.skip_blur);
    put!("morph_kernel", config.morph_kernel);
    put!("morph_shape", config.morph_shape.name());
    put!("morph_close_iterations", config.morph_close_iterations);
    put!("morph_open_iterations", config.morph_open_iterations);
    put!("min_area", config.min_area);
    put!(
        "contrast_threshold",
        config.contrast_threshold.map(yaml_f32)
    );
    put!(
        "contrast_percentile",
        config.contrast_percentile.map(yaml_f32)
    );
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
        config.camera_stable.then_some(config.peak_on_shift.name())
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

fn yaml_f32(value: f32) -> f64 {
    ((value as f64) * 1_000_000.0).round() / 1_000_000.0
}

pub fn run_binding_config(config: Config) -> Result<()> {
    run_config(config)
}

pub fn config_from_paths(video_path: PathBuf, output_dir: PathBuf) -> Config {
    Config {
        video_path,
        output_dir,
        ..Config::default()
    }
}

pub fn output_dir(path: impl AsRef<Path>) -> PathBuf {
    path.as_ref().to_path_buf()
}
