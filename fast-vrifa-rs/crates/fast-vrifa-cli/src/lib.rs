use anyhow::{anyhow, bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use clap::Parser;
use fast_vrifa_core::{CpuBackend, ImageBackend, PeakImageBackend};
use indexmap::IndexMap;
use ndarray::{s, Array2, Array3};
use opencv::core::{self, Point, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;
use serde_yaml::Value;
use std::collections::VecDeque;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::Arc;
use std::time::Instant;
use vrifa_annotations::AnnotationFrame;
use vrifa_core::colorspace::{convert_frame_to_colorspace, ColorSpace};
use vrifa_core::contours::extract_bounding_boxes;
use vrifa_core::cvutil;
use vrifa_core::delta::compute_delta;
use vrifa_core::heatmap::apply_turbo_colormap;
use vrifa_core::lock::{apply_locking, LockState};
use vrifa_core::morphology::{MorphShape, MorphologyParams};
use vrifa_core::overlay::create_overlay;
use vrifa_core::peak::update_peak_brightness;
use vrifa_core::reference::{
    compute_dynamic_factor, select_dynamic_reference_index, DynamicReferenceParams,
};
use vrifa_core::roi::resolve_roi_margins;
use vrifa_core::threshold;
use vrifa_io::{AsyncPngWriter, AsyncVideoWriter, VideoReader};

#[cfg(feature = "cuda")]
use fast_vrifa_cuda::CudaBackend;
#[cfg(feature = "wgpu")]
use fast_vrifa_wgpu::WgpuBackend;

pub use vrifa_cli::Config;
use vrifa_cli::ReferenceMode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendMode {
    Delegate,
    Cpu,
    Wgpu,
    Cuda,
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

struct ConvertedFrame<D> {
    device_lab: Option<D>,
    host: Option<Array3<f32>>,
}

struct DetectionOutputs {
    delta_norm: Array2<u8>,
    mask: Array2<u8>,
}

pub fn run() -> Result<()> {
    let raw_args: Vec<OsString> = env::args_os().collect();
    let (backend, stripped_args) = parse_backend_args(raw_args)?;
    if matches!(backend, BackendMode::Delegate) || contains_debug_dump_flags(&stripped_args) {
        let status = forward_to_reference(stripped_args.iter().skip(1))?;
        return handle_reference_status(status);
    }

    let cli_args = vrifa_cli::CliArgs::try_parse_from(stripped_args)
        .context("parsing fast-vrifa CLI arguments")?;
    let config = Config::try_from(cli_args)?;

    run_with_backend(config, backend)
}

pub fn run_config(config: Config) -> Result<()> {
    vrifa_cli::run_binding_config(config).context("delegating bound config to reference vrifa")
}

pub fn run_with_backend_name(config: Config, backend: &str) -> Result<()> {
    let backend = BackendMode::parse(backend)?;
    run_with_backend(config, backend)
}

pub fn run_with_backend(config: Config, backend: BackendMode) -> Result<()> {
    match backend {
        BackendMode::Delegate => run_config(config),
        BackendMode::Cpu => run_cpu_backend(config),
        BackendMode::Wgpu => run_wgpu_backend(config),
        BackendMode::Cuda => run_cuda_backend(config),
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

fn parse_backend_args(raw_args: Vec<OsString>) -> Result<(BackendMode, Vec<OsString>)> {
    let mut args = raw_args.into_iter();
    let program = args.next().unwrap_or_else(|| OsString::from("fast-vrifa"));
    let mut stripped = vec![program];
    let mut backend = BackendMode::Delegate;

    while let Some(arg) = args.next() {
        if let Some(text) = arg.to_str() {
            if text == "--backend" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--backend requires a value"))?;
                backend = BackendMode::parse(
                    value
                        .to_str()
                        .ok_or_else(|| anyhow!("--backend value must be valid UTF-8"))?,
                )?;
                continue;
            }
            if let Some(value) = text.strip_prefix("--backend=") {
                backend = BackendMode::parse(value)?;
                continue;
            }
        }
        stripped.push(arg);
    }

    Ok((backend, stripped))
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

fn run_cpu_backend(config: Config) -> Result<()> {
    let backend = CpuBackend;
    run_hybrid_pipeline(config, &backend)
}

fn run_wgpu_backend(config: Config) -> Result<()> {
    #[cfg(feature = "wgpu")]
    {
        let backend = WgpuBackend::new().context("initializing wgpu backend")?;
        return run_hybrid_pipeline(config, &backend);
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let _ = config;
        bail!("--backend wgpu requires building fast-vrifa with --features wgpu");
    }
}

fn run_cuda_backend(config: Config) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let backend = CudaBackend::new();
        if !matches!(backend.status(), fast_vrifa_core::BackendStatus::Ready) {
            let detail = backend
                .init_error()
                .unwrap_or("CUDA runtime initialization failed");
            bail!("--backend cuda is unavailable on this machine: {detail}");
        }
        return run_hybrid_pipeline(config, &backend);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = config;
        bail!("--backend cuda requires building fast-vrifa with --features cuda");
    }
}

fn run_hybrid_pipeline<B>(config: Config, backend: &B) -> Result<()>
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
    let reference_values_needed = !(config.peak_reference && config.darken_only);
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

    let roi_margins = resolve_roi_margins(
        config.roi_margin,
        config.roi_margin_top,
        config.roi_margin_bottom,
        config.roi_margin_left,
        config.roi_margin_right,
    );
    let device_roi_mask = backend.build_roi_mask(
        (first_frame_converted.dim().0, first_frame_converted.dim().1),
        roi_margins,
    )?;
    let roi_mask = backend.download_mask_u8(&device_roi_mask)?;
    let roi_pixels = roi_mask.iter().filter(|value| **value > 0).count();
    let mut lock_state = (config.lock_frames > 0).then(|| LockState::new(roi_mask.dim()));
    let device_delta_eligible =
        config.darken_only && matches!(config.colorspace, ColorSpace::Cielab);
    let mut peak_brightness_map = if config.peak_reference && !device_delta_eligible {
        Some(first_frame_converted.slice(s![.., .., 0]).to_owned())
    } else {
        None
    };
    let mut peak_brightness_device = if config.peak_reference && device_delta_eligible {
        first_frame
            .device_lab
            .as_ref()
            .map(|frame| backend.extract_l_plane(frame))
            .transpose()?
    } else {
        None
    };

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
    let need_host_current = !device_delta_eligible
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
            if config.peak_reference {
                if device_delta_eligible {
                    let device_lab = current.device_lab.as_ref().ok_or_else(|| {
                        anyhow!("device peak path requires a device CIELAB frame")
                    })?;
                    peak_brightness_device = Some(backend.update_peak_brightness_device(
                        device_lab,
                        peak_brightness_device.as_ref(),
                    )?);
                } else {
                    peak_brightness_map = Some(update_peak_brightness(
                        frame_converted.ok_or_else(|| {
                            anyhow!("host peak path requires a converted host frame")
                        })?,
                        peak_brightness_map.as_ref(),
                    )?);
                }
            }

            let delta = if let Some(device_lab) = current.device_lab.as_ref() {
                if device_delta_eligible {
                    let device_delta = if config.peak_reference {
                        backend.compute_delta_darken_only_device(
                            device_lab,
                            peak_brightness_device.as_ref().ok_or_else(|| {
                                anyhow!("peak reference was enabled without a device peak map")
                            })?,
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
                    backend.download_plane_f32(&device_delta)?
                } else {
                    let host_reference = reference_for_frame
                        .as_ref()
                        .or((!reference_values_needed).then_some(&first_frame_converted))
                        .ok_or_else(|| anyhow!("missing reference frame for host delta"))?;
                    compute_delta(
                        frame_converted.ok_or_else(|| {
                            anyhow!("host delta path requires a converted host frame")
                        })?,
                        host_reference,
                        &roi_mask,
                        &config.channel_weights,
                        config.darken_only,
                        peak_brightness_map
                            .as_ref()
                            .filter(|_| config.peak_reference),
                    )?
                }
            } else {
                let host_reference = reference_for_frame
                    .as_ref()
                    .or((!reference_values_needed).then_some(&first_frame_converted))
                    .ok_or_else(|| anyhow!("missing reference frame for host delta"))?;
                compute_delta(
                    frame_converted.ok_or_else(|| {
                        anyhow!("host delta path requires a converted host frame")
                    })?,
                    host_reference,
                    &roi_mask,
                    &config.channel_weights,
                    config.darken_only,
                    peak_brightness_map
                        .as_ref()
                        .filter(|_| config.peak_reference),
                )?
            };

            let detect = detect_mask_and_delta_norm(
                &delta,
                &roi_mask,
                &MorphologyParams {
                    blur_kernel: config.blur_kernel,
                    morph_kernel: config.morph_kernel,
                    min_area: config.min_area,
                    manual_threshold: config.contrast_threshold,
                    percentile_threshold: config.contrast_percentile,
                    threshold_offset: config.threshold_offset,
                    blur_enabled: !config.skip_blur,
                    morph_shape: config.morph_shape,
                    morph_close_iterations: config.morph_close_iterations,
                    morph_open_iterations: config.morph_open_iterations,
                },
            )?;
            let mask = Arc::new(apply_locking(
                &detect.mask,
                config.lock_frames,
                lock_state.as_mut(),
            )?);
            let write_overlay = config.write_overlay_pngs || config.write_overlay_video;
            let write_heatmap = config.write_heatmap_pngs || config.write_heatmap_video;
            let overlay = if write_overlay {
                Some(Arc::new(create_overlay(&frame_bgr, &mask)?))
            } else {
                None
            };
            let heatmap = if write_heatmap {
                Some(Arc::new(apply_turbo_colormap(&detect.delta_norm)?))
            } else {
                None
            };

            if !config.annotation_formats.is_empty() {
                let boxes = extract_bounding_boxes(
                    &mask,
                    config.annotation_segmentation_tolerance,
                    config.annotation_segmentation_max_edge_length,
                )?;
                let frame_bgr = if let Some(writer) = coco_image_writer.as_mut() {
                    writer.write_bgr(
                        coco_images_dir.join(format!("frame_{frame_index:06}.png")),
                        frame_bgr.clone(),
                    )?;
                    None
                } else {
                    Some(frame_bgr.clone())
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
            if let (Some(writer), Some(overlay)) = (overlay_png_writer.as_mut(), overlay.as_ref()) {
                writer.write_bgr(overlay_dir.join(&basename), (**overlay).clone())?;
            }
            if let (Some(writer), Some(heatmap)) = (heatmap_png_writer.as_mut(), heatmap.as_ref()) {
                writer.write_bgr(heatmap_dir.join(&basename), (**heatmap).clone())?;
            }
            if let Some(writer) = mask_writer.as_mut() {
                writer.write_gray(mask.clone())?;
            }
            if let (Some(writer), Some(overlay)) = (overlay_writer.as_mut(), overlay.as_ref()) {
                writer.write_bgr(overlay.clone())?;
            }
            if let (Some(writer), Some(heatmap)) = (heat_writer.as_mut(), heatmap.as_ref()) {
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

fn detect_mask_and_delta_norm(
    delta: &Array2<f32>,
    roi_mask: &Array2<u8>,
    params: &MorphologyParams,
) -> Result<DetectionOutputs> {
    let delta_blur = if params.blur_enabled {
        let mut kernel = params.blur_kernel;
        if kernel % 2 == 0 {
            kernel += 1;
        }
        let src = cvutil::array2_f32_to_mat(delta)?;
        let mut dst = opencv::core::Mat::default();
        imgproc::gaussian_blur_def(&src, &mut dst, Size::new(kernel as i32, kernel as i32), 0.0)?;
        dst
    } else {
        cvutil::array2_f32_to_mat(delta)?
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

    Ok(DetectionOutputs {
        delta_norm,
        mask: cvutil::mat_to_array2_u8(&binary)?,
    })
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
        delegated_backend_label, parse_backend_args, reference_binary_candidates, BackendMode,
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
        let (backend, stripped) = parse_backend_args(args).unwrap();
        assert_eq!(backend, BackendMode::Wgpu);
        assert_eq!(stripped.len(), 3);
        assert_eq!(stripped[1], OsString::from("--video-path"));
    }

    #[test]
    fn backend_parser_accepts_cpu_and_cuda() {
        assert_eq!(BackendMode::parse("cpu").unwrap(), BackendMode::Cpu);
        assert_eq!(BackendMode::parse("cuda").unwrap(), BackendMode::Cuda);
    }
}
