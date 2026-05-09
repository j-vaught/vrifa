use anyhow::{bail, Context, Result};
use fast_vrifa_core::{BackendStatus, CpuBackend, ImageBackend, RoiMargins};
use fast_vrifa_cuda::CudaBackend;
use image::RgbImage;
use ndarray::{s, Array2, Array3};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 2 || args.len() > 5 {
        bail!("usage: stage1_bench <source.png> [iterations=50] [roi_margin=0.0] [channel_weight=1.0]");
    }

    let source_path = Path::new(&args[1]);
    let iterations = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("parsing iterations")?
        .unwrap_or(50);
    let roi_margin = args
        .get(3)
        .map(|value| value.parse::<f32>())
        .transpose()
        .context("parsing roi_margin")?
        .unwrap_or(0.0);
    let channel_weight = args
        .get(4)
        .map(|value| value.parse::<f32>())
        .transpose()
        .context("parsing channel_weight")?
        .unwrap_or(1.0);

    let frame_bgr = load_bgr_png(source_path)?;
    let roi_margins = RoiMargins {
        top: roi_margin,
        bottom: roi_margin,
        left: roi_margin,
        right: roi_margin,
    };

    let cpu = CpuBackend;
    let cuda = CudaBackend::new();
    if !matches!(cuda.status(), BackendStatus::Ready) {
        bail!(
            "CUDA backend unavailable: {}",
            cuda.init_error().unwrap_or("unknown initialization failure")
        );
    }

    let cpu_stage = run_stage_once(&cpu, &frame_bgr, roi_margins, channel_weight)?;
    let cuda_stage = run_stage_once(&cuda, &frame_bgr, roi_margins, channel_weight)?;
    println!(
        "parity: lab_max_abs_diff={} mask_max_abs_diff={} delta_max_abs_diff={:.6}",
        max_abs_u8(&cpu_stage.lab, &cuda_stage.lab)?,
        max_abs_u8_2d(&cpu_stage.mask, &cuda_stage.mask)?,
        max_abs_f32(&cpu_stage.delta, &cuda_stage.delta)?,
    );

    let cpu_duration = benchmark_backend(
        &cpu,
        &frame_bgr,
        &cpu_stage.reference_plane,
        roi_margins,
        channel_weight,
        iterations,
    )?;
    let cuda_duration = benchmark_backend(
        &cuda,
        &frame_bgr,
        &cpu_stage.reference_plane,
        roi_margins,
        channel_weight,
        iterations,
    )?;

    let cpu_ms = cpu_duration.as_secs_f64() * 1_000.0;
    let cuda_ms = cuda_duration.as_secs_f64() * 1_000.0;
    println!(
        "timing: iterations={} cpu_total_ms={:.3} cpu_avg_ms={:.3} cuda_total_ms={:.3} cuda_avg_ms={:.3} speedup={:.3}x",
        iterations,
        cpu_ms,
        cpu_ms / iterations as f64,
        cuda_ms,
        cuda_ms / iterations as f64,
        cpu_duration.as_secs_f64() / cuda_duration.as_secs_f64(),
    );

    Ok(())
}

struct StageOutputs {
    lab: Array3<u8>,
    mask: Array2<u8>,
    delta: Array2<f32>,
    reference_plane: Array2<f32>,
}

fn run_stage_once<B>(
    backend: &B,
    frame_bgr: &Array3<u8>,
    roi_margins: RoiMargins,
    channel_weight: f32,
) -> Result<StageOutputs>
where
    B: ImageBackend,
{
    let uploaded = backend.upload_frame_bgr(frame_bgr)?;
    let lab_device = backend.convert_bgr_to_lab(&uploaded)?;
    let lab_host = backend.download_frame_f32(&lab_device)?;
    let lab_u8 = lab_host.mapv(|value| value as u8);
    let reference_plane = lab_host.slice(s![.., .., 0]).to_owned();
    let mask_device = backend.build_roi_mask((lab_host.dim().0, lab_host.dim().1), roi_margins)?;
    let mask_host = backend.download_mask_u8(&mask_device)?;
    let delta_device =
        backend.compute_delta_darken_only(&lab_device, &reference_plane, &mask_device, channel_weight)?;
    let delta_host = backend.download_plane_f32(&delta_device)?;
    Ok(StageOutputs {
        lab: lab_u8,
        mask: mask_host,
        delta: delta_host,
        reference_plane,
    })
}

fn benchmark_backend<B>(
    backend: &B,
    frame_bgr: &Array3<u8>,
    reference_plane: &Array2<f32>,
    roi_margins: RoiMargins,
    channel_weight: f32,
    iterations: usize,
) -> Result<Duration>
where
    B: ImageBackend,
{
    let start = Instant::now();
    for _ in 0..iterations {
        let uploaded = backend.upload_frame_bgr(frame_bgr)?;
        let lab_device = backend.convert_bgr_to_lab(&uploaded)?;
        let mask_device = backend.build_roi_mask((frame_bgr.dim().0, frame_bgr.dim().1), roi_margins)?;
        let delta_device =
            backend.compute_delta_darken_only(&lab_device, reference_plane, &mask_device, channel_weight)?;
        let _ = backend.download_plane_f32(&delta_device)?;
    }
    Ok(start.elapsed())
}

fn load_bgr_png(path: &Path) -> Result<Array3<u8>> {
    let image = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_rgb8();
    rgb_to_bgr_array(&image)
}

fn rgb_to_bgr_array(image: &RgbImage) -> Result<Array3<u8>> {
    let (width, height) = image.dimensions();
    let mut values = Vec::with_capacity((width * height * 3) as usize);
    for pixel in image.pixels() {
        values.push(pixel[2]);
        values.push(pixel[1]);
        values.push(pixel[0]);
    }
    Array3::from_shape_vec((height as usize, width as usize, 3), values)
        .context("reshaping PNG bytes into BGR array")
}

fn max_abs_u8(left: &Array3<u8>, right: &Array3<u8>) -> Result<u8> {
    anyhow::ensure!(left.dim() == right.dim(), "shape mismatch for u8 3D comparison");
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0))
}

fn max_abs_u8_2d(left: &Array2<u8>, right: &Array2<u8>) -> Result<u8> {
    anyhow::ensure!(left.dim() == right.dim(), "shape mismatch for u8 2D comparison");
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0))
}

fn max_abs_f32(left: &Array2<f32>, right: &Array2<f32>) -> Result<f32> {
    anyhow::ensure!(left.dim() == right.dim(), "shape mismatch for f32 comparison");
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max))
}
