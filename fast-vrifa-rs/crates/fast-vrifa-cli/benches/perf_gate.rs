use fast_vrifa_cli::{run_with_backend_options, BackendMode, Config, RunOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;
use vrifa_core::colorspace::ColorSpace;

struct Case {
    label: &'static str,
    video_path: &'static str,
    output_dir: &'static str,
    roi_margin: f32,
    camera_stable: bool,
    pre_delta_blur_kernel: usize,
    write_videos: bool,
    write_pngs: bool,
    annotation_formats: &'static [&'static str],
    options: RunOptions,
    max_seconds: f64,
}

fn measure_case(repo_root: &Path, case: &Case) -> f64 {
    let _ = std::fs::remove_dir_all(case.output_dir);
    let mut config = Config {
        video_path: repo_root.join(case.video_path),
        output_dir: PathBuf::from(case.output_dir),
        roi_margin: case.roi_margin,
        camera_stable: case.camera_stable,
        pre_delta_blur_kernel: case.pre_delta_blur_kernel,
        write_videos: case.write_videos,
        write_mask_pngs: case.write_pngs,
        write_overlay_pngs: case.write_pngs,
        write_heatmap_pngs: case.write_pngs,
        write_mask_video: case.write_videos,
        write_overlay_video: case.write_videos,
        write_heatmap_video: case.write_videos,
        annotation_formats: case
            .annotation_formats
            .iter()
            .map(|format| (*format).to_string())
            .collect(),
        ..Config::default()
    };
    config.colorspace = ColorSpace::Cielab;
    config.channel_weights = vec![1.0; config.colorspace.channel_count()];

    let started = Instant::now();
    run_with_backend_options(config, BackendMode::Cuda, case.options)
        .unwrap_or_else(|err| panic!("{} failed: {err}", case.label));
    started.elapsed().as_secs_f64()
}

fn run_case(repo_root: &Path, case: &Case) {
    let mut samples = [0.0f64; 3];
    for sample in &mut samples {
        *sample = measure_case(repo_root, case);
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let elapsed = samples[1];
    println!(
        "{}: {:.3} s (runs: {:.3}, {:.3}, {:.3})",
        case.label, elapsed, samples[0], samples[1], samples[2]
    );
    assert!(
        elapsed <= case.max_seconds,
        "{} exceeded performance gate: {:.3} s > {:.3} s",
        case.label,
        elapsed,
        case.max_seconds
    );
}

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repository root");
    let default = RunOptions::default();
    let mask_only = RunOptions {
        mask_only: true,
        ..RunOptions::default()
    };
    let cases = [
        Case {
            label: "input_1_detector",
            video_path: "data/input_1.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_1_detector",
            roi_margin: 0.15,
            camera_stable: false,
            pre_delta_blur_kernel: 0,
            write_videos: false,
            write_pngs: false,
            annotation_formats: &[],
            options: default,
            max_seconds: 3.8,
        },
        Case {
            label: "input_1_core",
            video_path: "data/input_1.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_1_core",
            roi_margin: 0.15,
            camera_stable: false,
            pre_delta_blur_kernel: 0,
            write_videos: true,
            write_pngs: false,
            annotation_formats: &[],
            options: default,
            max_seconds: 4.1,
        },
        Case {
            label: "input_1_full",
            video_path: "data/input_1.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_1_full",
            roi_margin: 0.15,
            camera_stable: false,
            pre_delta_blur_kernel: 0,
            write_videos: true,
            write_pngs: true,
            annotation_formats: &["coco"],
            options: default,
            max_seconds: 14.0,
        },
        Case {
            label: "input_1_mask_only",
            video_path: "data/input_1.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_1_mask_only",
            roi_margin: 0.15,
            camera_stable: false,
            pre_delta_blur_kernel: 0,
            write_videos: true,
            write_pngs: true,
            annotation_formats: &["coco"],
            options: mask_only,
            max_seconds: 6.2,
        },
        Case {
            label: "input_1_stabilized",
            video_path: "data/input_1.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_1_stabilized",
            roi_margin: 0.15,
            camera_stable: true,
            pre_delta_blur_kernel: 5,
            write_videos: false,
            write_pngs: false,
            annotation_formats: &[],
            options: default,
            max_seconds: 6.5,
        },
        Case {
            label: "input_2_detector",
            video_path: "data/input_2.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_2_detector",
            roi_margin: 0.0,
            camera_stable: false,
            pre_delta_blur_kernel: 0,
            write_videos: false,
            write_pngs: false,
            annotation_formats: &[],
            options: default,
            max_seconds: 1.1,
        },
        Case {
            label: "input_2_core",
            video_path: "data/input_2.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_2_core",
            roi_margin: 0.0,
            camera_stable: false,
            pre_delta_blur_kernel: 0,
            write_videos: true,
            write_pngs: false,
            annotation_formats: &[],
            options: default,
            max_seconds: 1.2,
        },
        Case {
            label: "input_2_full",
            video_path: "data/input_2.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_2_full",
            roi_margin: 0.0,
            camera_stable: false,
            pre_delta_blur_kernel: 0,
            write_videos: true,
            write_pngs: true,
            annotation_formats: &["coco"],
            options: default,
            max_seconds: 2.3,
        },
        Case {
            label: "input_2_mask_only",
            video_path: "data/input_2.mp4",
            output_dir: "/tmp/fast_vrifa_bench_input_2_mask_only",
            roi_margin: 0.0,
            camera_stable: false,
            pre_delta_blur_kernel: 0,
            write_videos: true,
            write_pngs: true,
            annotation_formats: &["coco"],
            options: mask_only,
            max_seconds: 1.4,
        },
    ];

    for case in &cases {
        run_case(&repo_root, case);
    }
}
