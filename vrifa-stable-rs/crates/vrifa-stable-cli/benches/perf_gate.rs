use std::path::{Path, PathBuf};
use std::time::Instant;
use vrifa_stable_cli::{run_config, Config};

struct Case {
    label: &'static str,
    video_path: &'static str,
    output_dir: &'static str,
    roi_margin: f32,
    write_videos: bool,
    write_pngs: bool,
    annotation_formats: &'static [&'static str],
    max_seconds: f64,
}

fn run_case(
    repo_root: &Path,
    case: &Case,
) {
    let _ = std::fs::remove_dir_all(case.output_dir);
    let mut config = Config {
        video_path: repo_root.join(case.video_path),
        output_dir: PathBuf::from(case.output_dir),
        roi_margin: case.roi_margin,
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
    config.channel_weights = vec![1.0; config.colorspace.channel_count()];

    let started = Instant::now();
    run_config(config).unwrap_or_else(|err| panic!("{} failed: {err}", case.label));
    let elapsed = started.elapsed().as_secs_f64();
    println!("{}: {elapsed:.3} s", case.label);
    assert!(
        elapsed <= case.max_seconds,
        "{} exceeded performance gate: {elapsed:.3} s > {:.3} s",
        case.label,
        case.max_seconds
    );
}

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repository root");
    let cases = [
        Case {
            label: "input_1_detector",
            video_path: "data/input_1.mp4",
            output_dir: "/tmp/vrifa_bench_input_1_detector",
            roi_margin: 0.15,
            write_videos: false,
            write_pngs: false,
            annotation_formats: &[],
            max_seconds: 32.0,
        },
        Case {
            label: "input_1_core",
            video_path: "data/input_1.mp4",
            output_dir: "/tmp/vrifa_bench_input_1_core",
            roi_margin: 0.15,
            write_videos: true,
            write_pngs: false,
            annotation_formats: &[],
            max_seconds: 35.0,
        },
        Case {
            label: "input_1_full",
            video_path: "data/input_1.mp4",
            output_dir: "/tmp/vrifa_bench_input_1_full",
            roi_margin: 0.15,
            write_videos: true,
            write_pngs: true,
            annotation_formats: &["coco"],
            max_seconds: 90.0,
        },
        Case {
            label: "input_2_detector",
            video_path: "data/input_2.mp4",
            output_dir: "/tmp/vrifa_bench_input_2_detector",
            roi_margin: 0.0,
            write_videos: false,
            write_pngs: false,
            annotation_formats: &[],
            max_seconds: 5.0,
        },
        Case {
            label: "input_2_core",
            video_path: "data/input_2.mp4",
            output_dir: "/tmp/vrifa_bench_input_2_core",
            roi_margin: 0.0,
            write_videos: true,
            write_pngs: false,
            annotation_formats: &[],
            max_seconds: 6.0,
        },
        Case {
            label: "input_2_full",
            video_path: "data/input_2.mp4",
            output_dir: "/tmp/vrifa_bench_input_2_full",
            roi_margin: 0.0,
            write_videos: true,
            write_pngs: true,
            annotation_formats: &["coco"],
            max_seconds: 11.0,
        },
    ];

    for case in &cases {
        run_case(&repo_root, case);
    }
}
