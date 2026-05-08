use std::path::{Path, PathBuf};
use std::time::Instant;
use vrifa_cli::{run_config, Config};

fn run_case(
    label: &str,
    repo_root: &Path,
    video_path: &str,
    output_dir: &str,
    roi_margin: f32,
    max_seconds: f64,
) {
    let _ = std::fs::remove_dir_all(output_dir);
    let mut config = Config {
        video_path: repo_root.join(video_path),
        output_dir: PathBuf::from(output_dir),
        roi_margin,
        write_videos: true,
        write_mask_video: true,
        write_overlay_video: true,
        write_heatmap_video: true,
        annotation_formats: vec!["coco".to_string()],
        ..Config::default()
    };
    config.channel_weights = vec![1.0; config.colorspace.channel_count()];

    let started = Instant::now();
    run_config(config).unwrap_or_else(|err| panic!("{label} failed: {err}"));
    let elapsed = started.elapsed().as_secs_f64();
    println!("{label}: {elapsed:.3} s");
    assert!(
        elapsed <= max_seconds,
        "{label} exceeded performance gate: {elapsed:.3} s > {max_seconds:.3} s"
    );
}

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repository root");
    run_case(
        "input_1",
        &repo_root,
        "data/input_1.mp4",
        "/tmp/vrifa_bench_input_1",
        0.15,
        35.0,
    );
    run_case(
        "input_2",
        &repo_root,
        "data/input_2.mp4",
        "/tmp/vrifa_bench_input_2",
        0.0,
        5.0,
    );
}
