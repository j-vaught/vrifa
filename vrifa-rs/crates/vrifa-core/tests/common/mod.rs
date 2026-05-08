#![allow(dead_code)]

use image::ImageReader;
use ndarray::{Array, Array0, Array2, Array3, ArrayBase, Data, Dimension};
use ndarray_npy::{read_npy, NpzReader, ReadableElement};
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};
use vrifa_core::{AnnotationBox, ColorSpace, MorphShape, MorphologyParams};

pub const ALL_STAGE_FRAMES: [(&str, usize); 6] = [
    ("input_1", 50),
    ("input_1", 200),
    ("input_1", 500),
    ("input_2", 30),
    ("input_2", 60),
    ("input_2", 90),
];
pub const INPUT_2_FRAMES: [usize; 3] = [30, 60, 90];

#[derive(Clone, Debug, Deserialize)]
pub struct FixtureConfig {
    pub colorspace: String,
    pub blur_kernel: usize,
    pub morph_kernel: usize,
    pub min_area: usize,
    pub manual_threshold: Option<f32>,
    pub percentile_threshold: Option<f32>,
    pub threshold_offset: f32,
    pub channel_weights: Vec<f32>,
    pub blur_enabled: bool,
    pub morph_shape: String,
    pub morph_close_iterations: usize,
    pub morph_open_iterations: usize,
    pub darken_only: bool,
    pub peak_reference: bool,
    pub annotation_segmentation_tolerance: f32,
    pub annotation_segmentation_max_edge_length: f32,
    pub lock_frames: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DynamicParamsFixture {
    pub factor: f32,
    pub frame_index: usize,
    pub fps: f32,
    pub roi_pixels: usize,
    pub target_fraction: f32,
    pub lag_scale: f32,
    pub linear_mode: bool,
    pub linear_start: usize,
    pub linear_max: usize,
    pub total_frames: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DynamicExpectedFixture {
    pub factor: f32,
    pub delta_t: f32,
    pub reference_index: usize,
}

pub fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

pub fn input_root(input: &str) -> PathBuf {
    fixtures_root().join(input)
}

pub fn frame_root(input: &str, frame: usize) -> PathBuf {
    input_root(input).join(format!("frame_{frame:06}"))
}

pub fn load_fixture_config(input: &str) -> FixtureConfig {
    load_json(input_root(input).join("config.json"))
}

pub fn load_dynamic_params() -> DynamicParamsFixture {
    load_json(fixtures_root().join("reference_dynamic/params.json"))
}

pub fn load_dynamic_expected() -> DynamicExpectedFixture {
    load_json(fixtures_root().join("reference_dynamic/expected.json"))
}

pub fn colorspace_from_fixture(input: &str) -> ColorSpace {
    ColorSpace::parse(&load_fixture_config(input).colorspace).expect("fixture colorspace parses")
}

pub fn morphology_params(config: &FixtureConfig) -> MorphologyParams {
    MorphologyParams {
        blur_kernel: config.blur_kernel,
        morph_kernel: config.morph_kernel,
        min_area: config.min_area,
        manual_threshold: config.manual_threshold,
        percentile_threshold: config.percentile_threshold,
        threshold_offset: config.threshold_offset,
        blur_enabled: config.blur_enabled,
        morph_shape: MorphShape::parse(&config.morph_shape),
        morph_close_iterations: config.morph_close_iterations,
        morph_open_iterations: config.morph_open_iterations,
    }
}

pub fn load_source_bgr(input: &str, frame: usize) -> Array3<u8> {
    let image = ImageReader::open(frame_root(input, frame).join("source.png"))
        .expect("open source png")
        .decode()
        .expect("decode source png")
        .to_rgb8();
    let (width, height) = image.dimensions();
    let mut out = Array3::<u8>::zeros((height as usize, width as usize, 3));
    for (x, y, pixel) in image.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        out[(y as usize, x as usize, 0)] = b;
        out[(y as usize, x as usize, 1)] = g;
        out[(y as usize, x as usize, 2)] = r;
    }
    out
}

pub fn load_frame_converted_f32(input: &str, frame: usize) -> Array3<f32> {
    load_array(&frame_root(input, frame).join("frame_converted"))
}

pub fn load_frame_converted_u8(input: &str, frame: usize) -> Array3<u8> {
    load_frame_converted_f32(input, frame).mapv(|value| value as u8)
}

pub fn load_roi_mask(input: &str) -> Array2<u8> {
    load_array(&input_root(input).join("roi_mask"))
}

pub fn load_delta(input: &str, frame: usize) -> Array2<f32> {
    load_array(&frame_root(input, frame).join("delta"))
}

pub fn load_peak_before(input: &str, frame: usize) -> Array2<f32> {
    load_array(&frame_root(input, frame).join("peak_before"))
}

pub fn load_delta_blur(input: &str, frame: usize) -> Array2<f32> {
    load_array(&frame_root(input, frame).join("delta_blur"))
}

pub fn load_delta_norm(input: &str, frame: usize) -> Array2<u8> {
    load_array(&frame_root(input, frame).join("delta_norm"))
}

pub fn load_binary(input: &str, frame: usize) -> Array2<u8> {
    load_array(&frame_root(input, frame).join("binary"))
}

pub fn load_mask(input: &str, frame: usize) -> Array2<u8> {
    load_array(&frame_root(input, frame).join("mask"))
}

pub fn load_mask_pre_lock(input: &str, frame: usize) -> Array2<u8> {
    load_array(&frame_root(input, frame).join("mask_pre_lock"))
}

pub fn load_overlay(input: &str, frame: usize) -> Array3<u8> {
    load_array(&frame_root(input, frame).join("overlay"))
}

pub fn load_heatmap(input: &str, frame: usize) -> Array3<u8> {
    load_array(&frame_root(input, frame).join("heatmap"))
}

pub fn load_threshold(input: &str, frame: usize) -> f32 {
    let scalar: Array0<f32> = load_array(&frame_root(input, frame).join("threshold"));
    scalar.into_scalar()
}

pub fn load_contours_rows(input: &str, frame: usize) -> Vec<[i32; 6]> {
    let rows: Array2<i32> = load_array(&frame_root(input, frame).join("contours_boxes"));
    rows.outer_iter()
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect()
}

pub fn load_peak_after_3(input: &str) -> Array2<f32> {
    load_array(&input_root(input).join("peak_after_3"))
}

pub fn load_lock_sequence() -> Array3<u8> {
    load_array(&fixtures_root().join("lock/mask_sequence"))
}

pub fn load_locked_mask() -> Array2<u8> {
    load_array(&fixtures_root().join("lock/locked_mask"))
}

pub fn load_lock_frames() -> usize {
    let scalar: Array0<i32> = load_array(&fixtures_root().join("lock/lock_frames"));
    scalar.into_scalar() as usize
}

pub fn load_dynamic_measurements() -> Array2<f32> {
    load_array(&fixtures_root().join("reference_dynamic/measurements"))
}

pub fn sort_box_rows(rows: &mut [[i32; 6]]) {
    rows.sort_by_key(|row| (row[1], row[0], row[2], row[3]));
}

pub fn box_rows(boxes: &[AnnotationBox]) -> Vec<[i32; 6]> {
    let mut rows: Vec<[i32; 6]> = boxes
        .iter()
        .map(|box_| {
            [
                box_.x,
                box_.y,
                box_.w,
                box_.h,
                box_.area,
                box_.segmentation.len() as i32,
            ]
        })
        .collect();
    sort_box_rows(&mut rows);
    rows
}

pub fn assert_u8_exact<S, D>(label: &str, actual: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
where
    S: Data<Elem = u8>,
    D: Dimension,
{
    assert_eq!(actual.raw_dim(), expected.raw_dim(), "{label} shape mismatch");
    if actual != expected {
        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (*a as i16 - *e as i16).unsigned_abs())
            .max()
            .unwrap_or(0);
        panic!("{label} differs; max_abs_diff={max_abs}");
    }
}

pub fn assert_u8_max_abs<S, D>(
    label: &str,
    actual: &ArrayBase<S, D>,
    expected: &ArrayBase<S, D>,
    tolerance: u8,
) where
    S: Data<Elem = u8>,
    D: Dimension,
{
    assert_eq!(actual.raw_dim(), expected.raw_dim(), "{label} shape mismatch");
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (*a as i16 - *e as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_abs <= tolerance,
        "{label} max_abs_diff={max_abs} exceeded tolerance={tolerance}"
    );
}

pub fn assert_f32_max_abs<S, D>(
    label: &str,
    actual: &ArrayBase<S, D>,
    expected: &ArrayBase<S, D>,
    tolerance: f32,
) where
    S: Data<Elem = f32>,
    D: Dimension,
{
    assert_eq!(actual.raw_dim(), expected.raw_dim(), "{label} shape mismatch");
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (*a - *e).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= tolerance,
        "{label} max_abs_diff={max_abs} exceeded tolerance={tolerance}"
    );
}

pub fn assert_f32_relative<S, D>(
    label: &str,
    actual: &ArrayBase<S, D>,
    expected: &ArrayBase<S, D>,
    tolerance: f32,
) where
    S: Data<Elem = f32>,
    D: Dimension,
{
    assert_eq!(actual.raw_dim(), expected.raw_dim(), "{label} shape mismatch");
    let mut max_scaled = 0.0f32;
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        let diff = (*actual_value - *expected_value).abs();
        let scaled = diff / expected_value.abs().max(1.0);
        max_scaled = max_scaled.max(scaled);
    }
    assert!(
        max_scaled <= tolerance,
        "{label} max_scaled_diff={max_scaled} exceeded tolerance={tolerance}"
    );
}

fn load_array<T, D>(base_path: &Path) -> Array<T, D>
where
    T: ReadableElement,
    D: Dimension,
{
    let npy_path = base_path.with_extension("npy");
    if npy_path.exists() {
        return read_npy(&npy_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", npy_path.display()));
    }

    let npz_path = base_path.with_extension("npz");
    let file = File::open(&npz_path)
        .unwrap_or_else(|error| panic!("failed to open {}: {error}", npz_path.display()));
    let mut archive = NpzReader::new(file)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", npz_path.display()));
    archive.by_name("data.npy").unwrap_or_else(|error| {
        panic!("failed to read data.npy from {}: {error}", npz_path.display())
    })
}

fn load_json<T: for<'de> Deserialize<'de>>(path: PathBuf) -> T {
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}
