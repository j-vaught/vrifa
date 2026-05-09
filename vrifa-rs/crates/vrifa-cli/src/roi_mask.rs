use anyhow::{anyhow, bail, Context, Result};
#[cfg(test)]
use image::{GrayImage, ImageBuffer, Luma};
use ndarray::Array2;
use serde::Deserialize;
#[cfg(test)]
use std::fs;
use std::fs::File;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

pub fn load_roi_mask(path: &Path, video_path: &Path, shape: (usize, usize)) -> Result<Array2<u8>> {
    match extension(path)?.as_str() {
        "png" => load_roi_mask_png(path, shape),
        "json" => load_roi_mask_coco(path, video_path, shape),
        other => bail!(
            "--roi-mask expects a .png or .json file, got extension '.{other}'"
        ),
    }
}

pub fn load_roi_mask_png(path: &Path, shape: (usize, usize)) -> Result<Array2<u8>> {
    let image = image::open(path)
        .with_context(|| format!("opening ROI mask {}", path.display()))?
        .to_luma8();
    let (width, height) = image.dimensions();
    ensure_dimensions(width as usize, height as usize, shape)?;
    let mask = Array2::from_shape_vec(
        (height as usize, width as usize),
        image
            .into_raw()
            .into_iter()
            .map(|value| u8::from(value > 127))
            .collect(),
    )
    .context("reshaping ROI mask PNG")?;
    ensure_non_empty(&mask, path)?;
    Ok(mask)
}

pub fn load_roi_mask_coco(
    path: &Path,
    video_path: &Path,
    shape: (usize, usize),
) -> Result<Array2<u8>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let coco: CocoFile =
        serde_json::from_reader(file).with_context(|| format!("parsing {}", path.display()))?;
    let image = choose_coco_image(&coco.images, video_path)?;
    let mut polygons = Vec::new();
    for annotation in coco.annotations.iter().filter(|ann| ann.image_id == image.id) {
        if let Some(segmentation) = &annotation.segmentation {
            polygons.extend(segmentation.polygons());
        }
    }
    if polygons.is_empty() {
        bail!(
            "--roi-mask file '{}' contains no polygon annotations for '{}'",
            path.display(),
            image.file_name
        );
    }
    let mask = rasterize_polygons(shape, &polygons);
    ensure_non_empty(&mask, path)?;
    Ok(mask)
}

pub fn rasterize_polygons(shape: (usize, usize), polygons: &[Vec<f32>]) -> Array2<u8> {
    let (height, width) = shape;
    let mut mask = Array2::<u8>::zeros((height, width));
    for polygon in polygons {
        if polygon.len() < 6 {
            continue;
        }
        let points = polygon
            .chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect::<Vec<_>>();
        if points.len() < 3 {
            continue;
        }
        for y in 0..height {
            let scan_y = y as f32 + 0.5;
            let mut intersections = Vec::new();
            for ((x0, y0), (x1, y1)) in points
                .iter()
                .copied()
                .zip(points.iter().copied().cycle().skip(1))
                .take(points.len())
            {
                if (y0 - y1).abs() <= f32::EPSILON {
                    continue;
                }
                let min_y = y0.min(y1);
                let max_y = y0.max(y1);
                if scan_y < min_y || scan_y >= max_y {
                    continue;
                }
                let t = (scan_y - y0) / (y1 - y0);
                intersections.push(x0 + t * (x1 - x0));
            }
            intersections.sort_by(|left, right| left.total_cmp(right));
            for pair in intersections.chunks_exact(2) {
                let start = pair[0].min(pair[1]);
                let end = pair[0].max(pair[1]);
                let x0 = ((start - 0.5).ceil().max(0.0)) as usize;
                let x1 = ((end - 0.5).floor().min(width.saturating_sub(1) as f32)) as isize;
                if x0 >= width || x1 < x0 as isize {
                    continue;
                }
                for x in x0..=(x1 as usize) {
                    mask[(y, x)] = 1;
                }
            }
        }
    }
    mask
}

fn extension(path: &Path) -> Result<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .ok_or_else(|| anyhow!("--roi-mask path '{}' has no extension", path.display()))
}

fn ensure_dimensions(width: usize, height: usize, expected: (usize, usize)) -> Result<()> {
    let (expected_height, expected_width) = expected;
    if (height, width) == (expected_height, expected_width) {
        return Ok(());
    }
    bail!(
        "ROI mask is {}×{} but video is {}×{}",
        width,
        height,
        expected_width,
        expected_height
    );
}

fn ensure_non_empty(mask: &Array2<u8>, path: &Path) -> Result<()> {
    if mask.iter().any(|value| *value > 0) {
        return Ok(());
    }
    bail!(
        "--roi-mask '{}' produced an empty mask after clipping",
        path.display()
    );
}

fn choose_coco_image<'a>(images: &'a [CocoImage], video_path: &Path) -> Result<&'a CocoImage> {
    if images.len() == 1 {
        return Ok(&images[0]);
    }

    let video_stem = video_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow!("video path must have a UTF-8 stem"))?;
    let prefix = format!("{video_stem}__");
    let mut matches = images
        .iter()
        .filter(|image| image.file_name.starts_with(&prefix))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        bail!(
            "--roi-mask file has no images matching the video name; pass an explicit mask file or rename the video to match an entry"
        );
    }
    matches.sort_by_key(|image| extract_frame_index(&image.file_name).unwrap_or(0));
    matches
        .pop()
        .ok_or_else(|| anyhow!("matched COCO image list unexpectedly empty"))
}

fn extract_frame_index(file_name: &str) -> Option<usize> {
    let marker = file_name.rfind("frame_")?;
    let digits = file_name[marker + "frame_".len()..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

#[derive(Deserialize)]
struct CocoFile {
    #[serde(default)]
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<CocoAnnotation>,
}

#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
}

#[derive(Deserialize)]
struct CocoAnnotation {
    image_id: u64,
    #[serde(default)]
    segmentation: Option<Segmentation>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Segmentation {
    MultiPolygon(Vec<Vec<f32>>),
    Polygon(Vec<f32>),
    Other(()),
}

impl Segmentation {
    fn polygons(&self) -> Vec<Vec<f32>> {
        match self {
            Self::MultiPolygon(polygons) => polygons.clone(),
            Self::Polygon(polygon) => vec![polygon.clone()],
            Self::Other(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{load_roi_mask_coco, load_roi_mask_png, rasterize_polygons};
    use super::{fs, AtomicU64, GrayImage, ImageBuffer, Luma, Ordering, Path, PathBuf};

    fn temp_path(name: &str) -> PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("vrifa-roi-mask-tests-{id}"));
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn png_mask_round_trip_matches_reference() {
        let polygon = vec![1.0, 1.0, 5.0, 1.0, 5.0, 4.0, 1.0, 4.0];
        let expected = rasterize_polygons((6, 8), &[polygon]);
        let image = GrayImage::from_fn(8, 6, |x, y| {
            let value = if expected[(y as usize, x as usize)] > 0 {
                255
            } else {
                0
            };
            Luma([value])
        });
        let path = temp_path("roi.png");
        image.save(&path).unwrap();
        let actual = load_roi_mask_png(&path, (6, 8)).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn coco_loader_picks_highest_matching_frame_index() {
        let path = temp_path("labels.json");
        fs::write(
            &path,
            r#"{
                "images": [
                    {"id": 1, "file_name": "input_1__frame_000010.png"},
                    {"id": 2, "file_name": "input_1__frame_000055.png"},
                    {"id": 3, "file_name": "input_2__frame_000090.png"}
                ],
                "annotations": [
                    {"image_id": 1, "segmentation": [[1,1, 2,1, 2,2, 1,2]]},
                    {"image_id": 2, "segmentation": [[3,1, 5,1, 5,3, 3,3]]},
                    {"image_id": 3, "segmentation": [[1,3, 2,3, 2,4, 1,4]]}
                ]
            }"#,
        )
        .unwrap();

        let input1 = load_roi_mask_coco(&path, Path::new("data/input_1.mp4"), (6, 8)).unwrap();
        let input2 = load_roi_mask_coco(&path, Path::new("data/input_2.mp4"), (6, 8)).unwrap();

        assert_eq!(
            input1,
            rasterize_polygons((6, 8), &[vec![3.0, 1.0, 5.0, 1.0, 5.0, 3.0, 3.0, 3.0]])
        );
        assert_eq!(
            input2,
            rasterize_polygons((6, 8), &[vec![1.0, 3.0, 2.0, 3.0, 2.0, 4.0, 1.0, 4.0]])
        );
    }

    #[test]
    fn png_dimension_mismatch_reports_both_sizes() {
        let path = temp_path("small.png");
        let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_pixel(4, 3, Luma([255]));
        image.save(&path).unwrap();
        let error = load_roi_mask_png(&path, (6, 8)).unwrap_err().to_string();
        assert!(error.contains("ROI mask is 4×3 but video is 8×6"));
    }
}
