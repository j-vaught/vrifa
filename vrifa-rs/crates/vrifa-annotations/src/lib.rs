use anyhow::{Context, Result};
use ndarray::Array3;
use rayon::prelude::*;
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use vrifa_core::AnnotationBox;

#[derive(Clone, Debug)]
pub struct AnnotationFrame {
    pub frame_index: usize,
    pub frame_bgr: Option<Array3<u8>>,
    pub boxes: Vec<AnnotationBox>,
}

pub fn export_annotation_outputs(
    output_dir: impl AsRef<Path>,
    records: &[AnnotationFrame],
    selected_indices: &[usize],
    width: usize,
    height: usize,
    formats: &[String],
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    if records.is_empty() || selected_indices.is_empty() || formats.is_empty() {
        return Ok(());
    }
    if formats.iter().any(|fmt| fmt == "coco") {
        coco::export(output_dir, records, selected_indices, width, height)?;
    }
    if formats.iter().any(|fmt| fmt == "yolov5") {
        yolov5::export(output_dir, records, selected_indices, width, height)?;
    }
    if formats.iter().any(|fmt| fmt == "darknet") {
        darknet::export(output_dir, records, selected_indices, width, height)?;
    }
    Ok(())
}

fn ensure_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))
}

fn write_bgr_png(path: &Path, frame: &Array3<u8>) -> Result<()> {
    let (height, width, channels) = frame.dim();
    anyhow::ensure!(channels == 3, "annotation image must be BGR");
    let mut img = image::RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            img.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([frame[(y, x, 2)], frame[(y, x, 1)], frame[(y, x, 0)]]),
            );
        }
    }
    img.save(path)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub mod coco {
    use super::*;

    #[derive(Serialize)]
    struct CocoOutput {
        licenses: Vec<License>,
        info: Info,
        categories: Vec<Category>,
        images: Vec<Image>,
        annotations: Vec<Annotation>,
    }

    #[derive(Serialize)]
    struct License {
        name: &'static str,
        id: i32,
        url: &'static str,
    }

    #[derive(Serialize)]
    struct Info {
        contributor: &'static str,
        date_created: &'static str,
        description: &'static str,
        url: &'static str,
        version: &'static str,
        year: &'static str,
    }

    #[derive(Serialize)]
    struct Category {
        id: i32,
        name: &'static str,
        supercategory: &'static str,
    }

    #[derive(Serialize)]
    struct Image {
        id: usize,
        width: usize,
        height: usize,
        file_name: String,
        license: i32,
        flickr_url: &'static str,
        coco_url: &'static str,
        date_captured: i32,
    }

    #[derive(Serialize)]
    struct Attributes {
        occluded: bool,
        rotation: f32,
    }

    #[derive(Serialize)]
    struct Annotation {
        id: usize,
        image_id: usize,
        category_id: i32,
        segmentation: Vec<Vec<f32>>,
        area: i32,
        bbox: [i32; 4],
        iscrowd: i32,
        attributes: Attributes,
    }

    pub fn export(
        output_dir: &Path,
        records: &[AnnotationFrame],
        selected_indices: &[usize],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let coco_root = output_dir.join("formatCOCO");
        let annotations_dir = coco_root.join("annotations");
        let images_dir = coco_root.join("images").join("default");
        ensure_dir(&annotations_dir)?;
        ensure_dir(&images_dir)?;

        let mut output = CocoOutput {
            licenses: vec![License {
                name: "",
                id: 0,
                url: "",
            }],
            info: Info {
                contributor: "",
                date_created: "",
                description: "",
                url: "",
                version: "",
                year: "",
            },
            categories: vec![
                Category {
                    id: 1,
                    name: "dry",
                    supercategory: "",
                },
                Category {
                    id: 2,
                    name: "wet",
                    supercategory: "",
                },
            ],
            images: Vec::new(),
            annotations: Vec::new(),
        };

        let selected_records: Vec<&AnnotationFrame> = selected_indices
            .iter()
            .copied()
            .map(|index| &records[index])
            .collect();
        selected_records.par_iter().try_for_each(|record| {
            if let Some(frame_bgr) = record.frame_bgr.as_ref() {
                let frame_filename = format!("frame_{:06}.png", record.frame_index);
                write_bgr_png(&images_dir.join(frame_filename), frame_bgr)?;
            }
            Ok::<(), anyhow::Error>(())
        })?;

        let mut annotation_id = 1usize;
        for (image_offset, record) in selected_records.iter().enumerate() {
            let image_id = image_offset + 1;
            let frame_filename = format!("frame_{:06}.png", record.frame_index);

            output.images.push(Image {
                id: image_id,
                width,
                height,
                file_name: frame_filename,
                license: 0,
                flickr_url: "",
                coco_url: "",
                date_captured: 0,
            });

            for bbox in &record.boxes {
                output.annotations.push(Annotation {
                    id: annotation_id,
                    image_id,
                    category_id: 1,
                    segmentation: if bbox.segmentation.is_empty() {
                        Vec::new()
                    } else {
                        vec![bbox.segmentation.clone()]
                    },
                    area: bbox.area,
                    bbox: [bbox.x, bbox.y, bbox.w, bbox.h],
                    iscrowd: 0,
                    attributes: Attributes {
                        occluded: false,
                        rotation: 0.0,
                    },
                });
                annotation_id += 1;
            }
        }

        let path = annotations_dir.join("instances_default.json");
        let file = File::create(&path).with_context(|| format!("creating {}", path.display()))?;
        serde_json::to_writer(file, &output)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

pub mod yolov5 {
    use super::*;

    pub fn export(
        output_dir: &Path,
        records: &[AnnotationFrame],
        selected_indices: &[usize],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let yolo_root = output_dir.join("formatYOLO");
        let images_dir = yolo_root.join("images").join("train");
        let labels_dir = yolo_root.join("labels").join("train");
        ensure_dir(&images_dir)?;
        ensure_dir(&labels_dir)?;

        let mut train_list = Vec::new();
        for record_index in selected_indices.iter().copied() {
            let record = &records[record_index];
            let frame_filename = format!("frame_{:06}.png", record.frame_index);
            if let Some(frame_bgr) = record.frame_bgr.as_ref() {
                write_bgr_png(&images_dir.join(&frame_filename), frame_bgr)?;
            }
            train_list.push(format!("data/images/train/{frame_filename}"));

            let mut label_file =
                File::create(labels_dir.join(format!("frame_{:06}.txt", record.frame_index)))?;
            for bbox in &record.boxes {
                if bbox.segmentation.len() >= 6 {
                    let mut parts = Vec::with_capacity(bbox.segmentation.len());
                    for pair in bbox.segmentation.chunks_exact(2) {
                        parts.push(format!("{:.6}", pair[0] / width as f32));
                        parts.push(format!("{:.6}", pair[1] / height as f32));
                    }
                    writeln!(label_file, "0 {}", parts.join(" "))?;
                } else {
                    let x = bbox.x as f32;
                    let y = bbox.y as f32;
                    let w = bbox.w as f32;
                    let h = bbox.h as f32;
                    writeln!(
                        label_file,
                        "0 {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                        x / width as f32,
                        y / height as f32,
                        (x + w) / width as f32,
                        y / height as f32,
                        (x + w) / width as f32,
                        (y + h) / height as f32,
                        x / width as f32,
                        (y + h) / height as f32
                    )?;
                }
            }
        }

        fs::write(
            yolo_root.join("data.yaml"),
            "names:\n  0: dry\n  1: wet\npath: .\ntrain: train.txt\n",
        )?;
        let mut train = File::create(yolo_root.join("train.txt"))?;
        for entry in train_list {
            writeln!(train, "{entry}")?;
        }
        Ok(())
    }
}

pub mod darknet {
    use super::*;

    pub fn export(
        output_dir: &Path,
        records: &[AnnotationFrame],
        selected_indices: &[usize],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let darknet_root = output_dir.join("formatYOLO_v2");
        let obj_train_data = darknet_root.join("obj_train_data");
        ensure_dir(&obj_train_data)?;

        let mut train_list = Vec::new();
        for record_index in selected_indices.iter().copied() {
            let record = &records[record_index];
            let frame_filename = format!("frame_{:06}.png", record.frame_index);
            if let Some(frame_bgr) = record.frame_bgr.as_ref() {
                write_bgr_png(&obj_train_data.join(&frame_filename), frame_bgr)?;
            }
            train_list.push(format!("data/obj_train_data/{frame_filename}"));

            let mut label_file =
                File::create(obj_train_data.join(format!("frame_{:06}.txt", record.frame_index)))?;
            for bbox in &record.boxes {
                let cx = (bbox.x as f32 + bbox.w as f32 / 2.0) / width as f32;
                let cy = (bbox.y as f32 + bbox.h as f32 / 2.0) / height as f32;
                let nw = bbox.w as f32 / width as f32;
                let nh = bbox.h as f32 / height as f32;
                writeln!(label_file, "0 {cx:.6} {cy:.6} {nw:.6} {nh:.6}")?;
            }
        }

        fs::write(
            darknet_root.join("obj.data"),
            "classes = 2\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n",
        )?;
        fs::write(darknet_root.join("obj.names"), "dry\nwet\n")?;
        let mut train = File::create(darknet_root.join("train.txt"))?;
        for entry in train_list {
            writeln!(train, "{entry}")?;
        }
        Ok(())
    }
}
