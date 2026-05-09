use crate::{cvutil, Result};
use ndarray::Array2;
use opencv::core::{Point, Vector};
use opencv::imgproc;

#[derive(Clone, Debug, PartialEq)]
pub struct AnnotationBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub area: i32,
    pub segmentation: Vec<f32>,
}

pub fn densify_polygon(points: &[(f32, f32)], max_edge_length: f32) -> Vec<(f32, f32)> {
    if max_edge_length <= 0.0 || points.len() < 2 {
        return points.to_vec();
    }
    let mut result = Vec::new();
    for i in 0..points.len() {
        let (x1, y1) = points[i];
        let (x2, y2) = points[(i + 1) % points.len()];
        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist = (dx * dx + dy * dy).sqrt();
        let segments = if dist > 0.0 {
            (dist / max_edge_length).ceil().max(1.0) as usize
        } else {
            1
        };
        for s in 0..segments {
            let t = s as f32 / segments as f32;
            result.push((x1 + dx * t, y1 + dy * t));
        }
    }
    result
}

pub fn extract_bounding_boxes(
    mask: &Array2<u8>,
    tolerance: f32,
    max_edge_length: f32,
) -> Result<Vec<AnnotationBox>> {
    let mask_mat = cvutil::array2_u8_to_mat(mask)?;
    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours_def(
        &mask_mat,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
    )?;

    let mut boxes = Vec::new();
    for contour in contours {
        let area = imgproc::contour_area(&contour, false)? as i32;
        if area <= 0 {
            continue;
        }
        let rect = imgproc::bounding_rect(&contour)?;
        if rect.width <= 0 || rect.height <= 0 {
            continue;
        }

        let approx = if tolerance > 0.0 {
            let mut approx = Vector::<Point>::new();
            imgproc::approx_poly_dp(&contour, &mut approx, tolerance as f64, true)?;
            if approx.is_empty() {
                contour.clone()
            } else {
                approx
            }
        } else {
            contour.clone()
        };

        let mut points: Vec<(f32, f32)> = approx
            .iter()
            .map(|point| (point.x as f32, point.y as f32))
            .collect();
        if max_edge_length > 0.0 {
            points = densify_polygon(&points, max_edge_length);
        }
        let mut segmentation: Vec<f32> = points.iter().flat_map(|(x, y)| [*x, *y]).collect();
        if segmentation.len() < 6 {
            let x = rect.x as f32;
            let y = rect.y as f32;
            let w = rect.width as f32;
            let h = rect.height as f32;
            segmentation = vec![x, y, x + w, y, x + w, y + h, x, y + h];
        }

        boxes.push(AnnotationBox {
            x: rect.x,
            y: rect.y,
            w: rect.width,
            h: rect.height,
            area,
            segmentation,
        });
    }
    Ok(boxes)
}
