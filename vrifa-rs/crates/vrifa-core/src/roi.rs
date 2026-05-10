use ndarray::{Array2, Zip};

#[derive(Clone, Copy, Debug)]
pub struct RoiMargins {
    pub top: f32,
    pub bottom: f32,
    pub left: f32,
    pub right: f32,
}

pub fn resolve_roi_margins(
    roi_margin: f32,
    top: Option<f32>,
    bottom: Option<f32>,
    left: Option<f32>,
    right: Option<f32>,
) -> RoiMargins {
    fn pick(default: f32, value: Option<f32>) -> f32 {
        value.unwrap_or(default).clamp(0.0, 0.49)
    }

    RoiMargins {
        top: pick(roi_margin, top),
        bottom: pick(roi_margin, bottom),
        left: pick(roi_margin, left),
        right: pick(roi_margin, right),
    }
}

pub fn build_roi_mask(shape: (usize, usize), margins: RoiMargins) -> Array2<u8> {
    build_roi_mask_with_override(shape, margins, None)
}

pub fn build_roi_mask_with_override(
    shape: (usize, usize),
    margins: RoiMargins,
    prebuilt: Option<&Array2<u8>>,
) -> Array2<u8> {
    if let Some(mask) = prebuilt {
        return mask.clone();
    }
    let (height, width) = shape;
    let top = (margins.top * height as f32) as usize;
    let mut bottom = height.saturating_sub((margins.bottom * height as f32) as usize);
    let left = (margins.left * width as f32) as usize;
    let mut right = width.saturating_sub((margins.right * width as f32) as usize);

    if bottom <= top {
        bottom = height.min(top + 1);
    }
    if right <= left {
        right = width.min(left + 1);
    }

    let mut mask = Array2::<u8>::zeros((height, width));
    if top < bottom && left < right {
        mask.slice_mut(ndarray::s![top..bottom, left..right])
            .fill(1);
    }
    mask
}

pub fn clip_mask_to_roi(mask: &mut Array2<u8>, roi_mask: &Array2<u8>) {
    assert_eq!(
        mask.dim(),
        roi_mask.dim(),
        "mask and ROI shape must match for clipping"
    );
    Zip::from(mask).and(roi_mask).for_each(|pixel, &roi| {
        if roi == 0 {
            *pixel = 0;
        }
    });
}

pub fn is_rectangular_roi_mask(mask: &Array2<u8>) -> bool {
    let mut min_y = usize::MAX;
    let mut max_y = 0usize;
    let mut min_x = usize::MAX;
    let mut max_x = 0usize;
    let mut found = false;

    for ((y, x), &value) in mask.indexed_iter() {
        if value == 0 {
            continue;
        }
        found = true;
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
    }

    if !found {
        return false;
    }

    for ((y, x), &value) in mask.indexed_iter() {
        let inside = y >= min_y && y <= max_y && x >= min_x && x <= max_x;
        if inside != (value > 0) {
            return false;
        }
    }
    true
}
