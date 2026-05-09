use ndarray::Array2;

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
