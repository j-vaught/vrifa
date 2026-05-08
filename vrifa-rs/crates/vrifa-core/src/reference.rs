#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceMode {
    First,
    Running,
    Prev(usize),
    Absolute(usize),
    Dynamic,
}

pub fn compute_dynamic_factor(measurements: &[(f32, f32)]) -> Option<f32> {
    let mut valid: Vec<f32> = measurements
        .iter()
        .filter_map(|(time, area)| (*time > 0.0 && *area > 0.0).then_some(*area / time.powf(1.5)))
        .collect();
    if valid.is_empty() {
        return None;
    }
    valid.sort_by(|a, b| a.total_cmp(b));
    let mid = valid.len() / 2;
    if valid.len() % 2 == 0 {
        Some((valid[mid - 1] + valid[mid]) / 2.0)
    } else {
        Some(valid[mid])
    }
}

#[derive(Clone, Debug)]
pub struct DynamicReferenceParams {
    pub factor: Option<f32>,
    pub target_fraction: f32,
    pub lag_scale: f32,
    pub linear_mode: bool,
    pub linear_start: usize,
    pub linear_max: usize,
    pub total_frames: Option<usize>,
}

pub fn select_dynamic_reference_index(
    frame_index: usize,
    fps: f32,
    roi_pixels: usize,
    params: &DynamicReferenceParams,
) -> usize {
    if frame_index <= 1 {
        return 1;
    }
    if params.linear_mode {
        let progress = params
            .total_frames
            .filter(|total| *total > 0)
            .map(|total| (frame_index.saturating_sub(1)) as f32 / total as f32)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let linear_range = params.linear_max.saturating_sub(params.linear_start);
        let mut delta_frames = params.linear_start as f32 + progress * linear_range as f32;
        delta_frames *= params.lag_scale.max(0.0);
        let delta_frames = (delta_frames.round() as usize).min(frame_index - 1);
        return frame_index.saturating_sub(delta_frames).max(1);
    }

    let Some(factor) = params.factor else {
        return 1;
    };
    if roi_pixels == 0 || fps <= 0.0 {
        return 1;
    }
    let time_current = (frame_index - 1) as f32 / fps;
    let target_area = params.target_fraction.clamp(0.0, 1.0) * roi_pixels as f32;
    let factor = factor.max(1e-9);
    let mut delta_t = ((target_area / factor) + time_current.sqrt()).powi(2) - time_current;
    delta_t *= params.lag_scale.max(0.0);
    let ref_time = (time_current - delta_t.max(0.0)).max(0.0);
    let ref_index = (ref_time * fps) as usize + 1;
    ref_index.clamp(1, frame_index - 1)
}
