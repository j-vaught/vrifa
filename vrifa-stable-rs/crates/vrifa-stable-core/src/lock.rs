use crate::{Result, VrifaError};
use ndarray::Array2;

#[derive(Clone, Debug)]
pub struct LockState {
    pub counter: Array2<u16>,
    pub locked: Array2<u8>,
}

impl LockState {
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            counter: Array2::zeros(shape),
            locked: Array2::zeros(shape),
        }
    }
}

pub fn apply_locking(
    mask: &Array2<u8>,
    lock_frames: usize,
    state: Option<&mut LockState>,
) -> Result<Array2<u8>> {
    if lock_frames == 0 {
        return Ok(mask.clone());
    }
    let Some(state) = state else {
        return Ok(mask.clone());
    };
    if state.counter.dim() != mask.dim() || state.locked.dim() != mask.dim() {
        return Err(VrifaError::Shape(
            "lock state shape does not match mask".to_string(),
        ));
    }

    let mut output = mask.clone();
    for ((y, x), value) in mask.indexed_iter() {
        if *value > 0 {
            state.counter[(y, x)] = state.counter[(y, x)].saturating_add(1);
        } else {
            state.counter[(y, x)] = 0;
        }
        if state.counter[(y, x)] >= lock_frames as u16 {
            state.locked[(y, x)] = 255;
        }
        if state.locked[(y, x)] > output[(y, x)] {
            output[(y, x)] = state.locked[(y, x)];
        }
    }
    Ok(output)
}
