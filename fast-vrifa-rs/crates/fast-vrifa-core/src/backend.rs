use anyhow::Result;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use vrifa_core::roi::RoiMargins;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Wgpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BackendStatus {
    Ready,
    Placeholder,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceShape {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

pub trait ImageBackend: Send + Sync {
    type DeviceFrameBgr;
    type DeviceFrameLab;
    type DevicePlaneF32;
    type DeviceMaskU8;

    fn kind(&self) -> BackendKind;
    fn label(&self) -> &'static str;
    fn status(&self) -> BackendStatus;

    fn upload_frame_bgr(&self, frame_bgr: &Array3<u8>) -> Result<Self::DeviceFrameBgr>;
    fn convert_bgr_to_lab(&self, frame_bgr: &Self::DeviceFrameBgr) -> Result<Self::DeviceFrameLab>;
    fn download_frame_f32(&self, frame_lab: &Self::DeviceFrameLab) -> Result<Array3<f32>>;
    fn build_roi_mask(
        &self,
        shape: (usize, usize),
        margins: RoiMargins,
    ) -> Result<Self::DeviceMaskU8>;
    fn download_mask_u8(&self, mask: &Self::DeviceMaskU8) -> Result<Array2<u8>>;
    fn compute_delta_darken_only(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Array2<f32>,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32>;
    fn download_plane_f32(&self, plane: &Self::DevicePlaneF32) -> Result<Array2<f32>>;
}
