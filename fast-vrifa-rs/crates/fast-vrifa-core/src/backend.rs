use anyhow::Result;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use vrifa_core::morphology::MorphShape;
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

pub trait PeakImageBackend: ImageBackend {
    fn extract_l_plane(&self, frame_lab: &Self::DeviceFrameLab) -> Result<Self::DevicePlaneF32>;
    fn update_peak_brightness_device(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        previous_peak: Option<&Self::DevicePlaneF32>,
    ) -> Result<Self::DevicePlaneF32>;
    fn compute_delta_darken_only_device(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Self::DevicePlaneF32,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32>;

    fn blur_and_normalize_delta(
        &self,
        delta: &Self::DevicePlaneF32,
        blur_kernel: usize,
        blur_enabled: bool,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let _ = (delta, blur_kernel, blur_enabled);
        Ok(None)
    }

    fn threshold_and_morph_mask(
        &self,
        delta_norm: &Self::DeviceMaskU8,
        threshold_value: f32,
        morph_shape: MorphShape,
        morph_kernel: usize,
        morph_close_iterations: usize,
        morph_open_iterations: usize,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let _ = (
            delta_norm,
            threshold_value,
            morph_shape,
            morph_kernel,
            morph_close_iterations,
            morph_open_iterations,
        );
        Ok(None)
    }
}
