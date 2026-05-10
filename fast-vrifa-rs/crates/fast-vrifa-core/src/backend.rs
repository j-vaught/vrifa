use anyhow::Result;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use vrifa_core::delta::blur_plane;
use vrifa_core::morphology::MorphShape;
use vrifa_core::peak::update_peak_brightness_plane;
use vrifa_core::roi::RoiMargins;
use vrifa_core::warp::{apply_warp_plane, AffineWarp};

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
    type DeviceLockState;

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
    fn upload_mask_u8(&self, mask: &Array2<u8>) -> Result<Self::DeviceMaskU8>;
    fn download_mask_u8(&self, mask: &Self::DeviceMaskU8) -> Result<Array2<u8>>;
    fn upload_plane_f32(&self, plane: &Array2<f32>) -> Result<Self::DevicePlaneF32>;
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

    fn blur_plane_f32_device(
        &self,
        plane: &Self::DevicePlaneF32,
        kernel: usize,
    ) -> Result<Self::DevicePlaneF32> {
        let plane = self.download_plane_f32(plane)?;
        let blurred = blur_plane(&plane, kernel)?;
        self.upload_plane_f32(&blurred)
    }

    fn warp_plane_f32_device(
        &self,
        plane: &Self::DevicePlaneF32,
        matrix: &AffineWarp,
    ) -> Result<Self::DevicePlaneF32> {
        let plane = self.download_plane_f32(plane)?;
        let warped = apply_warp_plane(&plane, matrix)?;
        self.upload_plane_f32(&warped)
    }

    fn update_peak_brightness_plane_device(
        &self,
        frame_l: &Self::DevicePlaneF32,
        previous_peak: Option<&Self::DevicePlaneF32>,
    ) -> Result<Self::DevicePlaneF32> {
        let frame_l = self.download_plane_f32(frame_l)?;
        let previous_peak = previous_peak
            .map(|plane| self.download_plane_f32(plane))
            .transpose()?;
        let peak = update_peak_brightness_plane(&frame_l, previous_peak.as_ref())?;
        self.upload_plane_f32(&peak)
    }

    fn compute_delta_darken_only_planes_device(
        &self,
        frame_l: &Self::DevicePlaneF32,
        reference_plane: &Self::DevicePlaneF32,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32> {
        let frame_l = self.download_plane_f32(frame_l)?;
        let reference_plane = self.download_plane_f32(reference_plane)?;
        let roi_mask = self.download_mask_u8(roi_mask)?;
        anyhow::ensure!(
            frame_l.dim() == reference_plane.dim(),
            "reference plane shape does not match frame"
        );
        anyhow::ensure!(
            frame_l.dim() == roi_mask.dim(),
            "ROI mask shape does not match frame"
        );

        let (height, width) = frame_l.dim();
        let mut delta = Array2::<f32>::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                let raw = (reference_plane[(y, x)] - frame_l[(y, x)]) * channel_weight;
                let value = if raw > 0.0 { raw } else { 0.0 };
                delta[(y, x)] = value * roi_mask[(y, x)] as f32;
            }
        }

        self.upload_plane_f32(&delta)
    }

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

    fn threshold_and_morph_mask_auto(
        &self,
        delta_norm: &Self::DeviceMaskU8,
        threshold_offset: f32,
        morph_shape: MorphShape,
        morph_kernel: usize,
        morph_close_iterations: usize,
        morph_open_iterations: usize,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let _ = (
            delta_norm,
            threshold_offset,
            morph_shape,
            morph_kernel,
            morph_close_iterations,
            morph_open_iterations,
        );
        Ok(None)
    }

    fn filter_min_area_mask(
        &self,
        mask: &Self::DeviceMaskU8,
        min_area: usize,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let _ = (mask, min_area);
        Ok(None)
    }

    fn create_lock_state(&self, shape: (usize, usize)) -> Result<Option<Self::DeviceLockState>> {
        let _ = shape;
        Ok(None)
    }

    fn apply_locking_device(
        &self,
        mask: &Self::DeviceMaskU8,
        lock_frames: usize,
        state: &mut Self::DeviceLockState,
    ) -> Result<Option<Self::DeviceMaskU8>> {
        let _ = (mask, lock_frames, state);
        Ok(None)
    }
}
