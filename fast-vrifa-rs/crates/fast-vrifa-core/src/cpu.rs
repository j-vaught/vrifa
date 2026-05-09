use crate::{BackendKind, BackendStatus, ImageBackend, PeakImageBackend};
use anyhow::{Context, Result};
use ndarray::{Array2, Array3};
use vrifa_core::colorspace::{convert_frame_to_colorspace, ColorSpace};
use vrifa_core::peak::update_peak_brightness;
use vrifa_core::roi::{build_roi_mask, RoiMargins};

#[derive(Clone, Debug)]
pub struct CpuFrameBgr {
    pub data: Array3<u8>,
}

#[derive(Clone, Debug)]
pub struct CpuFrameLab {
    pub data: Array3<f32>,
}

#[derive(Clone, Debug)]
pub struct CpuPlaneF32 {
    pub data: Array2<f32>,
}

#[derive(Clone, Debug)]
pub struct CpuMaskU8 {
    pub data: Array2<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl ImageBackend for CpuBackend {
    type DeviceFrameBgr = CpuFrameBgr;
    type DeviceFrameLab = CpuFrameLab;
    type DevicePlaneF32 = CpuPlaneF32;
    type DeviceMaskU8 = CpuMaskU8;

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn label(&self) -> &'static str {
        "cpu"
    }

    fn status(&self) -> BackendStatus {
        BackendStatus::Ready
    }

    fn upload_frame_bgr(&self, frame_bgr: &Array3<u8>) -> Result<Self::DeviceFrameBgr> {
        Ok(CpuFrameBgr {
            data: frame_bgr.clone(),
        })
    }

    fn convert_bgr_to_lab(&self, frame_bgr: &Self::DeviceFrameBgr) -> Result<Self::DeviceFrameLab> {
        let converted = convert_frame_to_colorspace(&frame_bgr.data, ColorSpace::Cielab)
            .context("converting BGR frame to CIELAB on CPU backend")?
            .mapv(|value| value as f32);
        Ok(CpuFrameLab { data: converted })
    }

    fn download_frame_f32(&self, frame_lab: &Self::DeviceFrameLab) -> Result<Array3<f32>> {
        Ok(frame_lab.data.clone())
    }

    fn build_roi_mask(
        &self,
        shape: (usize, usize),
        margins: RoiMargins,
    ) -> Result<Self::DeviceMaskU8> {
        Ok(CpuMaskU8 {
            data: build_roi_mask(shape, margins),
        })
    }

    fn download_mask_u8(&self, mask: &Self::DeviceMaskU8) -> Result<Array2<u8>> {
        Ok(mask.data.clone())
    }

    fn upload_plane_f32(&self, plane: &Array2<f32>) -> Result<Self::DevicePlaneF32> {
        Ok(CpuPlaneF32 {
            data: plane.clone(),
        })
    }

    fn compute_delta_darken_only(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Array2<f32>,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32> {
        let (height, width, channels) = frame_lab.data.dim();
        anyhow::ensure!(
            channels > 0,
            "CIELAB frame must contain at least one channel"
        );
        anyhow::ensure!(
            reference_plane.dim() == (height, width),
            "reference plane shape does not match frame"
        );
        anyhow::ensure!(
            roi_mask.data.dim() == (height, width),
            "ROI mask shape does not match frame"
        );

        let mut delta = Array2::<f32>::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                let raw = (reference_plane[(y, x)] - frame_lab.data[(y, x, 0)]) * channel_weight;
                let value = if raw > 0.0 { raw } else { 0.0 };
                delta[(y, x)] = value * roi_mask.data[(y, x)] as f32;
            }
        }

        Ok(CpuPlaneF32 { data: delta })
    }

    fn download_plane_f32(&self, plane: &Self::DevicePlaneF32) -> Result<Array2<f32>> {
        Ok(plane.data.clone())
    }
}

impl PeakImageBackend for CpuBackend {
    fn extract_l_plane(&self, frame_lab: &Self::DeviceFrameLab) -> Result<Self::DevicePlaneF32> {
        Ok(CpuPlaneF32 {
            data: frame_lab.data.slice(ndarray::s![.., .., 0]).to_owned(),
        })
    }

    fn update_peak_brightness_device(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        previous_peak: Option<&Self::DevicePlaneF32>,
    ) -> Result<Self::DevicePlaneF32> {
        Ok(CpuPlaneF32 {
            data: update_peak_brightness(&frame_lab.data, previous_peak.map(|plane| &plane.data))?,
        })
    }

    fn compute_delta_darken_only_device(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Self::DevicePlaneF32,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32> {
        self.compute_delta_darken_only(frame_lab, &reference_plane.data, roi_mask, channel_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::CpuBackend;
    use crate::{BackendKind, BackendStatus, ImageBackend};
    use ndarray::array;
    use vrifa_core::roi::RoiMargins;

    #[test]
    fn cpu_backend_reports_ready_status() {
        let backend = CpuBackend;
        assert_eq!(backend.kind(), BackendKind::Cpu);
        assert_eq!(backend.status(), BackendStatus::Ready);
        assert_eq!(backend.label(), "cpu");
    }

    #[test]
    fn cpu_backend_builds_roi_mask_and_delta() {
        let backend = CpuBackend;
        let frame = array![[[10u8, 20u8, 30u8], [40u8, 50u8, 60u8]]];
        let uploaded = backend.upload_frame_bgr(&frame).unwrap();
        let lab = backend.convert_bgr_to_lab(&uploaded).unwrap();
        let mask = backend
            .build_roi_mask(
                (1, 2),
                RoiMargins {
                    top: 0.0,
                    bottom: 0.0,
                    left: 0.0,
                    right: 0.0,
                },
            )
            .unwrap();
        let reference = array![[255.0f32, 255.0f32]];
        let delta = backend
            .compute_delta_darken_only(&lab, &reference, &mask, 1.0)
            .unwrap();
        assert_eq!(backend.download_mask_u8(&mask).unwrap(), array![[1u8, 1u8]]);
        assert!(backend
            .download_plane_f32(&delta)
            .unwrap()
            .iter()
            .all(|value| *value >= 0.0));
    }
}
