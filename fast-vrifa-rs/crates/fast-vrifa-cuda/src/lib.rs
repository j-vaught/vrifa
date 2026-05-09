use fast_vrifa_core::{
    BackendKind, BackendStatus, CpuBackend, CpuFrameBgr, CpuFrameLab, CpuMaskU8, CpuPlaneF32,
    ImageBackend, RoiMargins,
};
use ndarray::{Array2, Array3};

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBackend {
    fallback: CpuBackend,
}

impl ImageBackend for CudaBackend {
    type DeviceFrameBgr = CpuFrameBgr;
    type DeviceFrameLab = CpuFrameLab;
    type DevicePlaneF32 = CpuPlaneF32;
    type DeviceMaskU8 = CpuMaskU8;

    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn label(&self) -> &'static str {
        "cuda-placeholder"
    }

    fn status(&self) -> BackendStatus {
        BackendStatus::Placeholder
    }

    fn upload_frame_bgr(&self, frame_bgr: &Array3<u8>) -> anyhow::Result<Self::DeviceFrameBgr> {
        self.fallback.upload_frame_bgr(frame_bgr)
    }

    fn convert_bgr_to_lab(
        &self,
        frame_bgr: &Self::DeviceFrameBgr,
    ) -> anyhow::Result<Self::DeviceFrameLab> {
        self.fallback.convert_bgr_to_lab(frame_bgr)
    }

    fn download_frame_f32(&self, frame_lab: &Self::DeviceFrameLab) -> anyhow::Result<Array3<f32>> {
        self.fallback.download_frame_f32(frame_lab)
    }

    fn build_roi_mask(
        &self,
        shape: (usize, usize),
        margins: RoiMargins,
    ) -> anyhow::Result<Self::DeviceMaskU8> {
        self.fallback.build_roi_mask(shape, margins)
    }

    fn download_mask_u8(&self, mask: &Self::DeviceMaskU8) -> anyhow::Result<Array2<u8>> {
        self.fallback.download_mask_u8(mask)
    }

    fn compute_delta_darken_only(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Array2<f32>,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> anyhow::Result<Self::DevicePlaneF32> {
        self.fallback.compute_delta_darken_only(
            frame_lab,
            reference_plane,
            roi_mask,
            channel_weight,
        )
    }

    fn download_plane_f32(&self, plane: &Self::DevicePlaneF32) -> anyhow::Result<Array2<f32>> {
        self.fallback.download_plane_f32(plane)
    }
}

#[cfg(test)]
mod tests {
    use super::CudaBackend;
    use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend};

    #[test]
    fn cuda_backend_is_placeholder_for_scaffold() {
        let backend = CudaBackend::default();
        assert_eq!(backend.kind(), BackendKind::Cuda);
        assert_eq!(backend.status(), BackendStatus::Placeholder);
        assert_eq!(backend.label(), "cuda-placeholder");
    }
}
