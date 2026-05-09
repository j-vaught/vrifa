use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BackendKind {
    DelegatedCpu,
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
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

pub trait ImageBackend: Send + Sync {
    type DeviceFrame;
    type DeviceMask;
    type DevicePlane;

    fn kind(&self) -> BackendKind;
    fn label(&self) -> &'static str;
    fn status(&self) -> BackendStatus;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DelegatedCpuBackend;

impl ImageBackend for DelegatedCpuBackend {
    type DeviceFrame = ();
    type DeviceMask = ();
    type DevicePlane = ();

    fn kind(&self) -> BackendKind {
        BackendKind::DelegatedCpu
    }

    fn label(&self) -> &'static str {
        "delegated-cpu"
    }

    fn status(&self) -> BackendStatus {
        BackendStatus::Ready
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendKind, BackendStatus, DelegatedCpuBackend, ImageBackend};

    #[test]
    fn delegated_backend_reports_ready_status() {
        let backend = DelegatedCpuBackend;
        assert_eq!(backend.kind(), BackendKind::DelegatedCpu);
        assert_eq!(backend.status(), BackendStatus::Ready);
        assert_eq!(backend.label(), "delegated-cpu");
    }
}
