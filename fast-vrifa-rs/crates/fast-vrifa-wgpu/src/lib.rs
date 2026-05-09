use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend};

#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuBackend;

impl ImageBackend for WgpuBackend {
    type DeviceFrame = ();
    type DeviceMask = ();
    type DevicePlane = ();

    fn kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn label(&self) -> &'static str {
        "wgpu-scaffold"
    }

    fn status(&self) -> BackendStatus {
        BackendStatus::Placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::WgpuBackend;
    use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend};

    #[test]
    fn wgpu_backend_is_placeholder_for_scaffold() {
        let backend = WgpuBackend;
        assert_eq!(backend.kind(), BackendKind::Wgpu);
        assert_eq!(backend.status(), BackendStatus::Placeholder);
    }
}
