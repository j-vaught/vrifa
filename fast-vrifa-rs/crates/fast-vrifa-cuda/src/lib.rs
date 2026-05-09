use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend};

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBackend;

impl ImageBackend for CudaBackend {
    type DeviceFrame = ();
    type DeviceMask = ();
    type DevicePlane = ();

    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn label(&self) -> &'static str {
        "cuda-scaffold"
    }

    fn status(&self) -> BackendStatus {
        BackendStatus::Placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::CudaBackend;
    use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend};

    #[test]
    fn cuda_backend_is_placeholder_for_scaffold() {
        let backend = CudaBackend;
        assert_eq!(backend.kind(), BackendKind::Cuda);
        assert_eq!(backend.status(), BackendStatus::Placeholder);
    }
}
