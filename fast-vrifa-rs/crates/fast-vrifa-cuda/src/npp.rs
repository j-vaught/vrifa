#[cfg(target_os = "linux")]
mod imp {
    use anyhow::{anyhow, bail, Context, Result};
    use cudarc::driver::sys::CUstream;
    use libloading::Library;
    use std::env;
    use std::ffi::c_void;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    pub type NppStatus = i32;
    pub const NPP_SUCCESS: NppStatus = 0;
    pub const NPPI_NORM_INF: i32 = 0;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct NppiSize {
        pub width: i32,
        pub height: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct NppiPoint {
        pub x: i32,
        pub y: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct NppiRect {
        pub x: i32,
        pub y: i32,
        pub width: i32,
        pub height: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct NppStreamContext {
        pub h_stream: CUstream,
        pub n_cuda_device_id: i32,
        pub n_multi_processor_count: i32,
        pub n_max_threads_per_multi_processor: i32,
        pub n_max_threads_per_block: i32,
        pub n_shared_mem_per_block: usize,
        pub n_cuda_dev_attr_compute_capability_major: i32,
        pub n_cuda_dev_attr_compute_capability_minor: i32,
        pub n_stream_flags: u32,
        pub n_reserved0: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct NppiCompressedMarkerLabelsInfo {
        pub n_marker_label_pixel_count: u32,
        pub n_contour_pixel_count: u32,
        pub n_contour_pixels_found: u32,
        pub o_contour_first_pixel_location: NppiPoint,
        pub o_marker_label_bounding_box: NppiRect,
    }

    type NppSetStream = unsafe extern "C" fn(CUstream) -> NppStatus;
    type NppGetStreamContext = unsafe extern "C" fn(*mut NppStreamContext) -> NppStatus;
    type NppiLabelMarkersUFGetBufferSize32u = unsafe extern "C" fn(NppiSize, *mut i32) -> NppStatus;
    type NppiLabelMarkersUF8u32u = unsafe extern "C" fn(
        *mut u8,
        i32,
        *mut u32,
        i32,
        NppiSize,
        i32,
        *mut u8,
        NppStreamContext,
    ) -> NppStatus;
    type NppiCompressMarkerLabelsGetBufferSize32u =
        unsafe extern "C" fn(i32, *mut i32) -> NppStatus;
    type NppiCompressMarkerLabelsUF32u = unsafe extern "C" fn(
        *mut u32,
        i32,
        NppiSize,
        i32,
        *mut i32,
        *mut u8,
        NppStreamContext,
    ) -> NppStatus;
    type NppiCompressedMarkerLabelsUFGetInfoListSize32u =
        unsafe extern "C" fn(u32, *mut u32) -> NppStatus;
    type NppiCompressedMarkerLabelsUFInfo32u = unsafe extern "C" fn(
        *mut u32,
        i32,
        NppiSize,
        u32,
        *mut NppiCompressedMarkerLabelsInfo,
        *mut u8,
        i32,
        *mut c_void,
        i32,
        *mut c_void,
        *mut u32,
        *mut u32,
        *mut u32,
        *mut u32,
        NppStreamContext,
    ) -> NppStatus;

    pub struct NppLibrary {
        _core: Library,
        _image: Library,
        pub npp_set_stream: NppSetStream,
        pub npp_get_stream_context: NppGetStreamContext,
        pub label_buffer_size: NppiLabelMarkersUFGetBufferSize32u,
        pub label_markers: NppiLabelMarkersUF8u32u,
        pub compress_buffer_size: NppiCompressMarkerLabelsGetBufferSize32u,
        pub compress_markers: NppiCompressMarkerLabelsUF32u,
        pub info_list_size: NppiCompressedMarkerLabelsUFGetInfoListSize32u,
        pub info_list: NppiCompressedMarkerLabelsUFInfo32u,
    }

    impl NppLibrary {
        pub fn load() -> Result<Arc<Self>> {
            let candidates = npp_library_candidates();
            let mut last_error = None;
            for base in candidates {
                match Self::load_from_base(&base) {
                    Ok(lib) => return Ok(Arc::new(lib)),
                    Err(err) => last_error = Some(err),
                }
            }
            Err(last_error.unwrap_or_else(|| anyhow!("unable to locate NPP shared libraries")))
        }

        fn load_from_base(base: &Path) -> Result<Self> {
            let core_path = base.join("libnppc.so");
            let image_path = base.join("libnppif.so");
            let core = unsafe { Library::new(&core_path) }
                .with_context(|| format!("loading {}", core_path.display()))?;
            let image = unsafe { Library::new(&image_path) }
                .with_context(|| format!("loading {}", image_path.display()))?;
            let library = unsafe {
                Self {
                    npp_set_stream: *core.get(b"nppSetStream\0")?,
                    npp_get_stream_context: *core.get(b"nppGetStreamContext\0")?,
                    label_buffer_size: *image.get(b"nppiLabelMarkersUFGetBufferSize_32u_C1R\0")?,
                    label_markers: *image.get(b"nppiLabelMarkersUF_8u32u_C1R_Ctx\0")?,
                    compress_buffer_size: *image
                        .get(b"nppiCompressMarkerLabelsGetBufferSize_32u_C1R\0")?,
                    compress_markers: *image.get(b"nppiCompressMarkerLabelsUF_32u_C1IR_Ctx\0")?,
                    info_list_size: *image
                        .get(b"nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R\0")?,
                    info_list: *image.get(b"nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx\0")?,
                    _core: core,
                    _image: image,
                }
            };
            Ok(library)
        }

        pub fn set_stream(&self, stream: CUstream) -> Result<()> {
            status(unsafe { (self.npp_set_stream)(stream) }, "nppSetStream")
        }

        pub fn stream_context(&self) -> Result<NppStreamContext> {
            let mut ctx = NppStreamContext::default();
            status(
                unsafe { (self.npp_get_stream_context)(&mut ctx as *mut _) },
                "nppGetStreamContext",
            )?;
            Ok(ctx)
        }
    }

    pub fn status(code: NppStatus, label: &str) -> Result<()> {
        if code == NPP_SUCCESS {
            Ok(())
        } else {
            bail!("{label} failed with NPP status code {code}")
        }
    }

    fn npp_library_candidates() -> Vec<PathBuf> {
        let mut bases = Vec::new();
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            bases.push(PathBuf::from(cuda_path).join("lib64"));
        }
        if let Ok(home) = env::var("HOME") {
            bases.push(PathBuf::from(home.clone()).join("cuda-12.4/lib64"));
            bases.push(PathBuf::from(home).join("cuda/lib64"));
        }
        bases.push(PathBuf::from("/usr/local/cuda/lib64"));
        bases.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
        bases
    }
}

#[cfg(target_os = "linux")]
pub use imp::*;

#[cfg(not(target_os = "linux"))]
mod imp {
    use anyhow::{bail, Result};
    use std::sync::Arc;

    pub struct NppLibrary;
    impl NppLibrary {
        pub fn load() -> Result<Arc<Self>> {
            bail!("NPP is only supported on Linux CUDA hosts")
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub use imp::*;
