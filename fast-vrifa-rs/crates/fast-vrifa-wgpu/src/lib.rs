use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use fast_vrifa_core::{BackendKind, BackendStatus, ImageBackend, RoiMargins};
use ndarray::{Array2, Array3};
use pollster::block_on;
use std::borrow::Cow;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::sync_channel;
use vrifa_core::colorspace::{convert_frame_to_colorspace, ColorSpace};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

pub struct WgpuFrameBgr {
    buffer: wgpu::Buffer,
    width: usize,
    height: usize,
}

pub struct WgpuFrameLab {
    buffer: wgpu::Buffer,
    width: usize,
    height: usize,
}

pub struct WgpuPlaneF32 {
    buffer: wgpu::Buffer,
    width: usize,
    height: usize,
}

pub struct WgpuMaskU8 {
    buffer: wgpu::Buffer,
    width: usize,
    height: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PixelCountUniform {
    pixel_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RoiUniform {
    width: u32,
    height: u32,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DeltaUniform {
    pixel_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    channel_weight: f32,
    _padf0: f32,
    _padf1: f32,
    _padf2: f32,
}

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    lab_lut: wgpu::Buffer,
    colorspace_layout: wgpu::BindGroupLayout,
    roi_layout: wgpu::BindGroupLayout,
    delta_layout: wgpu::BindGroupLayout,
    colorspace_pipeline: wgpu::ComputePipeline,
    roi_pipeline: wgpu::ComputePipeline,
    delta_pipeline: wgpu::ComputePipeline,
}

impl WgpuBackend {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("unable to acquire a wgpu adapter"))?;

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fast-vrifa-wgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .context("requesting wgpu device")?;

        let lab_lut_host = load_or_build_lab_lut()?;
        let lab_lut = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fast-vrifa-bgr2lab-lut"),
            contents: bytemuck::cast_slice(&lab_lut_host),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let colorspace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fast-vrifa-colorspace-layout"),
            entries: &[
                storage_buffer_entry(0, true),
                storage_buffer_entry(1, true),
                storage_buffer_entry(2, false),
                uniform_buffer_entry(3),
            ],
        });
        let roi_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fast-vrifa-roi-layout"),
            entries: &[storage_buffer_entry(0, false), uniform_buffer_entry(1)],
        });
        let delta_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fast-vrifa-delta-layout"),
            entries: &[
                storage_buffer_entry(0, true),
                storage_buffer_entry(1, true),
                storage_buffer_entry(2, true),
                storage_buffer_entry(3, false),
                uniform_buffer_entry(4),
            ],
        });

        let colorspace_pipeline = create_compute_pipeline(
            &device,
            "fast-vrifa-colorspace-pipeline",
            &colorspace_layout,
            include_str!("shaders/colorspace_bgr_to_lab.wgsl"),
        );
        let roi_pipeline = create_compute_pipeline(
            &device,
            "fast-vrifa-roi-pipeline",
            &roi_layout,
            include_str!("shaders/roi_mask.wgsl"),
        );
        let delta_pipeline = create_compute_pipeline(
            &device,
            "fast-vrifa-delta-pipeline",
            &delta_layout,
            include_str!("shaders/delta_darken_only.wgsl"),
        );

        Ok(Self {
            device,
            queue,
            lab_lut,
            colorspace_layout,
            roi_layout,
            delta_layout,
            colorspace_pipeline,
            roi_pipeline,
            delta_pipeline,
        })
    }

    fn create_storage_buffer(&self, label: &'static str, byte_len: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn dispatch_1d(
        &self,
        label: &'static str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        pixel_count: usize,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let groups = ((pixel_count as u32).saturating_add(WORKGROUP_SIZE - 1)) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn readback_bytes(&self, source: &wgpu::Buffer, byte_len: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fast-vrifa-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fast-vrifa-readback-copy"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| anyhow!("wgpu readback callback dropped"))?
            .context("mapping wgpu readback buffer")?;
        let bytes = slice.get_mapped_range().to_vec();
        let _ = slice;
        staging.unmap();
        Ok(bytes)
    }
}

impl ImageBackend for WgpuBackend {
    type DeviceFrameBgr = WgpuFrameBgr;
    type DeviceFrameLab = WgpuFrameLab;
    type DevicePlaneF32 = WgpuPlaneF32;
    type DeviceMaskU8 = WgpuMaskU8;

    fn kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn label(&self) -> &'static str {
        "wgpu"
    }

    fn status(&self) -> BackendStatus {
        BackendStatus::Ready
    }

    fn upload_frame_bgr(&self, frame_bgr: &Array3<u8>) -> Result<Self::DeviceFrameBgr> {
        let (height, width, channels) = frame_bgr.dim();
        anyhow::ensure!(channels == 3, "expected a 3-channel BGR frame");
        let packed = pack_bgr_pixels(frame_bgr)?;
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fast-vrifa-bgr-frame"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        Ok(WgpuFrameBgr {
            buffer,
            width,
            height,
        })
    }

    fn convert_bgr_to_lab(&self, frame_bgr: &Self::DeviceFrameBgr) -> Result<Self::DeviceFrameLab> {
        let pixel_count = frame_bgr.width * frame_bgr.height;
        let output =
            self.create_storage_buffer("fast-vrifa-lab-frame", byte_len_for_u32(pixel_count));
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fast-vrifa-colorspace-uniform"),
                contents: bytemuck::bytes_of(&PixelCountUniform {
                    pixel_count: pixel_count as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fast-vrifa-colorspace-bind-group"),
            layout: &self.colorspace_layout,
            entries: &[
                storage_buffer_binding(0, &frame_bgr.buffer),
                storage_buffer_binding(1, &self.lab_lut),
                storage_buffer_binding(2, &output),
                uniform_buffer_binding(3, &uniform),
            ],
        });
        self.dispatch_1d(
            "fast-vrifa-colorspace-dispatch",
            &self.colorspace_pipeline,
            &bind_group,
            pixel_count,
        );
        Ok(WgpuFrameLab {
            buffer: output,
            width: frame_bgr.width,
            height: frame_bgr.height,
        })
    }

    fn download_frame_f32(&self, frame_lab: &Self::DeviceFrameLab) -> Result<Array3<f32>> {
        let pixel_count = frame_lab.width * frame_lab.height;
        let bytes = self.readback_bytes(&frame_lab.buffer, byte_len_for_u32(pixel_count))?;
        let values = bytes_to_u32(&bytes)?
            .into_iter()
            .flat_map(|packed| {
                [
                    (packed & 0xff) as f32,
                    ((packed >> 8) & 0xff) as f32,
                    ((packed >> 16) & 0xff) as f32,
                ]
            })
            .collect::<Vec<_>>();
        Array3::from_shape_vec((frame_lab.height, frame_lab.width, 3), values)
            .context("reshaping downloaded CIELAB frame")
    }

    fn build_roi_mask(
        &self,
        shape: (usize, usize),
        margins: RoiMargins,
    ) -> Result<Self::DeviceMaskU8> {
        let (height, width) = shape;
        let top = (margins.top * height as f32) as usize;
        let mut bottom = height.saturating_sub((margins.bottom * height as f32) as usize);
        let left = (margins.left * width as f32) as usize;
        let mut right = width.saturating_sub((margins.right * width as f32) as usize);
        if bottom <= top {
            bottom = height.min(top + 1);
        }
        if right <= left {
            right = width.min(left + 1);
        }

        let pixel_count = width * height;
        let output =
            self.create_storage_buffer("fast-vrifa-roi-mask", byte_len_for_u32(pixel_count));
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fast-vrifa-roi-uniform"),
                contents: bytemuck::bytes_of(&RoiUniform {
                    width: width as u32,
                    height: height as u32,
                    top: top as u32,
                    bottom: bottom as u32,
                    left: left as u32,
                    right: right as u32,
                    _pad0: 0,
                    _pad1: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fast-vrifa-roi-bind-group"),
            layout: &self.roi_layout,
            entries: &[
                storage_buffer_binding(0, &output),
                uniform_buffer_binding(1, &uniform),
            ],
        });
        self.dispatch_1d(
            "fast-vrifa-roi-dispatch",
            &self.roi_pipeline,
            &bind_group,
            pixel_count,
        );
        Ok(WgpuMaskU8 {
            buffer: output,
            width,
            height,
        })
    }

    fn download_mask_u8(&self, mask: &Self::DeviceMaskU8) -> Result<Array2<u8>> {
        let pixel_count = mask.width * mask.height;
        let bytes = self.readback_bytes(&mask.buffer, byte_len_for_u32(pixel_count))?;
        let values = bytes_to_u32(&bytes)?
            .into_iter()
            .map(|value| if value == 0 { 0u8 } else { 1u8 })
            .collect::<Vec<_>>();
        Array2::from_shape_vec((mask.height, mask.width), values).context("reshaping ROI mask")
    }

    fn compute_delta_darken_only(
        &self,
        frame_lab: &Self::DeviceFrameLab,
        reference_plane: &Array2<f32>,
        roi_mask: &Self::DeviceMaskU8,
        channel_weight: f32,
    ) -> Result<Self::DevicePlaneF32> {
        anyhow::ensure!(
            reference_plane.dim() == (frame_lab.height, frame_lab.width),
            "reference plane shape does not match frame"
        );
        anyhow::ensure!(
            (roi_mask.height, roi_mask.width) == (frame_lab.height, frame_lab.width),
            "ROI mask shape does not match frame"
        );

        let pixel_count = frame_lab.width * frame_lab.height;
        let reference = reference_plane
            .as_slice_memory_order()
            .ok_or_else(|| anyhow!("reference plane must be contiguous"))?;
        let reference_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fast-vrifa-reference-plane"),
                contents: bytemuck::cast_slice(reference),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let output =
            self.create_storage_buffer("fast-vrifa-delta-plane", byte_len_for_f32(pixel_count));
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fast-vrifa-delta-uniform"),
                contents: bytemuck::bytes_of(&DeltaUniform {
                    pixel_count: pixel_count as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    channel_weight,
                    _padf0: 0.0,
                    _padf1: 0.0,
                    _padf2: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fast-vrifa-delta-bind-group"),
            layout: &self.delta_layout,
            entries: &[
                storage_buffer_binding(0, &frame_lab.buffer),
                storage_buffer_binding(1, &reference_buffer),
                storage_buffer_binding(2, &roi_mask.buffer),
                storage_buffer_binding(3, &output),
                uniform_buffer_binding(4, &uniform),
            ],
        });
        self.dispatch_1d(
            "fast-vrifa-delta-dispatch",
            &self.delta_pipeline,
            &bind_group,
            pixel_count,
        );
        Ok(WgpuPlaneF32 {
            buffer: output,
            width: frame_lab.width,
            height: frame_lab.height,
        })
    }

    fn download_plane_f32(&self, plane: &Self::DevicePlaneF32) -> Result<Array2<f32>> {
        let pixel_count = plane.width * plane.height;
        let bytes = self.readback_bytes(&plane.buffer, byte_len_for_f32(pixel_count))?;
        let values = bytes_to_f32(&bytes)?;
        Array2::from_shape_vec((plane.height, plane.width), values)
            .context("reshaping downloaded delta plane")
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::BindGroupLayout,
    source: &'static str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

fn storage_buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_buffer_binding<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_buffer_binding<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn pack_bgr_pixels(frame_bgr: &Array3<u8>) -> Result<Vec<u32>> {
    let bytes = frame_bgr
        .as_slice_memory_order()
        .ok_or_else(|| anyhow!("BGR frame must be contiguous"))?;
    let packed = bytes
        .chunks_exact(3)
        .map(|pixel| pixel[0] as u32 | ((pixel[1] as u32) << 8) | ((pixel[2] as u32) << 16))
        .collect();
    Ok(packed)
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    anyhow::ensure!(
        bytes.len() % 4 == 0,
        "f32 readback had a non-multiple-of-4 byte length"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect())
}

fn bytes_to_u32(bytes: &[u8]) -> Result<Vec<u32>> {
    anyhow::ensure!(
        bytes.len() % 4 == 0,
        "u32 readback had a non-multiple-of-4 byte length"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect())
}

fn byte_len_for_f32(count: usize) -> u64 {
    (count * std::mem::size_of::<f32>()) as u64
}

fn byte_len_for_u32(count: usize) -> u64 {
    (count * std::mem::size_of::<u32>()) as u64
}

fn lab_lut_cache_path() -> PathBuf {
    std::env::temp_dir().join("fast_vrifa_bgr2lab_u8_lut_v1.bin")
}

fn load_or_build_lab_lut() -> Result<Vec<u32>> {
    let expected_bytes = (1usize << 24) * std::mem::size_of::<u32>();
    let cache_path = lab_lut_cache_path();
    if let Ok(bytes) = fs::read(&cache_path) {
        if bytes.len() == expected_bytes {
            return bytes_to_u32(&bytes);
        }
    }

    const CHUNK_SIZE: usize = 1 << 20;
    let mut lut = vec![0u32; 1 << 24];
    for start in (0..lut.len()).step_by(CHUNK_SIZE) {
        let len = (lut.len() - start).min(CHUNK_SIZE);
        let mut frame = Array3::<u8>::zeros((len, 1, 3));
        for offset in 0..len {
            let packed = (start + offset) as u32;
            frame[(offset, 0, 0)] = (packed & 0xff) as u8;
            frame[(offset, 0, 1)] = ((packed >> 8) & 0xff) as u8;
            frame[(offset, 0, 2)] = ((packed >> 16) & 0xff) as u8;
        }
        let converted = convert_frame_to_colorspace(&frame, ColorSpace::Cielab)
            .context("building the BGR->CIELAB lookup table")?;
        for offset in 0..len {
            lut[start + offset] = converted[(offset, 0, 0)] as u32
                | ((converted[(offset, 0, 1)] as u32) << 8)
                | ((converted[(offset, 0, 2)] as u32) << 16);
        }
    }

    let _ = fs::write(&cache_path, bytemuck::cast_slice(&lut));
    Ok(lut)
}

#[cfg(test)]
mod tests {
    use super::WgpuBackend;
    use fast_vrifa_core::{ImageBackend, RoiMargins};
    use ndarray::array;

    #[test]
    fn wgpu_backend_runs_stage_one_path() {
        let backend = WgpuBackend::new().unwrap();
        let frame = array![[[0u8, 0u8, 0u8], [255u8, 255u8, 255u8]]];
        let uploaded = backend.upload_frame_bgr(&frame).unwrap();
        let converted = backend.convert_bgr_to_lab(&uploaded).unwrap();
        let host = backend.download_frame_f32(&converted).unwrap();
        assert_eq!(host.dim(), (1, 2, 3));

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
            .compute_delta_darken_only(&converted, &reference, &mask, 1.0)
            .unwrap();
        assert_eq!(backend.download_mask_u8(&mask).unwrap().shape(), &[1, 2]);
        assert_eq!(backend.download_plane_f32(&delta).unwrap().shape(), &[1, 2]);
    }
}
