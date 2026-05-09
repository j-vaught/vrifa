struct Params {
  pixel_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  channel_weight: f32,
  _padf0: f32,
  _padf1: f32,
  _padf2: f32,
}

@group(0) @binding(0) var<storage, read> frame_lab: array<u32>;
@group(0) @binding(1) var<storage, read> reference_plane: array<f32>;
@group(0) @binding(2) var<storage, read> roi_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_delta: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixel_count) {
    return;
  }

  let current_l = f32(frame_lab[index] & 0xffu);
  let raw = (reference_plane[index] - current_l) * params.channel_weight;
  let clipped = max(raw, 0.0);
  output_delta[index] = clipped * f32(roi_mask[index]);
}
