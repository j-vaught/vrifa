struct Params {
  pixel_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> frame_lab: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_plane: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixel_count) {
    return;
  }

  output_plane[index] = f32(frame_lab[index] & 0xffu);
}
