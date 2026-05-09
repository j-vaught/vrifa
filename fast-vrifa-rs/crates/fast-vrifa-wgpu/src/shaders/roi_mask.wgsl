struct Params {
  width: u32,
  height: u32,
  top: u32,
  bottom: u32,
  left: u32,
  right: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> output_mask: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let pixel_count = params.width * params.height;
  if (index >= pixel_count) {
    return;
  }

  let y = index / params.width;
  let x = index % params.width;
  let inside = y >= params.top && y < params.bottom && x >= params.left && x < params.right;
  output_mask[index] = select(0u, 1u, inside);
}
