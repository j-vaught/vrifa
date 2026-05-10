# Divergences

## MP4 postprocess path

`fast-vrifa` stages video output as raw frame streams under `<output_dir>/.streams/` during the timed pipeline. MP4 files are only produced when `--ffmpeg-postprocess` is enabled, in which case `ffmpeg` encodes `mask.mp4`, `overlay.mp4`, and `heatmap.mp4` after the per-frame run completes.

PNG mask, overlay, and heatmap outputs are unchanged and remain the authoritative parity artifacts. This divergence affects only the optional MP4 generation path and exists to keep video encoding off the timed CUDA core path.

## COCO bbox-only export

When `--coco-bbox-only` is enabled, `fast-vrifa` writes `instances_default.json` without emitting the per-frame source PNGs under `formatCOCO/images/default/`.

The JSON still preserves the image filenames it would have used and adds `images_omitted: true` plus `source_video: ...` at the top level so downstream tooling can recover the source frame from the original video. This divergence is opt-in and only affects COCO consumers that expected inline frame PNGs.

## WGPU colorspace lookup

The `wgpu` milestone uses an exact BGR->CIELAB lookup table generated from the locked CPU conversion to keep stage-1 parity exact.

## CUDA pre-delta blur

When the CUDA darken-only fast path runs with `--pre-delta-blur-kernel > 1`, `fast-vrifa` blurs the extracted `L*` plane on device before both the peak update and the delta stage.

The CPU reference blurs the full converted frame and then consumes the blurred `L*` channel. For darken-only CIELAB runs these are mathematically equivalent, but the CUDA path still records this as a GPU-stage divergence because the blur is implemented by the shared device Gaussian kernel rather than OpenCV.

## CUDA stabilization warp

When `--camera-stable` is enabled on the CUDA darken-only fast path, `fast-vrifa` keeps motion estimation and ECC fitting on the host, then applies the fitted warp to the device-side `L*` or peak plane before the delta stage consumes it.

The CPU reference uses OpenCV `warpAffine` on host arrays. The CUDA path uses a device bicubic plane warp with replicated borders, so stage dumps can differ slightly within the existing GPU tolerance even though final masks are expected to match the CPU reference.
