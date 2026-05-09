extern "C" __global__ void bgr_to_lab_lut(
    const unsigned int* input,
    const unsigned int* lut,
    unsigned int* output,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    output[index] = lut[input[index] & 0x00ffffffu];
}

extern "C" __global__ void build_roi_mask(
    unsigned char* output,
    int width,
    int height,
    int top,
    int bottom,
    int left,
    int right
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_count = width * height;
    if (index >= pixel_count) {
        return;
    }
    int y = index / width;
    int x = index - (y * width);
    output[index] =
        (y >= top && y < bottom && x >= left && x < right) ? 1u : 0u;
}

extern "C" __global__ void extract_l_plane(
    const unsigned int* frame_lab,
    float* output,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    output[index] = static_cast<float>(frame_lab[index] & 0xffu);
}

extern "C" __global__ void update_peak_brightness(
    const unsigned int* frame_lab,
    const float* previous_peak,
    float* output,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    float current_l = static_cast<float>(frame_lab[index] & 0xffu);
    float previous = previous_peak[index];
    output[index] = current_l > previous ? current_l : previous;
}

extern "C" __global__ void compute_delta_darken_only(
    const unsigned int* frame_lab,
    const float* reference_plane,
    const unsigned char* roi_mask,
    float* output,
    float channel_weight,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    float l_star = static_cast<float>(frame_lab[index] & 0xffu);
    float raw = (reference_plane[index] - l_star) * channel_weight;
    if (raw < 0.0f) {
        raw = 0.0f;
    }
    output[index] = roi_mask[index] ? raw : 0.0f;
}
