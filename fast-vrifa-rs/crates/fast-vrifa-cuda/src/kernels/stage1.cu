__device__ int reflect101(int value, int limit) {
    if (limit <= 1) {
        return 0;
    }
    while (value < 0 || value >= limit) {
        if (value < 0) {
            value = -value;
        } else {
            value = (2 * limit - 2) - value;
        }
    }
    return value;
}

__device__ int clamp_replicate(int value, int limit) {
    if (limit <= 1) {
        return 0;
    }
    if (value < 0) {
        return 0;
    }
    if (value >= limit) {
        return limit - 1;
    }
    return value;
}

__device__ float cubic_weight(float value) {
    const float a = -0.75f;
    value = fabsf(value);
    if (value <= 1.0f) {
        return ((a + 2.0f) * value - (a + 3.0f)) * value * value + 1.0f;
    }
    if (value < 2.0f) {
        return ((a * value - 5.0f * a) * value + 8.0f * a) * value - 4.0f * a;
    }
    return 0.0f;
}

__device__ float sample_bicubic_replicate(
    const float* input,
    int width,
    int height,
    float x,
    float y
) {
    int base_x = static_cast<int>(floorf(x));
    int base_y = static_cast<int>(floorf(y));
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int dy = -1; dy <= 2; ++dy) {
        int yy = clamp_replicate(base_y + dy, height);
        float wy = cubic_weight(y - static_cast<float>(base_y + dy));
        for (int dx = -1; dx <= 2; ++dx) {
            int xx = clamp_replicate(base_x + dx, width);
            float wx = cubic_weight(x - static_cast<float>(base_x + dx));
            float weight = wx * wy;
            sum += input[yy * width + xx] * weight;
            weight_sum += weight;
        }
    }
    return weight_sum != 0.0f ? (sum / weight_sum) : 0.0f;
}

extern "C" __global__ void bgr_to_lab_lut(
    const unsigned char* input,
    const unsigned int* lut,
    unsigned int* output,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    int base = index * 3;
    unsigned int packed =
        static_cast<unsigned int>(input[base]) |
        (static_cast<unsigned int>(input[base + 1]) << 8) |
        (static_cast<unsigned int>(input[base + 2]) << 16);
    output[index] = lut[packed];
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

extern "C" __global__ void update_peak_brightness_plane(
    const float* frame_l,
    const float* previous_peak,
    float* output,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    float current_l = frame_l[index];
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

extern "C" __global__ void compute_delta_darken_only_plane(
    const float* frame_l,
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
    float raw = (reference_plane[index] - frame_l[index]) * channel_weight;
    if (raw < 0.0f) {
        raw = 0.0f;
    }
    output[index] = roi_mask[index] ? raw : 0.0f;
}

extern "C" __global__ void gaussian_blur_f32(
    const float* input,
    const float* weights,
    float* output,
    int width,
    int height,
    int kernel_size,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    int radius = kernel_size / 2;
    int y = index / width;
    int x = index - (y * width);
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int yy = reflect101(y + dy, height);
        float wy = weights[dy + radius];
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = reflect101(x + dx, width);
            float wx = weights[dx + radius];
            sum += input[yy * width + xx] * wy * wx;
        }
    }
    output[index] = sum;
}

extern "C" __global__ void warp_affine_f32(
    const float* input,
    float* output,
    int width,
    int height,
    float m00,
    float m01,
    float m02,
    float m10,
    float m11,
    float m12,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    int y = index / width;
    int x = index - (y * width);
    float src_x = m00 * static_cast<float>(x) + m01 * static_cast<float>(y) + m02;
    float src_y = m10 * static_cast<float>(x) + m11 * static_cast<float>(y) + m12;
    output[index] = sample_bicubic_replicate(input, width, height, src_x, src_y);
}

extern "C" __global__ void reduce_minmax_nonnegative(
    const float* input,
    unsigned int* min_bits,
    unsigned int* max_bits,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    unsigned int bits = __float_as_uint(input[index]);
    atomicMin(min_bits, bits);
    atomicMax(max_bits, bits);
}

extern "C" __global__ void normalize_minmax_u8(
    const float* input,
    unsigned char* output,
    float min_value,
    float max_value,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    float value = input[index];
    float normalized = 0.0f;
    if (max_value > min_value) {
        normalized = (value - min_value) * (255.0f / (max_value - min_value));
    }
    if (normalized < 0.0f) {
        normalized = 0.0f;
    } else if (normalized > 255.0f) {
        normalized = 255.0f;
    }
    output[index] = static_cast<unsigned char>(normalized);
}

extern "C" __global__ void threshold_binary_u8(
    const unsigned char* input,
    unsigned char* output,
    float threshold_value,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    output[index] = static_cast<float>(input[index]) > threshold_value ? 255u : 0u;
}

extern "C" __global__ void threshold_binary_u8_device(
    const unsigned char* input,
    unsigned char* output,
    const float* threshold_value,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    float threshold = threshold_value[0];
    output[index] = static_cast<float>(input[index]) > threshold ? 255u : 0u;
}

extern "C" __global__ void histogram_u8(
    const unsigned char* input,
    unsigned int* histogram,
    int pixel_count
) {
    __shared__ unsigned int local_histogram[256];
    int tid = threadIdx.x;
    if (tid < 256) {
        local_histogram[tid] = 0u;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    while (index < pixel_count) {
        atomicAdd(&local_histogram[input[index]], 1u);
        index += stride;
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&histogram[tid], local_histogram[tid]);
    }
}

extern "C" __global__ void otsu_threshold_from_histogram(
    const unsigned int* histogram,
    float* threshold_out,
    float threshold_offset
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    unsigned int total = 0u;
    double sum = 0.0;
    for (int i = 0; i < 256; ++i) {
        unsigned int count = histogram[i];
        total += count;
        sum += static_cast<double>(i) * static_cast<double>(count);
    }

    double sum_background = 0.0;
    unsigned int weight_background = 0u;
    double best_variance = -1.0;
    int best_threshold = 0;

    for (int i = 0; i < 256; ++i) {
        unsigned int count = histogram[i];
        weight_background += count;
        if (weight_background == 0u) {
            continue;
        }
        unsigned int weight_foreground = total - weight_background;
        if (weight_foreground == 0u) {
            break;
        }
        sum_background += static_cast<double>(i) * static_cast<double>(count);
        double mean_background = sum_background / static_cast<double>(weight_background);
        double mean_foreground =
            (sum - sum_background) / static_cast<double>(weight_foreground);
        double diff = mean_background - mean_foreground;
        double between =
            static_cast<double>(weight_background) *
            static_cast<double>(weight_foreground) * diff * diff;
        if (between > best_variance) {
            best_variance = between;
            best_threshold = i;
        }
    }

    float threshold = static_cast<float>(best_threshold) + threshold_offset;
    if (threshold < 0.0f) {
        threshold = 0.0f;
    } else if (threshold > 255.0f) {
        threshold = 255.0f;
    }
    threshold_out[0] = threshold;
}

extern "C" __global__ void dilate_binary_u8(
    const unsigned char* input,
    const unsigned char* kernel_mask,
    unsigned char* output,
    int width,
    int height,
    int kernel_size,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    int radius = kernel_size / 2;
    int y = index / width;
    int x = index - (y * width);
    unsigned char result = 0u;
    for (int ky = 0; ky < kernel_size && result == 0u; ++ky) {
        int yy = y + ky - radius;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int kx = 0; kx < kernel_size; ++kx) {
            if (!kernel_mask[ky * kernel_size + kx]) {
                continue;
            }
            int xx = x + kx - radius;
            if (xx < 0 || xx >= width) {
                continue;
            }
            if (input[yy * width + xx] > 0u) {
                result = 255u;
                break;
            }
        }
    }
    output[index] = result;
}

extern "C" __global__ void erode_binary_u8(
    const unsigned char* input,
    const unsigned char* kernel_mask,
    unsigned char* output,
    int width,
    int height,
    int kernel_size,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    int radius = kernel_size / 2;
    int y = index / width;
    int x = index - (y * width);
    unsigned char result = 255u;
    for (int ky = 0; ky < kernel_size && result == 255u; ++ky) {
        int yy = y + ky - radius;
        for (int kx = 0; kx < kernel_size; ++kx) {
            if (!kernel_mask[ky * kernel_size + kx]) {
                continue;
            }
            int xx = x + kx - radius;
            if (yy < 0 || yy >= height || xx < 0 || xx >= width || input[yy * width + xx] == 0u) {
                result = 0u;
                break;
            }
        }
    }
    output[index] = result;
}

extern "C" __global__ void count_labeled_components_u32(
    const unsigned int* labels,
    unsigned int* counts,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    unsigned int label = labels[index];
    if (label == 0u) {
        return;
    }
    atomicAdd(&counts[label], 1u);
}

extern "C" __global__ void filter_labeled_components_u8(
    const unsigned char* source_mask,
    const unsigned int* labels,
    const unsigned int* counts,
    unsigned char* output,
    unsigned int min_area,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    if (source_mask[index] == 0u) {
        output[index] = 0u;
        return;
    }
    unsigned int label = labels[index];
    if (label == 0u) {
        output[index] = 0u;
        return;
    }
    output[index] = counts[label] >= min_area ? 255u : 0u;
}

extern "C" __global__ void apply_locking_u8(
    const unsigned char* input_mask,
    unsigned short* counter,
    unsigned char* locked,
    unsigned char* output_mask,
    unsigned short lock_frames,
    int pixel_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }

    unsigned short current_counter = input_mask[index] > 0u
        ? static_cast<unsigned short>(counter[index] == 65535u ? 65535u : counter[index] + 1u)
        : 0u;
    counter[index] = current_counter;
    if (current_counter >= lock_frames) {
        locked[index] = 255u;
    }
    unsigned char input_value = input_mask[index];
    unsigned char locked_value = locked[index];
    output_mask[index] = locked_value > input_value ? locked_value : input_value;
}
