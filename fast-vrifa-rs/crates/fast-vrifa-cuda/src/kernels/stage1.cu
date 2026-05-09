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

struct NppiPointCuda {
    int x;
    int y;
};

struct NppiRectCuda {
    int x;
    int y;
    int width;
    int height;
};

struct NppiCompressedMarkerLabelsInfoCuda {
    unsigned int nMarkerLabelPixelCount;
    unsigned int nContourPixelCount;
    unsigned int nContourPixelsFound;
    NppiPointCuda oContourFirstPixelLocation;
    NppiRectCuda oMarkerLabelBoundingBox;
};

extern "C" __global__ void filter_labeled_components_u8(
    const unsigned char* source_mask,
    const unsigned int* labels,
    const NppiCompressedMarkerLabelsInfoCuda* info_list,
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
    const NppiCompressedMarkerLabelsInfoCuda info = info_list[label - 1u];
    output[index] = info.nMarkerLabelPixelCount >= min_area ? 255u : 0u;
}
