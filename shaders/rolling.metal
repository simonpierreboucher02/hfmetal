#include <metal_stdlib>
using namespace metal;

kernel void rolling_sum(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        device const uint& count [[buffer(2)]],
                        device const uint& window [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid + window > count) return;
    float s = 0.0f;
    for (uint i = 0; i < window; ++i) {
        s += input[gid + i];
    }
    output[gid] = s;
}

kernel void rolling_mean(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         device const uint& count [[buffer(2)]],
                         device const uint& window [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid + window > count) return;
    float s = 0.0f;
    for (uint i = 0; i < window; ++i) {
        s += input[gid + i];
    }
    output[gid] = s / float(window);
}

kernel void rolling_variance(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             device const uint& count [[buffer(2)]],
                             device const uint& window [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid + window > count) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint i = 0; i < window; ++i) {
        float v = input[gid + i];
        sum += v;
        sum_sq += v * v;
    }
    float mean = sum / float(window);
    output[gid] = sum_sq / float(window) - mean * mean;
}
