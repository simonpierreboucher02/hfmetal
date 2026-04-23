#include <metal_stdlib>
using namespace metal;

kernel void realized_variance_kernel(device const float* returns [[buffer(0)]],
                                      device float* partial_sums [[buffer(1)]],
                                      device const uint& count [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint tg_size [[threads_per_threadgroup]],
                                      uint tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float shared[1024];

    float val = (gid < count) ? returns[gid] : 0.0f;
    shared[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[tg_id] = shared[0];
    }
}

kernel void bipower_variation_kernel(device const float* returns [[buffer(0)]],
                                      device float* partial_sums [[buffer(1)]],
                                      device const uint& count [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint tg_size [[threads_per_threadgroup]],
                                      uint tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float shared[1024];

    float val = 0.0f;
    if (gid > 0 && gid < count) {
        val = abs(returns[gid]) * abs(returns[gid - 1]);
    }
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[tg_id] = shared[0];
    }
}
