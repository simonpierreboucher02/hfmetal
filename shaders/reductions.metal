#include <metal_stdlib>
using namespace metal;

kernel void sum_reduce(device const float* input [[buffer(0)]],
                       device float* partial_sums [[buffer(1)]],
                       device const uint& count [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]],
                       uint tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float shared[1024];

    float val = (gid < count) ? input[gid] : 0.0f;
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

kernel void sum_of_squares(device const float* input [[buffer(0)]],
                           device float* partial_sums [[buffer(1)]],
                           device const uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint tg_size [[threads_per_threadgroup]],
                           uint tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float shared[1024];

    float val = (gid < count) ? input[gid] : 0.0f;
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
