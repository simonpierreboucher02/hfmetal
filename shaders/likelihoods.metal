#include <metal_stdlib>
using namespace metal;

kernel void normal_log_likelihood(device const float* y [[buffer(0)]],
                                   device const float* mu [[buffer(1)]],
                                   device const float& sigma [[buffer(2)]],
                                   device float* ll_out [[buffer(3)]],
                                   device const uint& count [[buffer(4)]],
                                   uint gid [[thread_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]],
                                   uint tg_id [[threadgroup_position_in_grid]]) {
    threadgroup float shared[1024];

    float val = 0.0f;
    if (gid < count) {
        float diff = y[gid] - mu[gid];
        float s2 = sigma * sigma;
        val = -0.5f * (log(2.0f * M_PI_F * s2) + diff * diff / s2);
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
        ll_out[tg_id] = shared[0];
    }
}
