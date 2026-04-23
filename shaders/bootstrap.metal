#include <metal_stdlib>
using namespace metal;

// Simple LCG PRNG for bootstrap index generation
inline uint lcg_next(uint state) {
    return state * 1664525u + 1013904223u;
}

kernel void block_bootstrap_indices(device uint* indices [[buffer(0)]],
                                     device const uint& n [[buffer(1)]],
                                     device const uint& block_size [[buffer(2)]],
                                     device const uint& seed_base [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    // Each thread generates one bootstrap sample's indices
    uint state = seed_base + gid * 7919u;
    uint n_blocks = (n + block_size - 1) / block_size;

    for (uint b = 0; b < n_blocks; ++b) {
        state = lcg_next(state);
        uint block_start = state % n;
        for (uint i = 0; i < block_size; ++i) {
            uint out_idx = b * block_size + i;
            if (out_idx >= n) break;
            indices[gid * n + out_idx] = (block_start + i) % n;
        }
    }
}

kernel void gather_by_index(device const float* source [[buffer(0)]],
                             device const uint* indices [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             device const uint& count [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    output[gid] = source[indices[gid]];
}
