#include <metal_stdlib>
using namespace metal;

kernel void log_returns(device const float* prices [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        device const uint& count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count - 1) return;
    out[gid] = log(prices[gid + 1] / prices[gid]);
}

kernel void simple_returns(device const float* prices [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           device const uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count - 1) return;
    out[gid] = (prices[gid + 1] - prices[gid]) / prices[gid];
}

kernel void log_returns_double(device const float* prices [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                device const uint& count [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    // Metal doesn't natively support f64 on all devices.
    // Use float precision with explicit conversion for now.
    if (gid >= count - 1) return;
    float p_curr = prices[gid];
    float p_next = prices[gid + 1];
    out[gid] = log(p_next / p_curr);
}
