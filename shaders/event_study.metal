#include <metal_stdlib>
using namespace metal;

kernel void extract_event_windows(device const float* returns [[buffer(0)]],
                                   device const uint* event_indices [[buffer(1)]],
                                   device float* windows_out [[buffer(2)]],
                                   device const uint& n_returns [[buffer(3)]],
                                   device const uint& n_events [[buffer(4)]],
                                   device const int& left_window [[buffer(5)]],
                                   device const int& right_window [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= n_events) return;

    uint event_idx = event_indices[gid];
    int window_size = right_window - left_window + 1;
    uint out_offset = gid * uint(window_size);

    for (int w = left_window; w <= right_window; ++w) {
        int src_idx = int(event_idx) + w;
        int out_idx = w - left_window;
        if (src_idx >= 0 && uint(src_idx) < n_returns) {
            windows_out[out_offset + uint(out_idx)] = returns[uint(src_idx)];
        } else {
            windows_out[out_offset + uint(out_idx)] = 0.0f; // NaN marker could be used
        }
    }
}

kernel void cumulative_returns(device const float* windows [[buffer(0)]],
                                device float* cum_returns [[buffer(1)]],
                                device const uint& n_events [[buffer(2)]],
                                device const uint& window_size [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= n_events) return;

    uint offset = gid * window_size;
    float cumsum = 0.0f;
    for (uint i = 0; i < window_size; ++i) {
        cumsum += windows[offset + i];
        cum_returns[offset + i] = cumsum;
    }
}
