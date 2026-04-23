#include <metal_stdlib>
using namespace metal;

// Compute X'X and X'y for many small regressions in parallel
// Each thread handles one regression (same X, different y)
kernel void batched_xtx_xty(
    device const float* X [[buffer(0)]],       // n x k design matrix (shared)
    device const float* Y [[buffer(1)]],       // n x batch_size (each column is a y)
    device float* XtX_out [[buffer(2)]],       // k x k (shared, computed once by thread 0)
    device float* XtY_out [[buffer(3)]],       // k x batch_size
    device const uint& n [[buffer(4)]],
    device const uint& k [[buffer(5)]],
    device const uint& batch_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch_size) return;

    // Compute X'y for this regression
    for (uint j = 0; j < k; ++j) {
        float dot = 0.0f;
        for (uint i = 0; i < n; ++i) {
            dot += X[i * k + j] * Y[i * batch_size + gid];
        }
        XtY_out[j * batch_size + gid] = dot;
    }

    // Thread 0 computes shared X'X
    if (gid == 0) {
        for (uint j = 0; j < k; ++j) {
            for (uint l = j; l < k; ++l) {
                float dot = 0.0f;
                for (uint i = 0; i < n; ++i) {
                    dot += X[i * k + j] * X[i * k + l];
                }
                XtX_out[j * k + l] = dot;
                XtX_out[l * k + j] = dot; // symmetric
            }
        }
    }
}

// Compute residual sum of squares for many regressions
kernel void batched_residuals(
    device const float* Y [[buffer(0)]],       // n x batch_size
    device const float* X [[buffer(1)]],       // n x k
    device const float* betas [[buffer(2)]],   // k x batch_size
    device float* rss [[buffer(3)]],           // batch_size
    device const uint& n [[buffer(4)]],
    device const uint& k [[buffer(5)]],
    device const uint& batch_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch_size) return;

    float ss = 0.0f;
    for (uint i = 0; i < n; ++i) {
        float fitted = 0.0f;
        for (uint j = 0; j < k; ++j) {
            fitted += X[i * k + j] * betas[j * batch_size + gid];
        }
        float resid = Y[i * batch_size + gid] - fitted;
        ss += resid * resid;
    }
    rss[gid] = ss;
}
