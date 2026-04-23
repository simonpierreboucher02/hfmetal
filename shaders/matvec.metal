#include <metal_stdlib>
using namespace metal;

// Batched matrix-vector multiply: y = A * x for many (A, x) pairs
// Each thread handles one row of one matrix-vector product
kernel void batched_matvec(
    device const float* matrices [[buffer(0)]],  // batch_size * rows * cols
    device const float* vectors [[buffer(1)]],   // batch_size * cols
    device float* results [[buffer(2)]],         // batch_size * rows
    device const uint& rows [[buffer(3)]],
    device const uint& cols [[buffer(4)]],
    device const uint& batch_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batch_size * rows;
    if (gid >= total) return;

    uint b = gid / rows;
    uint r = gid % rows;

    float dot = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        dot += matrices[b * rows * cols + r * cols + c] * vectors[b * cols + c];
    }
    results[b * rows + r] = dot;
}

// Outer product accumulation: C += alpha * x * y' for many pairs
kernel void batched_outer_product(
    device const float* x_vecs [[buffer(0)]],   // batch_size * m
    device const float* y_vecs [[buffer(1)]],   // batch_size * n
    device atomic_float* result [[buffer(2)]],   // m * n (accumulated)
    device const uint& m [[buffer(3)]],
    device const uint& n [[buffer(4)]],
    device const uint& batch_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint total_elements = m * n;
    if (gid >= total_elements) return;

    uint i = gid / n;
    uint j = gid % n;

    float sum = 0.0f;
    for (uint b = 0; b < batch_size; ++b) {
        sum += x_vecs[b * m + i] * y_vecs[b * n + j];
    }
    atomic_fetch_add_explicit(&result[gid], sum, memory_order_relaxed);
}
