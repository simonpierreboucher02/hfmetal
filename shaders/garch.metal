#include <metal_stdlib>
using namespace metal;

// GARCH(1,1) log-likelihood for a batch of parameter sets
// Each thread evaluates one parameter combination against the same return series
kernel void garch_loglikelihood_batch(
    device const float* returns [[buffer(0)]],   // n returns
    device const float4* params [[buffer(1)]],   // batch_size x (mu, omega, alpha, beta)
    device float* log_likelihoods [[buffer(2)]],  // batch_size outputs
    device const uint& n [[buffer(3)]],
    device const uint& batch_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch_size) return;

    float mu = params[gid].x;
    float omega = params[gid].y;
    float alpha = params[gid].z;
    float beta = params[gid].w;

    if (omega <= 0.0f || alpha < 0.0f || beta < 0.0f || alpha + beta >= 1.0f) {
        log_likelihoods[gid] = -1e30f;
        return;
    }

    float h = omega / (1.0f - alpha - beta);
    float ll = 0.0f;
    const float log2pi = 1.8378770664093453f;

    for (uint t = 0; t < n; ++t) {
        if (h < 1e-20f) h = 1e-20f;
        float e = returns[t] - mu;
        ll += -0.5f * (log2pi + log(h) + e * e / h);
        h = omega + alpha * e * e + beta * h;
    }

    log_likelihoods[gid] = ll;
}

// Compute conditional variance series for fitted GARCH
kernel void garch_conditional_variance(
    device const float* returns [[buffer(0)]],
    device float* cond_var [[buffer(1)]],
    device float* std_resid [[buffer(2)]],
    device const float& mu [[buffer(3)]],
    device const float& omega [[buffer(4)]],
    device const float& alpha [[buffer(5)]],
    device const float& beta [[buffer(6)]],
    device const uint& n [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    // This kernel runs single-threaded (sequential dependency)
    if (gid != 0) return;

    float h = omega / (1.0f - alpha - beta);
    for (uint t = 0; t < n; ++t) {
        if (h < 1e-20f) h = 1e-20f;
        cond_var[t] = h;
        float e = returns[t] - mu;
        std_resid[t] = e / sqrt(h);
        h = omega + alpha * e * e + beta * h;
    }
}
