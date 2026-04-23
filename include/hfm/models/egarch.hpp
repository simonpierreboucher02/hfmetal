#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

struct EGARCHOptions {
    f64 omega_init = -0.1;
    f64 alpha_init = 0.10;
    f64 gamma_init = -0.05;
    f64 beta_init = 0.98;
    std::size_t max_iter = 500;
    f64 tol = 1e-8;
    bool compute_std_errors = true;
    Backend backend = Backend::Auto;

    EGARCHOptions& set_max_iter(std::size_t m) { max_iter = m; return *this; }
    EGARCHOptions& set_tol(f64 t) { tol = t; return *this; }
};

struct EGARCHResult {
    f64 omega = 0.0;
    f64 alpha = 0.0;
    f64 gamma = 0.0;
    f64 beta = 0.0;
    Vector<f64> conditional_var;
    Vector<f64> std_residuals;
    f64 log_likelihood = 0.0;
    f64 aic = 0.0;
    f64 bic = 0.0;
    Vector<f64> std_errors;
    Vector<f64> t_stats;
    std::size_t n_obs = 0;
    std::size_t n_iter = 0;
    bool converged = false;
    f64 elapsed_ms = 0.0;
};

// EGARCH(1,1): log(σ²_t) = ω + α|z_{t-1}| + γz_{t-1} + β·log(σ²_{t-1})
// Captures leverage effect via γ (negative γ → bad news increases vol more)
Result<EGARCHResult> egarch(const Vector<f64>& returns,
                             const EGARCHOptions& opts = EGARCHOptions{});

f64 egarch_loglikelihood(const Vector<f64>& returns, f64 mu,
                          f64 omega, f64 alpha, f64 gamma, f64 beta);

} // namespace hfm
