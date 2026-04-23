#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

struct GJRGARCHOptions {
    f64 omega_init = 0.00001;
    f64 alpha_init = 0.05;
    f64 gamma_init = 0.04;
    f64 beta_init = 0.88;
    std::size_t max_iter = 500;
    f64 tol = 1e-8;
    bool compute_std_errors = true;
    Backend backend = Backend::Auto;

    GJRGARCHOptions& set_max_iter(std::size_t m) { max_iter = m; return *this; }
    GJRGARCHOptions& set_tol(f64 t) { tol = t; return *this; }
};

struct GJRGARCHResult {
    f64 omega = 0.0;
    f64 alpha = 0.0;
    f64 gamma = 0.0;
    f64 beta = 0.0;
    f64 persistence = 0.0;         // alpha + gamma/2 + beta
    f64 unconditional_var = 0.0;
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

// GJR-GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1}
// γ > 0 → negative shocks increase volatility more (leverage effect)
Result<GJRGARCHResult> gjr_garch(const Vector<f64>& returns,
                                   const GJRGARCHOptions& opts = GJRGARCHOptions{});

f64 gjr_garch_loglikelihood(const Vector<f64>& returns, f64 mu,
                              f64 omega, f64 alpha, f64 gamma, f64 beta);

} // namespace hfm
