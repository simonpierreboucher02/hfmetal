#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

struct GARCHOptions {
    std::size_t p = 1;
    std::size_t q = 1;
    f64 omega_init = 0.00001;
    f64 alpha_init = 0.05;
    f64 beta_init = 0.90;
    std::size_t max_iter = 500;
    f64 tol = 1e-8;
    bool compute_std_errors = true;
    Backend backend = Backend::Auto;

    GARCHOptions& set_max_iter(std::size_t m) { max_iter = m; return *this; }
    GARCHOptions& set_tol(f64 t) { tol = t; return *this; }
};

struct GARCHResult {
    f64 omega = 0.0;
    f64 alpha = 0.0;
    f64 beta = 0.0;
    f64 persistence = 0.0;         // alpha + beta
    f64 unconditional_var = 0.0;   // omega / (1 - alpha - beta)
    Vector<f64> conditional_var;   // sigma^2_t series
    Vector<f64> std_residuals;     // e_t / sigma_t
    f64 log_likelihood = 0.0;
    f64 aic = 0.0;
    f64 bic = 0.0;
    Vector<f64> std_errors;        // [omega_se, alpha_se, beta_se]
    Vector<f64> t_stats;
    std::size_t n_obs = 0;
    std::size_t n_iter = 0;
    bool converged = false;
    f64 elapsed_ms = 0.0;
};

// GARCH(1,1): sigma^2_t = omega + alpha * e^2_{t-1} + beta * sigma^2_{t-1}
// Input: return series (demeaned or raw — mean is subtracted internally)
Result<GARCHResult> garch(const Vector<f64>& returns,
                           const GARCHOptions& opts = GARCHOptions{});

// Compute GARCH(1,1) log-likelihood for given parameters
f64 garch_loglikelihood(const Vector<f64>& returns, f64 mu,
                         f64 omega, f64 alpha, f64 beta);

} // namespace hfm
