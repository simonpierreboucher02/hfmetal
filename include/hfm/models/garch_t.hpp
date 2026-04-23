#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

struct GARCHTOptions {
    f64 omega_init = 0.00001;
    f64 alpha_init = 0.05;
    f64 beta_init = 0.90;
    f64 nu_init = 8.0;
    std::size_t max_iter = 500;
    f64 tol = 1e-8;
    bool compute_std_errors = true;
    Backend backend = Backend::Auto;

    GARCHTOptions& set_max_iter(std::size_t m) { max_iter = m; return *this; }
    GARCHTOptions& set_tol(f64 t) { tol = t; return *this; }
    GARCHTOptions& set_nu_init(f64 v) { nu_init = v; return *this; }
};

struct GARCHTResult {
    f64 omega = 0.0;
    f64 alpha = 0.0;
    f64 beta = 0.0;
    f64 nu = 0.0;
    f64 persistence = 0.0;
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

// GARCH(1,1)-t: same variance dynamics as GARCH(1,1), but with Student-t innovations
// σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}, ε_t ~ t_ν(0, σ²_t)
Result<GARCHTResult> garch_t(const Vector<f64>& returns,
                               const GARCHTOptions& opts = GARCHTOptions{});

} // namespace hfm
