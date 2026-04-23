#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/timeseries/var.hpp"

namespace hfm {

struct IRFResult {
    std::vector<Matrix<f64>> irf;    // irf[h] is k×k matrix at horizon h
    std::size_t n_horizons = 0;
    std::size_t n_vars = 0;
    f64 elapsed_ms = 0.0;
};

// Orthogonalized IRF using Cholesky decomposition of Σ_u
Result<IRFResult> var_irf(const VARResult& var_result,
                           std::size_t n_horizons = 20);

struct FEVDResult {
    std::vector<Matrix<f64>> fevd;   // fevd[h] is k×k: row i, col j = share of
                                      // var i's h-step forecast error from shock j
    std::size_t n_horizons = 0;
    std::size_t n_vars = 0;
    f64 elapsed_ms = 0.0;
};

Result<FEVDResult> var_fevd(const VARResult& var_result,
                             std::size_t n_horizons = 20);

// ========== Forecast evaluation ==========

struct ForecastEvalResult {
    f64 mae = 0.0;
    f64 rmse = 0.0;
    f64 mape = 0.0;
    f64 mse = 0.0;
    f64 theil_u = 0.0;
    f64 r_squared = 0.0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

Result<ForecastEvalResult> forecast_eval(const Vector<f64>& actual,
                                          const Vector<f64>& forecast);

} // namespace hfm
