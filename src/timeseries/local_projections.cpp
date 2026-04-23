#include "hfm/timeseries/local_projections.hpp"
#include "hfm/covariance/covariance.hpp"
#include "hfm/linalg/solver.hpp"
#include <chrono>
#include <cmath>
#include <numbers>

namespace hfm {

Result<LPResult> local_projections(const Vector<f64>& y,
                                    const Vector<f64>& x,
                                    const Matrix<f64>& controls,
                                    const LPOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t T = y.size();
    std::size_t n_ctrl = controls.cols();
    std::size_t max_h = opts.max_horizon;
    std::size_t p = opts.n_lags;

    if (T != x.size() || (n_ctrl > 0 && controls.rows() != T)) {
        return Status::error(ErrorCode::DimensionMismatch, "lp: dimension mismatch");
    }
    if (T <= max_h + p + 2) {
        return Status::error(ErrorCode::InvalidArgument, "lp: not enough observations");
    }

    LPResult result;
    result.max_horizon = max_h;
    result.irf = Vector<f64>(max_h + 1, 0.0);
    result.irf_se = Vector<f64>(max_h + 1, 0.0);
    result.irf_lower = Vector<f64>(max_h + 1, 0.0);
    result.irf_upper = Vector<f64>(max_h + 1, 0.0);
    result.cumulative_irf = Vector<f64>(max_h + 1, 0.0);
    result.confidence_level = 0.95;
    f64 z = 1.96;

    for (std::size_t h = 0; h <= max_h; ++h) {
        std::size_t start_idx = p;
        std::size_t end_idx = T - h;
        if (end_idx <= start_idx) continue;
        std::size_t n_h = end_idx - start_idx;

        // Build design matrix: [const, x_t, controls_t, y_{t-1}...y_{t-p}, x_{t-1}...x_{t-p}]
        std::size_t k = (opts.add_intercept ? 1 : 0) + 1 + n_ctrl + 2 * p;
        Matrix<f64> X_h(n_h, k, 0.0);
        Vector<f64> y_h(n_h);

        for (std::size_t i = 0; i < n_h; ++i) {
            std::size_t t = start_idx + i;
            y_h[i] = y[t + h];

            std::size_t col = 0;
            if (opts.add_intercept) X_h(i, col++) = 1.0;
            X_h(i, col++) = x[t]; // impulse variable

            for (std::size_t j = 0; j < n_ctrl; ++j) {
                X_h(i, col++) = controls(t, j);
            }
            for (std::size_t lag = 1; lag <= p; ++lag) {
                X_h(i, col++) = y[t - lag];
            }
            for (std::size_t lag = 1; lag <= p; ++lag) {
                X_h(i, col++) = x[t - lag];
            }
        }

        auto beta_result = solve_least_squares(X_h, y_h);
        if (!beta_result) continue;
        auto& beta = beta_result.value();

        Vector<f64> fitted = matvec(X_h, beta);
        Vector<f64> resid = y_h - fitted;

        // IRF is the coefficient on x_t
        std::size_t irf_idx = opts.add_intercept ? 1 : 0;
        result.irf[h] = beta[irf_idx];

        // Covariance for SE
        Matrix<f64> XtX = matmul_AtB(X_h, X_h);
        auto XtX_inv = invert_spd(XtX);
        if (XtX_inv) {
            Matrix<f64> cov;
            switch (opts.covariance) {
                case CovarianceType::NeweyWest: {
                    i64 lag = (opts.hac_lag >= 0) ? opts.hac_lag : static_cast<i64>(h + 1);
                    cov = newey_west_covariance(X_h, resid, XtX_inv.value(), lag);
                    break;
                }
                case CovarianceType::White:
                    cov = white_covariance(X_h, resid, XtX_inv.value());
                    break;
                default:
                    cov = classical_covariance(X_h, resid, n_h, k);
                    break;
            }
            result.irf_se[h] = std::sqrt(std::max(0.0, cov(irf_idx, irf_idx)));
        }

        result.irf_lower[h] = result.irf[h] - z * result.irf_se[h];
        result.irf_upper[h] = result.irf[h] + z * result.irf_se[h];

        if (h == 0) {
            result.n_obs = n_h;
        }
    }

    // Cumulative IRF
    f64 cum = 0.0;
    for (std::size_t h = 0; h <= max_h; ++h) {
        cum += result.irf[h];
        result.cumulative_irf[h] = cum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

Result<LPResult> local_projections(const Vector<f64>& y, const LPOptions& opts) {
    Matrix<f64> no_controls(y.size(), 0);
    return local_projections(y, y, no_controls, opts);
}

} // namespace hfm
