#include "hfm/estimators/rolling_ols.hpp"
#include "hfm/linalg/solver.hpp"
#include "hfm/covariance/covariance.hpp"
#include <chrono>
#include <cmath>

namespace hfm {

Vector<f64> RollingOLSResult::beta_at(std::size_t idx) const {
    Vector<f64> b(n_regressors());
    for (std::size_t j = 0; j < n_regressors(); ++j) {
        b[j] = betas_(idx, j);
    }
    return b;
}

Vector<f64> RollingOLSResult::se_at(std::size_t idx) const {
    Vector<f64> s(n_regressors());
    for (std::size_t j = 0; j < n_regressors(); ++j) {
        s[j] = se_(idx, j);
    }
    return s;
}

namespace {

struct WindowOLS {
    Vector<f64> beta;
    Vector<f64> se;
    Vector<f64> t;
    f64 r2 = 0.0;
    f64 sigma = 0.0;
    bool valid = false;
};

WindowOLS fit_window(const Vector<f64>& y, const Matrix<f64>& X,
                      std::size_t start, std::size_t end,
                      CovarianceType cov_type, i64 hac_lag) {
    WindowOLS result;
    std::size_t n = end - start;
    std::size_t k = X.cols();

    Matrix<f64> Xw(n, k);
    Vector<f64> yw(n);
    for (std::size_t i = 0; i < n; ++i) {
        yw[i] = y[start + i];
        for (std::size_t j = 0; j < k; ++j) {
            Xw(i, j) = X(start + i, j);
        }
    }

    auto beta_res = solve_least_squares(Xw, yw);
    if (!beta_res) return result;

    result.beta = std::move(beta_res).value();
    result.valid = true;

    // Residuals
    auto fitted = matvec(Xw, result.beta);
    Vector<f64> resid = yw - fitted;

    // R-squared
    f64 y_mean = yw.mean();
    f64 ss_res = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ss_res += resid[i] * resid[i];
        f64 dev = yw[i] - y_mean;
        ss_tot += dev * dev;
    }
    result.r2 = (ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 0.0;
    result.sigma = std::sqrt(ss_res / static_cast<f64>(n - k));

    // Covariance
    Matrix<f64> XtX = matmul_AtB(Xw, Xw);
    auto XtX_inv_res = invert_spd(XtX);
    if (!XtX_inv_res) return result;
    Matrix<f64> XtX_inv = std::move(XtX_inv_res).value();

    Matrix<f64> cov;
    switch (cov_type) {
        case CovarianceType::Classical:
            cov = classical_covariance(Xw, resid, n, k);
            break;
        case CovarianceType::White:
            cov = white_covariance(Xw, resid, XtX_inv);
            break;
        case CovarianceType::NeweyWest:
            cov = newey_west_covariance(Xw, resid, XtX_inv, hac_lag);
            break;
        default:
            cov = classical_covariance(Xw, resid, n, k);
            break;
    }

    result.se = Vector<f64>(k);
    result.t = Vector<f64>(k);
    for (std::size_t j = 0; j < k; ++j) {
        result.se[j] = std::sqrt(std::max(0.0, cov(j, j)));
        result.t[j] = (result.se[j] > 0.0) ? (result.beta[j] / result.se[j]) : 0.0;
    }

    return result;
}

} // namespace

Result<RollingOLSResult> rolling_ols(const Vector<f64>& y, const Matrix<f64>& X,
                                      const RollingOptions& opts) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t k = X.cols();
    std::size_t window = opts.window;
    std::size_t step = std::max<std::size_t>(opts.step, 1);
    std::size_t min_obs = (opts.min_obs > 0) ? opts.min_obs : window;

    if (n != X.rows()) {
        return Status::error(ErrorCode::DimensionMismatch, "rolling_ols: y.size != X.rows");
    }
    if (n < min_obs) {
        return Status::error(ErrorCode::InvalidArgument, "rolling_ols: n < min_obs");
    }
    if (window <= k) {
        return Status::error(ErrorCode::InvalidArgument, "rolling_ols: window <= k");
    }

    // Count windows
    std::vector<std::size_t> starts, ends;
    for (std::size_t i = 0; i + window <= n; i += step) {
        starts.push_back(i);
        ends.push_back(i + window);
    }

    std::size_t n_win = starts.size();
    if (n_win == 0) {
        return Status::error(ErrorCode::InvalidArgument, "rolling_ols: no valid windows");
    }

    RollingOLSResult result;
    result.betas_ = Matrix<f64>(n_win, k, 0.0);
    result.se_ = Matrix<f64>(n_win, k, 0.0);
    result.t_ = Matrix<f64>(n_win, k, 0.0);
    result.r2_ = Vector<f64>(n_win, 0.0);
    result.sigma_ = Vector<f64>(n_win, 0.0);
    result.starts_ = std::move(starts);
    result.ends_ = std::move(ends);
    result.backend_used_ = Backend::CPU;

    for (std::size_t w = 0; w < n_win; ++w) {
        auto wols = fit_window(y, X, result.starts_[w], result.ends_[w],
                               opts.covariance, opts.hac_lag);
        if (wols.valid) {
            for (std::size_t j = 0; j < k; ++j) {
                result.betas_(w, j) = wols.beta[j];
                result.se_(w, j) = wols.se[j];
                result.t_(w, j) = wols.t[j];
            }
            result.r2_[w] = wols.r2;
            result.sigma_[w] = wols.sigma;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_ms_ = std::chrono::duration<f64, std::milli>(end_time - start_time).count();

    return result;
}

Result<RollingOLSResult> expanding_ols(const Vector<f64>& y, const Matrix<f64>& X,
                                        const RollingOptions& opts) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t k = X.cols();
    std::size_t min_obs = (opts.min_obs > 0) ? opts.min_obs : opts.window;
    std::size_t step = std::max<std::size_t>(opts.step, 1);

    if (n != X.rows()) {
        return Status::error(ErrorCode::DimensionMismatch, "expanding_ols: y.size != X.rows");
    }
    if (min_obs <= k) {
        return Status::error(ErrorCode::InvalidArgument, "expanding_ols: min_obs <= k");
    }

    std::vector<std::size_t> starts, ends;
    for (std::size_t end = min_obs; end <= n; end += step) {
        starts.push_back(0);
        ends.push_back(end);
    }

    std::size_t n_win = starts.size();
    if (n_win == 0) {
        return Status::error(ErrorCode::InvalidArgument, "expanding_ols: no valid windows");
    }

    RollingOLSResult result;
    result.betas_ = Matrix<f64>(n_win, k, 0.0);
    result.se_ = Matrix<f64>(n_win, k, 0.0);
    result.t_ = Matrix<f64>(n_win, k, 0.0);
    result.r2_ = Vector<f64>(n_win, 0.0);
    result.sigma_ = Vector<f64>(n_win, 0.0);
    result.starts_ = std::move(starts);
    result.ends_ = std::move(ends);
    result.backend_used_ = Backend::CPU;

    for (std::size_t w = 0; w < n_win; ++w) {
        auto wols = fit_window(y, X, result.starts_[w], result.ends_[w],
                               opts.covariance, opts.hac_lag);
        if (wols.valid) {
            for (std::size_t j = 0; j < k; ++j) {
                result.betas_(w, j) = wols.beta[j];
                result.se_(w, j) = wols.se[j];
                result.t_(w, j) = wols.t[j];
            }
            result.r2_[w] = wols.r2;
            result.sigma_[w] = wols.sigma;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_ms_ = std::chrono::duration<f64, std::milli>(end_time - start_time).count();

    return result;
}

} // namespace hfm
