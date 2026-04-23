#include "hfm/estimators/ols.hpp"
#include "hfm/linalg/solver.hpp"
#include "hfm/covariance/covariance.hpp"
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <numbers>

namespace hfm {

namespace {

f64 t_distribution_pvalue(f64 t_stat, std::size_t df) {
    // Approximation using normal for large df
    f64 abs_t = std::abs(t_stat);
    f64 p = std::erfc(abs_t / std::numbers::sqrt2);
    return p;
}

} // namespace

Result<OLSResult> ols(const Vector<f64>& y, const Matrix<f64>& X,
                       const OLSOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t k_base = X.cols();

    if (n != X.rows()) {
        return Status::error(ErrorCode::DimensionMismatch, "ols: y.size != X.rows");
    }
    if (n <= k_base) {
        return Status::error(ErrorCode::InvalidArgument, "ols: n <= k");
    }

    const Matrix<f64>* Xptr = &X;
    Matrix<f64> X_with_intercept;

    if (opts.add_intercept) {
        X_with_intercept = Matrix<f64>(n, k_base + 1);
        for (std::size_t i = 0; i < n; ++i) {
            X_with_intercept(i, 0) = 1.0;
            for (std::size_t j = 0; j < k_base; ++j) {
                X_with_intercept(i, j + 1) = X(i, j);
            }
        }
        Xptr = &X_with_intercept;
    }

    std::size_t k = Xptr->cols();

    auto beta_result = solve_least_squares(*Xptr, y);
    if (!beta_result) {
        return beta_result.status();
    }

    OLSResult result;
    result.beta_ = std::move(beta_result).value();
    result.n_ = n;
    result.k_ = k;
    result.backend_used_ = Backend::CPU;

    // Fitted values and residuals
    result.fitted_ = matvec(*Xptr, result.beta_);
    result.residuals_ = y - result.fitted_;

    // R-squared
    f64 y_mean = y.mean();
    f64 ss_res = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ss_res += result.residuals_[i] * result.residuals_[i];
        f64 dev = y[i] - y_mean;
        ss_tot += dev * dev;
    }
    result.r2_ = (ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 0.0;
    result.adj_r2_ = 1.0 - (1.0 - result.r2_) * static_cast<f64>(n - 1) / static_cast<f64>(n - k);
    result.sigma_ = std::sqrt(ss_res / static_cast<f64>(n - k));

    // Covariance matrix
    Matrix<f64> XtX = matmul_AtB(*Xptr, *Xptr);
    auto XtX_inv_result = invert_spd(XtX);
    if (!XtX_inv_result) {
        return XtX_inv_result.status();
    }
    Matrix<f64> XtX_inv = std::move(XtX_inv_result).value();

    switch (opts.covariance) {
        case CovarianceType::Classical:
            result.cov_ = classical_covariance(*Xptr, result.residuals_, n, k);
            break;
        case CovarianceType::White:
            result.cov_ = white_covariance(*Xptr, result.residuals_, XtX_inv);
            break;
        case CovarianceType::NeweyWest:
            result.cov_ = newey_west_covariance(*Xptr, result.residuals_, XtX_inv, opts.hac_lag);
            break;
        default:
            return Status::error(ErrorCode::NotImplemented, "covariance type not yet implemented");
    }

    // SE, t-stats, p-values
    result.se_ = Vector<f64>(k);
    result.t_ = Vector<f64>(k);
    result.pval_ = Vector<f64>(k);
    for (std::size_t j = 0; j < k; ++j) {
        result.se_[j] = std::sqrt(std::max(0.0, result.cov_(j, j)));
        result.t_[j] = (result.se_[j] > 0.0) ? (result.beta_[j] / result.se_[j]) : 0.0;
        result.pval_[j] = t_distribution_pvalue(result.t_[j], n - k);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms_ = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

std::string OLSResult::summary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "===== OLS Regression =====\n";
    ss << "N = " << n_ << ", K = " << k_ << "\n";
    ss << "R² = " << r2_ << ", Adj R² = " << adj_r2_ << "\n";
    ss << "σ = " << sigma_ << "\n";
    ss << "Backend: " << static_cast<int>(backend_used_) << "\n";
    ss << "Time: " << elapsed_ms_ << " ms\n\n";
    ss << std::setw(12) << "Coef"
       << std::setw(12) << "SE"
       << std::setw(12) << "t-stat"
       << std::setw(12) << "p-value" << "\n";
    ss << std::string(48, '-') << "\n";
    for (std::size_t j = 0; j < k_; ++j) {
        ss << std::setw(12) << beta_[j]
           << std::setw(12) << se_[j]
           << std::setw(12) << t_[j]
           << std::setw(12) << pval_[j] << "\n";
    }
    return ss.str();
}

} // namespace hfm
