#include "hfm/estimators/iv.hpp"
#include "hfm/linalg/solver.hpp"
#include "hfm/covariance/covariance.hpp"
#include <chrono>
#include <cmath>
#include <numbers>

namespace hfm {

Result<IVResult> iv_2sls(const Vector<f64>& y, const Matrix<f64>& X,
                          const Matrix<f64>& Z, const IVOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t k_base = X.cols();
    std::size_t l_base = Z.cols();

    if (n != X.rows() || n != Z.rows()) {
        return Status::error(ErrorCode::DimensionMismatch, "iv_2sls: dimension mismatch");
    }

    const Matrix<f64>* Xptr = &X;
    const Matrix<f64>* Zptr = &Z;
    Matrix<f64> X_int, Z_int;

    if (opts.add_intercept) {
        X_int = Matrix<f64>(n, k_base + 1);
        Z_int = Matrix<f64>(n, l_base + 1);
        for (std::size_t i = 0; i < n; ++i) {
            X_int(i, 0) = 1.0;
            Z_int(i, 0) = 1.0;
            for (std::size_t j = 0; j < k_base; ++j) X_int(i, j + 1) = X(i, j);
            for (std::size_t j = 0; j < l_base; ++j) Z_int(i, j + 1) = Z(i, j);
        }
        Xptr = &X_int;
        Zptr = &Z_int;
    }

    std::size_t k = Xptr->cols();
    std::size_t l = Zptr->cols();

    if (l < k) {
        return Status::error(ErrorCode::InvalidArgument, "iv_2sls: underidentified (instruments < regressors)");
    }
    if (n <= k) {
        return Status::error(ErrorCode::InvalidArgument, "iv_2sls: n <= k");
    }

    // Stage 1: Project X onto Z
    // X_hat = Z * (Z'Z)^{-1} * Z' * X = P_Z * X
    Matrix<f64> ZtZ = matmul_AtB(*Zptr, *Zptr);
    auto ZtZ_inv = invert_spd(ZtZ);
    if (!ZtZ_inv) {
        return Status::error(ErrorCode::SingularMatrix, "iv_2sls: Z'Z singular");
    }

    // P_Z = Z (Z'Z)^{-1} Z'
    // X_hat = P_Z * X  (column by column)
    Matrix<f64> X_hat(n, k, 0.0);
    for (std::size_t j = 0; j < k; ++j) {
        // Extract column j of X
        Vector<f64> x_j(n);
        for (std::size_t i = 0; i < n; ++i) x_j[i] = (*Xptr)(i, j);

        // Z' * x_j
        Vector<f64> Zt_xj(l, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t m = 0; m < l; ++m) {
                Zt_xj[m] += (*Zptr)(i, m) * x_j[i];
            }
        }

        // (Z'Z)^{-1} * Z'x_j
        auto gamma = matvec(ZtZ_inv.value(), Zt_xj);

        // X_hat_j = Z * gamma
        for (std::size_t i = 0; i < n; ++i) {
            f64 v = 0.0;
            for (std::size_t m = 0; m < l; ++m) {
                v += (*Zptr)(i, m) * gamma[m];
            }
            X_hat(i, j) = v;
        }
    }

    // Stage 2: OLS of y on X_hat
    // beta_2sls = (X_hat'X)^{-1} X_hat'y
    Matrix<f64> XhX = matmul_AtB(X_hat, *Xptr);
    auto XhX_inv = invert_spd(XhX);
    if (!XhX_inv) {
        return Status::error(ErrorCode::SingularMatrix, "iv_2sls: X_hat'X singular");
    }

    // X_hat' y
    Vector<f64> Xh_y(k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            Xh_y[j] += X_hat(i, j) * y[i];
        }
    }

    IVResult result;
    result.beta_ = matvec(XhX_inv.value(), Xh_y);
    result.n_ = n;
    result.k_ = k;
    result.n_instr_ = l;

    // Residuals (using original X, not X_hat)
    result.residuals_ = Vector<f64>(n);
    f64 y_mean = y.mean();
    f64 ss_res = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 fitted = 0.0;
        for (std::size_t j = 0; j < k; ++j)
            fitted += (*Xptr)(i, j) * result.beta_[j];
        result.residuals_[i] = y[i] - fitted;
        ss_res += result.residuals_[i] * result.residuals_[i];
        f64 dev = y[i] - y_mean;
        ss_tot += dev * dev;
    }
    result.r2_ = (ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 0.0;

    // Covariance: sigma^2 * (X_hat'X)^{-1} * X_hat'X_hat * (X_hat'X)^{-1}
    // or robust versions
    f64 sigma2 = ss_res / static_cast<f64>(n - k);

    // For robust SE, use the sandwich with X_hat
    Matrix<f64> XhXh = matmul_AtB(X_hat, X_hat);
    auto XhXh_inv = invert_spd(XhXh);

    switch (opts.covariance) {
        case CovarianceType::White:
            if (XhXh_inv) {
                result.cov_ = white_covariance(X_hat, result.residuals_, XhXh_inv.value());
            }
            break;
        case CovarianceType::NeweyWest:
            if (XhXh_inv) {
                result.cov_ = newey_west_covariance(X_hat, result.residuals_, XhXh_inv.value(), opts.hac_lag);
            }
            break;
        default: {
            // Classical 2SLS variance
            result.cov_ = Matrix<f64>(k, k);
            auto temp = matmul(XhX_inv.value(), XhXh);
            auto cov = matmul(temp, XhX_inv.value());
            for (std::size_t i = 0; i < k * k; ++i) {
                result.cov_.data()[i] = cov.data()[i] * sigma2;
            }
            break;
        }
    }

    // SE, t-stats, p-values
    result.se_ = Vector<f64>(k);
    result.t_ = Vector<f64>(k);
    result.pval_ = Vector<f64>(k);
    for (std::size_t j = 0; j < k; ++j) {
        result.se_[j] = std::sqrt(std::max(0.0, result.cov_(j, j)));
        result.t_[j] = (result.se_[j] > 0.0) ? (result.beta_[j] / result.se_[j]) : 0.0;
        result.pval_[j] = std::erfc(std::abs(result.t_[j]) / std::numbers::sqrt2);
    }

    // First-stage diagnostics (F-stat for the first endogenous regressor)
    {
        std::size_t first_endog = opts.add_intercept ? 1 : 0;
        if (first_endog < k) {
            Vector<f64> x1(n);
            for (std::size_t i = 0; i < n; ++i) x1[i] = (*Xptr)(i, first_endog);

            // First stage R²
            f64 x1_mean = x1.mean();
            f64 ss_tot_fs = 0.0, ss_res_fs = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                f64 dev = x1[i] - x1_mean;
                ss_tot_fs += dev * dev;
                f64 resid_fs = x1[i] - X_hat(i, first_endog);
                ss_res_fs += resid_fs * resid_fs;
            }
            result.first_stage_.r_squared = (ss_tot_fs > 0) ? (1.0 - ss_res_fs / ss_tot_fs) : 0.0;

            f64 df1 = static_cast<f64>(l - k + 1);
            f64 df2 = static_cast<f64>(n - l);
            if (df1 > 0 && df2 > 0 && result.first_stage_.r_squared < 1.0) {
                result.first_stage_.f_statistic = (result.first_stage_.r_squared / df1) /
                    ((1.0 - result.first_stage_.r_squared) / df2);
            }
            result.first_stage_.weak_instrument = result.first_stage_.f_statistic < 10.0;
        }
    }

    // Sargan test for overidentification
    if (l > k) {
        // Regress residuals on instruments
        auto u_on_Z = solve_least_squares(*Zptr, result.residuals_);
        if (u_on_Z) {
            auto u_fitted = matvec(*Zptr, u_on_Z.value());
            f64 ss_u = 0.0;
            for (std::size_t i = 0; i < n; ++i) ss_u += u_fitted[i] * u_fitted[i];
            result.sargan_ = static_cast<f64>(n) * ss_u / ss_res;
            result.sargan_pval_ = std::erfc(std::sqrt(result.sargan_ / 2.0));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms_ = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
