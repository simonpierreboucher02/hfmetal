#include "hfm/estimators/gls.hpp"
#include "hfm/linalg/solver.hpp"
#include "hfm/covariance/covariance.hpp"
#include <chrono>
#include <cmath>
#include <numbers>

namespace hfm {

namespace {

f64 t_pvalue(f64 t_stat) {
    return std::erfc(std::abs(t_stat) / std::numbers::sqrt2);
}

} // namespace

Result<GLSResult> gls(const Vector<f64>& y, const Matrix<f64>& X,
                       const Matrix<f64>& Omega, const GLSOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t k_base = X.cols();

    if (n != X.rows() || Omega.rows() != n || Omega.cols() != n) {
        return Status::error(ErrorCode::DimensionMismatch, "gls: dimension mismatch");
    }

    const Matrix<f64>* Xptr = &X;
    Matrix<f64> X_with_intercept;
    if (opts.add_intercept) {
        X_with_intercept = Matrix<f64>(n, k_base + 1);
        for (std::size_t i = 0; i < n; ++i) {
            X_with_intercept(i, 0) = 1.0;
            for (std::size_t j = 0; j < k_base; ++j)
                X_with_intercept(i, j + 1) = X(i, j);
        }
        Xptr = &X_with_intercept;
    }
    std::size_t k = Xptr->cols();

    // Compute Omega^{-1} via Cholesky
    auto chol_result = cholesky(Omega);
    if (!chol_result) {
        return Status::error(ErrorCode::SingularMatrix, "gls: Omega not positive definite");
    }

    // Omega_inv = (L L')^{-1}
    auto Omega_inv = invert_spd(Omega);
    if (!Omega_inv) {
        return Omega_inv.status();
    }

    // X' Omega^{-1} X
    auto Omega_inv_X = matmul(Omega_inv.value(), *Xptr);
    auto XtOiX = matmul_AtB(*Xptr, Omega_inv_X);
    auto XtOiX_inv = invert_spd(XtOiX);
    if (!XtOiX_inv) {
        return XtOiX_inv.status();
    }

    // X' Omega^{-1} y
    auto Omega_inv_y = matvec(Omega_inv.value(), y);
    auto XtOiy = Vector<f64>(k, 0.0);
    for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            XtOiy[j] += (*Xptr)(i, j) * Omega_inv_y[i];
        }
    }

    GLSResult result;
    result.beta_ = matvec(XtOiX_inv.value(), XtOiy);
    result.n_ = n;
    result.k_ = k;

    result.residuals_ = y - matvec(*Xptr, result.beta_);

    // GLS covariance: (X' Omega^{-1} X)^{-1}
    result.cov_ = XtOiX_inv.value();

    // R-squared
    f64 y_mean = y.mean();
    f64 ss_res = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ss_res += result.residuals_[i] * result.residuals_[i];
        f64 dev = y[i] - y_mean;
        ss_tot += dev * dev;
    }
    result.r2_ = (ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 0.0;

    result.se_ = Vector<f64>(k);
    result.t_ = Vector<f64>(k);
    result.pval_ = Vector<f64>(k);
    for (std::size_t j = 0; j < k; ++j) {
        result.se_[j] = std::sqrt(std::max(0.0, result.cov_(j, j)));
        result.t_[j] = (result.se_[j] > 0.0) ? (result.beta_[j] / result.se_[j]) : 0.0;
        result.pval_[j] = t_pvalue(result.t_[j]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms_ = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

Result<GLSResult> fgls(const Vector<f64>& y, const Matrix<f64>& X,
                        const GLSOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();

    // Step 1: OLS to get residuals
    OLSOptions ols_opts;
    ols_opts.add_intercept = opts.add_intercept;
    auto ols_res = ols(y, X, ols_opts);
    if (!ols_res) return ols_res.status();

    auto& resid = ols_res.value().residuals();

    // Step 2: Estimate Omega = diag(e_i^2) — heteroskedastic case
    Matrix<f64> Omega(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        f64 e2 = resid[i] * resid[i];
        Omega(i, i) = (e2 > 1e-15) ? e2 : 1e-15;
    }

    // Step 3: GLS with estimated Omega
    auto gls_result = gls(y, X, Omega, opts);
    if (!gls_result) return gls_result;

    auto end = std::chrono::high_resolution_clock::now();
    auto res = std::move(gls_result).value();
    res.elapsed_ms_ = std::chrono::duration<f64, std::milli>(end - start).count();

    return res;
}

} // namespace hfm
