#include "hfm/timeseries/var.hpp"
#include "hfm/linalg/solver.hpp"
#include "hfm/covariance/covariance.hpp"
#include <chrono>
#include <cmath>
#include <numbers>

namespace hfm {

Matrix<f64> var_lag_design_matrix(const Matrix<f64>& Y, std::size_t p, bool intercept) {
    std::size_t T_full = Y.rows();
    std::size_t n_vars = Y.cols();

    if (T_full <= p) return Matrix<f64>(0, 0);

    std::size_t T = T_full - p;
    std::size_t k = n_vars * p + (intercept ? 1 : 0);

    Matrix<f64> X(T, k, 0.0);
    for (std::size_t t = 0; t < T; ++t) {
        std::size_t col = 0;
        if (intercept) {
            X(t, col++) = 1.0;
        }
        for (std::size_t lag = 1; lag <= p; ++lag) {
            for (std::size_t v = 0; v < n_vars; ++v) {
                X(t, col++) = Y(p + t - lag, v);
            }
        }
    }
    return X;
}

Result<VARResult> var(const Matrix<f64>& Y, const VAROptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t T_full = Y.rows();
    std::size_t n_vars = Y.cols();
    std::size_t p = opts.p;

    if (n_vars < 2) {
        return Status::error(ErrorCode::InvalidArgument, "var: need at least 2 variables");
    }
    if (T_full <= p + 1) {
        return Status::error(ErrorCode::InvalidArgument, "var: not enough observations");
    }

    std::size_t T = T_full - p;
    auto X = var_lag_design_matrix(Y, p, opts.add_intercept);
    std::size_t k = X.cols();

    if (T <= k) {
        return Status::error(ErrorCode::InvalidArgument, "var: more parameters than observations");
    }

    VARResult result;
    result.p = p;
    result.n_vars = n_vars;
    result.n_obs = T;
    result.has_intercept = opts.add_intercept;
    result.coefficients = Matrix<f64>(n_vars, k, 0.0);
    result.residuals = Matrix<f64>(T, n_vars, 0.0);
    result.covariance_matrices.resize(n_vars);
    result.std_errors.resize(n_vars);
    result.t_stats.resize(n_vars);

    Matrix<f64> XtX = matmul_AtB(X, X);
    auto XtX_inv_result = invert_spd(XtX);
    if (!XtX_inv_result) {
        return XtX_inv_result.status();
    }
    Matrix<f64> XtX_inv = std::move(XtX_inv_result).value();

    for (std::size_t v = 0; v < n_vars; ++v) {
        Vector<f64> y_v(T);
        for (std::size_t t = 0; t < T; ++t) {
            y_v[t] = Y(p + t, v);
        }

        auto beta_result = solve_least_squares(X, y_v);
        if (!beta_result) return beta_result.status();

        auto& beta = beta_result.value();
        for (std::size_t j = 0; j < k; ++j) {
            result.coefficients(v, j) = beta[j];
        }

        Vector<f64> fitted = matvec(X, beta);
        Vector<f64> resid = y_v - fitted;
        for (std::size_t t = 0; t < T; ++t) {
            result.residuals(t, v) = resid[t];
        }

        switch (opts.covariance) {
            case CovarianceType::Classical:
                result.covariance_matrices[v] = classical_covariance(X, resid, T, k);
                break;
            case CovarianceType::White:
                result.covariance_matrices[v] = white_covariance(X, resid, XtX_inv);
                break;
            case CovarianceType::NeweyWest:
                result.covariance_matrices[v] = newey_west_covariance(X, resid, XtX_inv, opts.hac_lag);
                break;
            default:
                result.covariance_matrices[v] = classical_covariance(X, resid, T, k);
                break;
        }

        result.std_errors[v] = Vector<f64>(k);
        result.t_stats[v] = Vector<f64>(k);
        for (std::size_t j = 0; j < k; ++j) {
            result.std_errors[v][j] = std::sqrt(std::max(0.0, result.covariance_matrices[v](j, j)));
            result.t_stats[v][j] = (result.std_errors[v][j] > 0.0)
                ? (beta[j] / result.std_errors[v][j]) : 0.0;
        }
    }

    // Residual covariance matrix
    result.sigma_u = Matrix<f64>(n_vars, n_vars, 0.0);
    for (std::size_t i = 0; i < n_vars; ++i) {
        for (std::size_t j = 0; j < n_vars; ++j) {
            f64 s = 0.0;
            for (std::size_t t = 0; t < T; ++t) {
                s += result.residuals(t, i) * result.residuals(t, j);
            }
            result.sigma_u(i, j) = s / static_cast<f64>(T);
        }
    }

    // Log-likelihood (multivariate normal)
    f64 log_det = 0.0;
    {
        auto chol = cholesky(result.sigma_u);
        if (chol) {
            for (std::size_t i = 0; i < n_vars; ++i) {
                log_det += 2.0 * std::log(chol.value()(i, i));
            }
        }
    }
    f64 nT = static_cast<f64>(T);
    f64 nv = static_cast<f64>(n_vars);
    result.log_likelihood = -0.5 * nT * (nv * std::log(2.0 * std::numbers::pi) + log_det + nv);

    std::size_t total_params = n_vars * k;
    result.aic = -2.0 * result.log_likelihood + 2.0 * static_cast<f64>(total_params);
    result.bic = -2.0 * result.log_likelihood + std::log(nT) * static_cast<f64>(total_params);

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
