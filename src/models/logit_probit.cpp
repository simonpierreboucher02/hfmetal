#include "hfm/models/logit_probit.hpp"
#include "hfm/linalg/solver.hpp"
#include <chrono>
#include <cmath>
#include <numbers>

namespace hfm {

namespace {

f64 logistic(f64 z) {
    if (z > 500.0) return 1.0;
    if (z < -500.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

f64 normal_cdf(f64 z) {
    return 0.5 * std::erfc(-z / std::numbers::sqrt2);
}

f64 normal_pdf(f64 z) {
    return std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::numbers::pi);
}

f64 link_fn(f64 z, BinaryModelType type) {
    return (type == BinaryModelType::Logit) ? logistic(z) : normal_cdf(z);
}

f64 link_deriv(f64 z, BinaryModelType type) {
    if (type == BinaryModelType::Logit) {
        f64 p = logistic(z);
        return p * (1.0 - p);
    }
    return normal_pdf(z);
}

} // namespace

Result<BinaryModelResult> binary_model(const Vector<f64>& y, const Matrix<f64>& X,
                                        const BinaryModelOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t k_base = X.cols();

    if (n != X.rows()) {
        return Status::error(ErrorCode::DimensionMismatch, "binary_model: y.size != X.rows");
    }

    // Validate y is 0/1
    for (std::size_t i = 0; i < n; ++i) {
        if (y[i] != 0.0 && y[i] != 1.0) {
            return Status::error(ErrorCode::InvalidArgument, "binary_model: y must be 0 or 1");
        }
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

    // IRLS (iteratively reweighted least squares)
    Vector<f64> beta(k, 0.0);
    bool converged = false;
    std::size_t iter = 0;

    for (; iter < opts.max_iter; ++iter) {
        Vector<f64> z_vec(n);     // X * beta
        Vector<f64> p_vec(n);     // link(z)
        Vector<f64> w_vec(n);     // weights
        Vector<f64> z_tilde(n);   // working response

        for (std::size_t i = 0; i < n; ++i) {
            f64 z = 0.0;
            for (std::size_t j = 0; j < k; ++j)
                z += (*Xptr)(i, j) * beta[j];
            z_vec[i] = z;
            p_vec[i] = link_fn(z, opts.type);
            f64 dp = link_deriv(z, opts.type);
            dp = std::max(dp, 1e-10);
            w_vec[i] = dp * dp / std::max(p_vec[i] * (1.0 - p_vec[i]), 1e-10);
            z_tilde[i] = z + (y[i] - p_vec[i]) / dp;
        }

        // Weighted least squares: (X'WX)^{-1} X'W z_tilde
        Matrix<f64> XtWX(k, k, 0.0);
        Vector<f64> XtWz(k, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < k; ++j) {
                XtWz[j] += w_vec[i] * (*Xptr)(i, j) * z_tilde[i];
                for (std::size_t l = 0; l < k; ++l) {
                    XtWX(j, l) += w_vec[i] * (*Xptr)(i, j) * (*Xptr)(i, l);
                }
            }
        }

        auto new_beta_result = solve_spd(XtWX, XtWz);
        if (!new_beta_result) {
            return Status::error(ErrorCode::SingularMatrix, "binary_model: singular information matrix");
        }
        auto& new_beta = new_beta_result.value();

        f64 max_change = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            max_change = std::max(max_change, std::abs(new_beta[j] - beta[j]));
        }
        beta = std::move(new_beta);

        if (max_change < opts.tol) {
            converged = true;
            break;
        }
    }

    BinaryModelResult result;
    result.coefficients = beta;
    result.n_obs = n;
    result.n_regressors = k;
    result.n_iter = iter + 1;
    result.converged = converged;
    result.type = opts.type;

    // Final predictions and log-likelihood
    result.predicted_prob = Vector<f64>(n);
    result.log_likelihood = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 z = 0.0;
        for (std::size_t j = 0; j < k; ++j)
            z += (*Xptr)(i, j) * beta[j];
        f64 p = link_fn(z, opts.type);
        p = std::clamp(p, 1e-15, 1.0 - 1e-15);
        result.predicted_prob[i] = p;
        result.log_likelihood += y[i] * std::log(p) + (1.0 - y[i]) * std::log(1.0 - p);
    }

    // Null model log-likelihood
    f64 p_bar = 0.0;
    for (std::size_t i = 0; i < n; ++i) p_bar += y[i];
    p_bar /= static_cast<f64>(n);
    p_bar = std::clamp(p_bar, 1e-15, 1.0 - 1e-15);
    result.log_likelihood_null = static_cast<f64>(n) *
        (p_bar * std::log(p_bar) + (1.0 - p_bar) * std::log(1.0 - p_bar));

    result.pseudo_r_squared = 1.0 - result.log_likelihood / result.log_likelihood_null;
    result.aic = -2.0 * result.log_likelihood + 2.0 * static_cast<f64>(k);
    result.bic = -2.0 * result.log_likelihood + std::log(static_cast<f64>(n)) * static_cast<f64>(k);

    // Covariance from information matrix
    Matrix<f64> info(k, k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        f64 z = 0.0;
        for (std::size_t j = 0; j < k; ++j)
            z += (*Xptr)(i, j) * beta[j];
        f64 dp = link_deriv(z, opts.type);
        f64 p = result.predicted_prob[i];
        f64 w = dp * dp / std::max(p * (1.0 - p), 1e-10);
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t l = 0; l < k; ++l) {
                info(j, l) += w * (*Xptr)(i, j) * (*Xptr)(i, l);
            }
        }
    }

    auto info_inv = invert_spd(info);
    if (info_inv) {
        result.covariance_matrix = info_inv.value();
        result.std_errors = Vector<f64>(k);
        result.t_stats = Vector<f64>(k);
        result.p_values = Vector<f64>(k);
        for (std::size_t j = 0; j < k; ++j) {
            result.std_errors[j] = std::sqrt(std::max(0.0, result.covariance_matrix(j, j)));
            result.t_stats[j] = (result.std_errors[j] > 0.0)
                ? beta[j] / result.std_errors[j] : 0.0;
            result.p_values[j] = std::erfc(std::abs(result.t_stats[j]) / std::numbers::sqrt2);
        }
    }

    // Marginal effects at mean
    if (opts.compute_marginal_effects) {
        result.marginal_effects = Vector<f64>(k, 0.0);
        f64 z_mean = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            f64 x_mean = 0.0;
            for (std::size_t i = 0; i < n; ++i) x_mean += (*Xptr)(i, j);
            x_mean /= static_cast<f64>(n);
            z_mean += beta[j] * x_mean;
        }
        f64 dp_mean = link_deriv(z_mean, opts.type);
        for (std::size_t j = 0; j < k; ++j) {
            result.marginal_effects[j] = dp_mean * beta[j];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

Result<BinaryModelResult> logit(const Vector<f64>& y, const Matrix<f64>& X,
                                 const BinaryModelOptions& opts) {
    auto o = opts;
    o.type = BinaryModelType::Logit;
    return binary_model(y, X, o);
}

Result<BinaryModelResult> probit(const Vector<f64>& y, const Matrix<f64>& X,
                                  const BinaryModelOptions& opts) {
    auto o = opts;
    o.type = BinaryModelType::Probit;
    return binary_model(y, X, o);
}

} // namespace hfm
