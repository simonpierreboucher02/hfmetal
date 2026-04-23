#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

enum class BinaryModelType : u32 {
    Logit,
    Probit
};

struct BinaryModelOptions {
    BinaryModelType type = BinaryModelType::Logit;
    bool add_intercept = true;
    std::size_t max_iter = 100;
    f64 tol = 1e-8;
    bool compute_marginal_effects = true;

    BinaryModelOptions& set_type(BinaryModelType t) { type = t; return *this; }
    BinaryModelOptions& set_max_iter(std::size_t m) { max_iter = m; return *this; }
    BinaryModelOptions& set_tol(f64 t) { tol = t; return *this; }
};

struct BinaryModelResult {
    Vector<f64> coefficients;
    Vector<f64> std_errors;
    Vector<f64> t_stats;
    Vector<f64> p_values;
    Vector<f64> marginal_effects;     // at mean
    Vector<f64> predicted_prob;       // P(y=1 | X)
    Matrix<f64> covariance_matrix;
    f64 log_likelihood = 0.0;
    f64 log_likelihood_null = 0.0;
    f64 pseudo_r_squared = 0.0;       // McFadden
    f64 aic = 0.0;
    f64 bic = 0.0;
    std::size_t n_obs = 0;
    std::size_t n_regressors = 0;
    std::size_t n_iter = 0;
    bool converged = false;
    BinaryModelType type = BinaryModelType::Logit;
    f64 elapsed_ms = 0.0;
};

// y must be 0/1 vector
Result<BinaryModelResult> logit(const Vector<f64>& y, const Matrix<f64>& X,
                                 const BinaryModelOptions& opts = BinaryModelOptions{});

Result<BinaryModelResult> probit(const Vector<f64>& y, const Matrix<f64>& X,
                                  const BinaryModelOptions& opts = BinaryModelOptions{});

Result<BinaryModelResult> binary_model(const Vector<f64>& y, const Matrix<f64>& X,
                                        const BinaryModelOptions& opts = BinaryModelOptions{});

} // namespace hfm
