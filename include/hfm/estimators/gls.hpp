#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"

namespace hfm {

struct GLSOptions {
    Backend backend = Backend::Auto;
    bool add_intercept = false;

    GLSOptions& set_backend(Backend b) { backend = b; return *this; }
    GLSOptions& set_add_intercept(bool v) { add_intercept = v; return *this; }
};

class GLSResult {
public:
    const Vector<f64>& coefficients() const { return beta_; }
    const Vector<f64>& residuals() const { return residuals_; }
    const Matrix<f64>& covariance_matrix() const { return cov_; }
    const Vector<f64>& std_errors() const { return se_; }
    const Vector<f64>& t_stats() const { return t_; }
    const Vector<f64>& p_values() const { return pval_; }

    std::size_t n_obs() const { return n_; }
    std::size_t n_regressors() const { return k_; }
    f64 r_squared() const { return r2_; }
    f64 elapsed_ms() const { return elapsed_ms_; }

private:
    friend Result<GLSResult> gls(const Vector<f64>& y, const Matrix<f64>& X,
                                  const Matrix<f64>& Omega, const GLSOptions& opts);
    friend Result<GLSResult> fgls(const Vector<f64>& y, const Matrix<f64>& X,
                                   const GLSOptions& opts);

    Vector<f64> beta_;
    Vector<f64> residuals_;
    Matrix<f64> cov_;
    Vector<f64> se_;
    Vector<f64> t_;
    Vector<f64> pval_;
    std::size_t n_ = 0;
    std::size_t k_ = 0;
    f64 r2_ = 0.0;
    f64 elapsed_ms_ = 0.0;
};

// GLS: beta = (X'Omega^{-1}X)^{-1} X'Omega^{-1}y
// Omega is the n x n error covariance matrix
Result<GLSResult> gls(const Vector<f64>& y, const Matrix<f64>& X,
                       const Matrix<f64>& Omega, const GLSOptions& opts = GLSOptions{});

// Feasible GLS: estimates Omega from OLS residuals, then applies GLS
Result<GLSResult> fgls(const Vector<f64>& y, const Matrix<f64>& X,
                        const GLSOptions& opts = GLSOptions{});

} // namespace hfm
