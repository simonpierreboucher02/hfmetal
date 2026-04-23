#pragma once

#include <string>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

enum class CovarianceType : u32 {
    Classical,
    White,
    NeweyWest,
    ClusteredOneWay,
    ClusteredTwoWay
};

struct OLSOptions {
    Backend backend = Backend::Auto;
    CovarianceType covariance = CovarianceType::Classical;
    i64 hac_lag = -1;  // -1 = auto
    bool add_intercept = false;

    OLSOptions& set_backend(Backend b) { backend = b; return *this; }
    OLSOptions& set_covariance(CovarianceType c) { covariance = c; return *this; }
    OLSOptions& set_hac_lag(i64 lag) { hac_lag = lag; return *this; }
    OLSOptions& set_add_intercept(bool v) { add_intercept = v; return *this; }
};

class OLSResult {
public:
    OLSResult() = default;

    const Vector<f64>& coefficients() const { return beta_; }
    const Vector<f64>& residuals() const { return residuals_; }
    const Vector<f64>& fitted_values() const { return fitted_; }
    const Matrix<f64>& covariance_matrix() const { return cov_; }
    const Vector<f64>& std_errors() const { return se_; }
    const Vector<f64>& t_stats() const { return t_; }
    const Vector<f64>& p_values() const { return pval_; }

    std::size_t n_obs() const { return n_; }
    std::size_t n_regressors() const { return k_; }
    f64 r_squared() const { return r2_; }
    f64 adj_r_squared() const { return adj_r2_; }
    f64 sigma() const { return sigma_; }
    Backend backend_used() const { return backend_used_; }
    f64 elapsed_ms() const { return elapsed_ms_; }

    std::string summary() const;

private:
    friend Result<OLSResult> ols(const Vector<f64>& y, const Matrix<f64>& X,
                                  const OLSOptions& opts);

    Vector<f64> beta_;
    Vector<f64> residuals_;
    Vector<f64> fitted_;
    Matrix<f64> cov_;
    Vector<f64> se_;
    Vector<f64> t_;
    Vector<f64> pval_;
    std::size_t n_ = 0;
    std::size_t k_ = 0;
    f64 r2_ = 0.0;
    f64 adj_r2_ = 0.0;
    f64 sigma_ = 0.0;
    Backend backend_used_ = Backend::CPU;
    f64 elapsed_ms_ = 0.0;
};

Result<OLSResult> ols(const Vector<f64>& y, const Matrix<f64>& X,
                       const OLSOptions& opts = OLSOptions{});

} // namespace hfm
