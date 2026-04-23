#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"

namespace hfm {

struct IVOptions {
    CovarianceType covariance = CovarianceType::Classical;
    i64 hac_lag = -1;
    bool add_intercept = false;

    IVOptions& set_covariance(CovarianceType c) { covariance = c; return *this; }
    IVOptions& set_hac_lag(i64 lag) { hac_lag = lag; return *this; }
    IVOptions& set_add_intercept(bool v) { add_intercept = v; return *this; }
};

struct IVFirstStage {
    f64 f_statistic = 0.0;
    f64 r_squared = 0.0;
    f64 partial_r_squared = 0.0;
    bool weak_instrument = false;   // F < 10 rule of thumb
};

class IVResult {
public:
    const Vector<f64>& coefficients() const { return beta_; }
    const Vector<f64>& residuals() const { return residuals_; }
    const Matrix<f64>& covariance_matrix() const { return cov_; }
    const Vector<f64>& std_errors() const { return se_; }
    const Vector<f64>& t_stats() const { return t_; }
    const Vector<f64>& p_values() const { return pval_; }

    std::size_t n_obs() const { return n_; }
    std::size_t n_regressors() const { return k_; }
    std::size_t n_instruments() const { return n_instr_; }
    f64 r_squared() const { return r2_; }

    const IVFirstStage& first_stage() const { return first_stage_; }
    f64 sargan_stat() const { return sargan_; }
    f64 sargan_pvalue() const { return sargan_pval_; }
    bool overidentified() const { return n_instr_ > k_; }

    f64 elapsed_ms() const { return elapsed_ms_; }

private:
    friend Result<IVResult> iv_2sls(const Vector<f64>& y, const Matrix<f64>& X,
                                     const Matrix<f64>& Z, const IVOptions& opts);

    Vector<f64> beta_;
    Vector<f64> residuals_;
    Matrix<f64> cov_;
    Vector<f64> se_;
    Vector<f64> t_;
    Vector<f64> pval_;
    std::size_t n_ = 0;
    std::size_t k_ = 0;
    std::size_t n_instr_ = 0;
    f64 r2_ = 0.0;
    IVFirstStage first_stage_;
    f64 sargan_ = 0.0;
    f64 sargan_pval_ = 0.0;
    f64 elapsed_ms_ = 0.0;
};

// 2SLS: y = X*beta + e, instruments Z
// X: n x k endogenous/exogenous regressors
// Z: n x l instruments (l >= k for identification)
Result<IVResult> iv_2sls(const Vector<f64>& y, const Matrix<f64>& X,
                          const Matrix<f64>& Z, const IVOptions& opts = IVOptions{});

} // namespace hfm
