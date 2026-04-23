#pragma once

#include <vector>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"

namespace hfm {

struct RollingOptions {
    std::size_t window = 250;
    std::size_t step = 1;
    std::size_t min_obs = 0;  // 0 = use window
    CovarianceType covariance = CovarianceType::Classical;
    i64 hac_lag = -1;
    Backend backend = Backend::Auto;
    bool store_residuals = false;
    bool expanding = false;

    RollingOptions& set_window(std::size_t w) { window = w; return *this; }
    RollingOptions& set_step(std::size_t s) { step = s; return *this; }
    RollingOptions& set_min_obs(std::size_t m) { min_obs = m; return *this; }
    RollingOptions& set_covariance(CovarianceType c) { covariance = c; return *this; }
    RollingOptions& set_hac_lag(i64 lag) { hac_lag = lag; return *this; }
    RollingOptions& set_backend(Backend b) { backend = b; return *this; }
    RollingOptions& set_store_residuals(bool v) { store_residuals = v; return *this; }
    RollingOptions& set_expanding(bool v) { expanding = v; return *this; }
};

class RollingOLSResult {
public:
    std::size_t n_windows() const { return betas_.rows(); }
    std::size_t n_regressors() const { return betas_.cols(); }

    const Matrix<f64>& betas() const { return betas_; }
    const Matrix<f64>& std_errors() const { return se_; }
    const Matrix<f64>& t_stats() const { return t_; }
    const Vector<f64>& r_squared() const { return r2_; }
    const Vector<f64>& sigma() const { return sigma_; }
    const std::vector<std::size_t>& window_starts() const { return starts_; }
    const std::vector<std::size_t>& window_ends() const { return ends_; }

    f64 elapsed_ms() const { return elapsed_ms_; }
    Backend backend_used() const { return backend_used_; }

    Vector<f64> beta_at(std::size_t window_idx) const;
    Vector<f64> se_at(std::size_t window_idx) const;

private:
    friend Result<RollingOLSResult> rolling_ols(const Vector<f64>& y, const Matrix<f64>& X,
                                                 const RollingOptions& opts);
    friend Result<RollingOLSResult> expanding_ols(const Vector<f64>& y, const Matrix<f64>& X,
                                                   const RollingOptions& opts);

    Matrix<f64> betas_;
    Matrix<f64> se_;
    Matrix<f64> t_;
    Vector<f64> r2_;
    Vector<f64> sigma_;
    std::vector<std::size_t> starts_;
    std::vector<std::size_t> ends_;
    f64 elapsed_ms_ = 0.0;
    Backend backend_used_ = Backend::CPU;
};

Result<RollingOLSResult> rolling_ols(const Vector<f64>& y, const Matrix<f64>& X,
                                      const RollingOptions& opts = RollingOptions{});

Result<RollingOLSResult> expanding_ols(const Vector<f64>& y, const Matrix<f64>& X,
                                        const RollingOptions& opts = RollingOptions{});

} // namespace hfm
