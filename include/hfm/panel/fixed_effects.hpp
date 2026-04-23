#pragma once

#include <vector>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/estimators/ols.hpp"
#include "hfm/covariance/covariance.hpp"

namespace hfm {

enum class PanelEffect : u32 {
    Entity,
    Time,
    TwoWay
};

struct PanelOptions {
    PanelEffect effect = PanelEffect::Entity;
    bool cluster_entity = true;
    bool cluster_time = false;
    Backend backend = Backend::Auto;

    PanelOptions& set_effect(PanelEffect e) { effect = e; return *this; }
    PanelOptions& set_cluster_entity(bool v) { cluster_entity = v; return *this; }
    PanelOptions& set_cluster_time(bool v) { cluster_time = v; return *this; }
};

class PanelResult {
public:
    const Vector<f64>& coefficients() const { return beta_; }
    const Matrix<f64>& covariance_matrix() const { return cov_; }
    const Vector<f64>& std_errors() const { return se_; }
    const Vector<f64>& t_stats() const { return t_; }
    const Vector<f64>& p_values() const { return pval_; }
    const Vector<f64>& residuals() const { return residuals_; }

    std::size_t n_obs() const { return n_; }
    std::size_t n_groups() const { return n_groups_; }
    std::size_t n_regressors() const { return k_; }
    f64 r_squared() const { return r2_; }
    f64 r_squared_within() const { return r2_within_; }
    f64 elapsed_ms() const { return elapsed_ms_; }

    std::string summary() const;

private:
    friend Result<PanelResult> fixed_effects(const Vector<f64>& y, const Matrix<f64>& X,
                                              const Vector<i64>& entity_ids,
                                              const Vector<i64>& time_ids,
                                              const PanelOptions& opts);

    Vector<f64> beta_;
    Matrix<f64> cov_;
    Vector<f64> se_;
    Vector<f64> t_;
    Vector<f64> pval_;
    Vector<f64> residuals_;
    std::size_t n_ = 0;
    std::size_t n_groups_ = 0;
    std::size_t k_ = 0;
    f64 r2_ = 0.0;
    f64 r2_within_ = 0.0;
    f64 elapsed_ms_ = 0.0;
};

Result<PanelResult> fixed_effects(const Vector<f64>& y, const Matrix<f64>& X,
                                   const Vector<i64>& entity_ids,
                                   const Vector<i64>& time_ids,
                                   const PanelOptions& opts = PanelOptions{});

} // namespace hfm
