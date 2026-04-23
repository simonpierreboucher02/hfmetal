#include "hfm/panel/fixed_effects.hpp"
#include "hfm/linalg/solver.hpp"
#include "hfm/covariance/covariance.hpp"
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <numbers>

namespace hfm {

namespace {

struct GroupStats {
    f64 sum = 0.0;
    std::size_t count = 0;
    f64 mean() const { return (count > 0) ? sum / static_cast<f64>(count) : 0.0; }
};

void demean_by_group(Vector<f64>& v, const Vector<i64>& group_ids) {
    std::unordered_map<i64, GroupStats> stats;
    for (std::size_t i = 0; i < v.size(); ++i) {
        auto& s = stats[group_ids[i]];
        s.sum += v[i];
        s.count++;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] -= stats[group_ids[i]].mean();
    }
}

void demean_matrix_by_group(Matrix<f64>& M, const Vector<i64>& group_ids) {
    for (std::size_t j = 0; j < M.cols(); ++j) {
        std::unordered_map<i64, GroupStats> stats;
        for (std::size_t i = 0; i < M.rows(); ++i) {
            auto& s = stats[group_ids[i]];
            s.sum += M(i, j);
            s.count++;
        }
        for (std::size_t i = 0; i < M.rows(); ++i) {
            M(i, j) -= stats[group_ids[i]].mean();
        }
    }
}

std::size_t count_unique(const Vector<i64>& ids) {
    std::unordered_map<i64, bool> seen;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        seen[ids[i]] = true;
    }
    return seen.size();
}

} // namespace

Result<PanelResult> fixed_effects(const Vector<f64>& y, const Matrix<f64>& X,
                                   const Vector<i64>& entity_ids,
                                   const Vector<i64>& time_ids,
                                   const PanelOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t k = X.cols();

    if (X.rows() != n || entity_ids.size() != n || time_ids.size() != n) {
        return Status::error(ErrorCode::DimensionMismatch, "fixed_effects: dimension mismatch");
    }

    // Demean
    Vector<f64> y_dm(n);
    Matrix<f64> X_dm(n, k);
    for (std::size_t i = 0; i < n; ++i) {
        y_dm[i] = y[i];
        for (std::size_t j = 0; j < k; ++j) {
            X_dm(i, j) = X(i, j);
        }
    }

    switch (opts.effect) {
        case PanelEffect::Entity:
            demean_by_group(y_dm, entity_ids);
            demean_matrix_by_group(X_dm, entity_ids);
            break;
        case PanelEffect::Time:
            demean_by_group(y_dm, time_ids);
            demean_matrix_by_group(X_dm, time_ids);
            break;
        case PanelEffect::TwoWay:
            demean_by_group(y_dm, entity_ids);
            demean_matrix_by_group(X_dm, entity_ids);
            demean_by_group(y_dm, time_ids);
            demean_matrix_by_group(X_dm, time_ids);
            break;
    }

    // OLS on demeaned data
    auto beta_res = solve_least_squares(X_dm, y_dm);
    if (!beta_res) return beta_res.status();

    PanelResult result;
    result.beta_ = std::move(beta_res).value();
    result.n_ = n;
    result.k_ = k;
    result.n_groups_ = count_unique(entity_ids);

    // Residuals
    auto fitted = matvec(X_dm, result.beta_);
    result.residuals_ = y_dm - fitted;

    // R-squared within
    f64 ss_res = 0.0, ss_tot = 0.0;
    f64 y_dm_mean = y_dm.mean();
    for (std::size_t i = 0; i < n; ++i) {
        ss_res += result.residuals_[i] * result.residuals_[i];
        f64 dev = y_dm[i] - y_dm_mean;
        ss_tot += dev * dev;
    }
    result.r2_within_ = (ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 0.0;
    result.r2_ = result.r2_within_;

    // Covariance
    Matrix<f64> XtX = matmul_AtB(X_dm, X_dm);
    auto XtX_inv_res = invert_spd(XtX);
    if (!XtX_inv_res) return XtX_inv_res.status();
    Matrix<f64> XtX_inv = std::move(XtX_inv_res).value();

    if (opts.cluster_entity && opts.cluster_time) {
        result.cov_ = twoway_clustered_covariance(X_dm, result.residuals_, XtX_inv,
                                                    entity_ids, time_ids);
    } else if (opts.cluster_entity) {
        result.cov_ = clustered_covariance(X_dm, result.residuals_, XtX_inv, entity_ids);
    } else if (opts.cluster_time) {
        result.cov_ = clustered_covariance(X_dm, result.residuals_, XtX_inv, time_ids);
    } else {
        result.cov_ = classical_covariance(X_dm, result.residuals_, n, k);
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

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms_ = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

std::string PanelResult::summary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "===== Panel Fixed Effects =====\n";
    ss << "N = " << n_ << ", Groups = " << n_groups_ << ", K = " << k_ << "\n";
    ss << "R² (within) = " << r2_within_ << "\n";
    ss << "Time: " << elapsed_ms_ << " ms\n\n";
    ss << std::setw(12) << "Coef"
       << std::setw(12) << "SE"
       << std::setw(12) << "t-stat"
       << std::setw(12) << "p-value" << "\n";
    ss << std::string(48, '-') << "\n";
    for (std::size_t j = 0; j < k_; ++j) {
        ss << std::setw(12) << beta_[j]
           << std::setw(12) << se_[j]
           << std::setw(12) << t_[j]
           << std::setw(12) << pval_[j] << "\n";
    }
    return ss.str();
}

} // namespace hfm
