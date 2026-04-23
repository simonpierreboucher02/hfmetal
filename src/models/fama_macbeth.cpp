#include "hfm/models/fama_macbeth.hpp"
#include "hfm/linalg/solver.hpp"
#include <chrono>
#include <cmath>
#include <map>
#include <numbers>

namespace hfm {

Result<FamaMacBethResult> fama_macbeth(const Vector<f64>& y,
                                        const Matrix<f64>& X,
                                        const Vector<i64>& time_ids,
                                        const FamaMacBethOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    std::size_t k = X.cols();

    if (X.rows() != n || time_ids.size() != n) {
        return Status::error(ErrorCode::DimensionMismatch, "fama_macbeth: dimension mismatch");
    }

    // Group observations by time period (ordered)
    std::map<i64, std::vector<std::size_t>> time_groups;
    for (std::size_t i = 0; i < n; ++i) {
        time_groups[time_ids[i]].push_back(i);
    }

    std::size_t T = time_groups.size();
    if (T < 2) {
        return Status::error(ErrorCode::InvalidArgument, "fama_macbeth: need at least 2 time periods");
    }

    // Run cross-sectional OLS for each time period
    Matrix<f64> gamma_series(T, k, 0.0);
    std::size_t t_idx = 0;

    for (auto& [_, indices] : time_groups) {
        std::size_t nt = indices.size();
        if (nt <= k) { ++t_idx; continue; }

        Matrix<f64> Xt(nt, k);
        Vector<f64> yt(nt);
        for (std::size_t i = 0; i < nt; ++i) {
            yt[i] = y[indices[i]];
            for (std::size_t j = 0; j < k; ++j) {
                Xt(i, j) = X(indices[i], j);
            }
        }

        auto beta_res = solve_least_squares(Xt, yt);
        if (beta_res) {
            auto& beta = beta_res.value();
            for (std::size_t j = 0; j < k; ++j) {
                gamma_series(t_idx, j) = beta[j];
            }
        }
        ++t_idx;
    }

    // Average over time
    FamaMacBethResult result;
    result.n_periods = T;
    result.n_factors = k;
    result.gamma_series = gamma_series;
    result.gamma = Vector<f64>(k, 0.0);

    for (std::size_t j = 0; j < k; ++j) {
        f64 sum = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            sum += gamma_series(t, j);
        }
        result.gamma[j] = sum / static_cast<f64>(T);
    }

    // Standard errors
    result.std_errors = Vector<f64>(k);
    result.t_stats = Vector<f64>(k);
    result.p_values = Vector<f64>(k);

    if (opts.newey_west_correction && opts.hac_lag != 0) {
        i64 max_lag = opts.hac_lag;
        if (max_lag < 0) {
            max_lag = static_cast<i64>(std::floor(std::pow(static_cast<f64>(T), 1.0 / 3.0)));
        }

        for (std::size_t j = 0; j < k; ++j) {
            // Newey-West on the gamma time series
            f64 var = 0.0;
            for (std::size_t t = 0; t < T; ++t) {
                f64 d = gamma_series(t, j) - result.gamma[j];
                var += d * d;
            }
            var /= static_cast<f64>(T);

            for (i64 lag = 1; lag <= max_lag; ++lag) {
                f64 w = 1.0 - static_cast<f64>(lag) / (static_cast<f64>(max_lag) + 1.0);
                f64 cov = 0.0;
                for (std::size_t t = static_cast<std::size_t>(lag); t < T; ++t) {
                    cov += (gamma_series(t, j) - result.gamma[j]) *
                           (gamma_series(t - static_cast<std::size_t>(lag), j) - result.gamma[j]);
                }
                cov /= static_cast<f64>(T);
                var += 2.0 * w * cov;
            }

            result.std_errors[j] = std::sqrt(var / static_cast<f64>(T));
        }
    } else {
        // Simple Fama-MacBeth SE
        for (std::size_t j = 0; j < k; ++j) {
            f64 ss = 0.0;
            for (std::size_t t = 0; t < T; ++t) {
                f64 d = gamma_series(t, j) - result.gamma[j];
                ss += d * d;
            }
            result.std_errors[j] = std::sqrt(ss / (static_cast<f64>(T) * static_cast<f64>(T - 1)));
        }
    }

    for (std::size_t j = 0; j < k; ++j) {
        result.t_stats[j] = (result.std_errors[j] > 0.0) ?
            (result.gamma[j] / result.std_errors[j]) : 0.0;
        result.p_values[j] = std::erfc(std::abs(result.t_stats[j]) / std::numbers::sqrt2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
