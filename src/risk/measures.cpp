#include "hfm/risk/measures.hpp"
#include "hfm/linalg/solver.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <algorithm>

namespace hfm {

namespace {

f64 normal_quantile(f64 p) {
    // Rational approximation (Abramowitz & Stegun)
    if (p <= 0.0) return -1e10;
    if (p >= 1.0) return 1e10;
    if (p == 0.5) return 0.0;

    bool lower = (p < 0.5);
    f64 pp = lower ? p : (1.0 - p);

    f64 t = std::sqrt(-2.0 * std::log(pp));
    f64 c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    f64 d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    f64 z = t - (c0 + c1 * t + c2 * t * t) /
                 (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    return lower ? -z : z;
}

} // namespace

// ========== VaR / CVaR ==========

Result<VaRResult> value_at_risk(const Vector<f64>& returns, f64 confidence,
                                 VaRMethod method) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();
    if (n < 10) {
        return Status::error(ErrorCode::InvalidArgument,
                             "value_at_risk: need at least 10 observations");
    }
    if (confidence <= 0.0 || confidence >= 1.0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "value_at_risk: confidence must be in (0, 1)");
    }

    VaRResult result;
    result.n_obs = n;
    result.confidence = confidence;
    result.method = method;

    if (method == VaRMethod::Historical) {
        std::vector<f64> sorted(returns.data(), returns.data() + n);
        std::sort(sorted.begin(), sorted.end());

        std::size_t var_idx = static_cast<std::size_t>(
            std::floor((1.0 - confidence) * static_cast<f64>(n)));
        if (var_idx >= n) var_idx = n - 1;
        result.var = -sorted[var_idx];

        f64 cvar_sum = 0.0;
        std::size_t cvar_count = 0;
        for (std::size_t i = 0; i <= var_idx; ++i) {
            cvar_sum += sorted[i];
            ++cvar_count;
        }
        result.cvar = (cvar_count > 0) ? -(cvar_sum / static_cast<f64>(cvar_count)) : result.var;

    } else if (method == VaRMethod::Parametric) {
        f64 mu = returns.mean();
        f64 var = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            f64 d = returns[i] - mu;
            var += d * d;
        }
        var /= static_cast<f64>(n - 1);
        f64 sigma = std::sqrt(var);

        f64 z = normal_quantile(1.0 - confidence);
        result.var = -(mu + z * sigma);

        f64 phi_z = std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::numbers::pi);
        result.cvar = -(mu - sigma * phi_z / (1.0 - confidence));

    } else {
        f64 mu = returns.mean();
        f64 fn = static_cast<f64>(n);
        f64 m2 = 0.0, m3 = 0.0, m4 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            f64 d = returns[i] - mu;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
        }
        f64 s2 = m2 / fn;
        f64 sigma = std::sqrt(m2 / (fn - 1.0));
        f64 skew = (s2 > 0.0) ? (m3 / fn) / std::pow(std::sqrt(s2), 3.0) : 0.0;
        f64 kurt = (s2 > 0.0) ? (m4 / fn) / (s2 * s2) - 3.0 : 0.0;

        f64 z = normal_quantile(1.0 - confidence);
        f64 z_cf = z + (z * z - 1.0) * skew / 6.0 +
                   (z * z * z - 3.0 * z) * kurt / 24.0 -
                   (2.0 * z * z * z - 5.0 * z) * skew * skew / 36.0;

        result.var = -(mu + z_cf * sigma);

        std::vector<f64> sorted(returns.data(), returns.data() + n);
        std::sort(sorted.begin(), sorted.end());
        std::size_t var_idx = static_cast<std::size_t>(
            std::floor((1.0 - confidence) * fn));
        if (var_idx >= n) var_idx = n - 1;
        f64 cvar_sum = 0.0;
        for (std::size_t i = 0; i <= var_idx; ++i) cvar_sum += sorted[i];
        result.cvar = -(cvar_sum / static_cast<f64>(var_idx + 1));
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== Drawdown ==========

Result<DrawdownResult> drawdown_analysis(const Vector<f64>& returns) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();
    if (n < 2) {
        return Status::error(ErrorCode::InvalidArgument,
                             "drawdown_analysis: need at least 2 observations");
    }

    Vector<f64> cumulative(n);
    cumulative[0] = returns[0];
    for (std::size_t i = 1; i < n; ++i) {
        cumulative[i] = cumulative[i - 1] + returns[i];
    }

    Vector<f64> running_max(n);
    running_max[0] = cumulative[0];
    for (std::size_t i = 1; i < n; ++i) {
        running_max[i] = std::max(running_max[i - 1], cumulative[i]);
    }

    DrawdownResult result;
    result.n_obs = n;
    result.drawdown_series.resize(n);
    result.max_drawdown = 0.0;

    f64 dd_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result.drawdown_series[i] = cumulative[i] - running_max[i];
        dd_sum += result.drawdown_series[i];
        if (result.drawdown_series[i] < -result.max_drawdown) {
            result.max_drawdown = -result.drawdown_series[i];
            result.trough_idx = i;
        }
    }
    result.avg_drawdown = -dd_sum / static_cast<f64>(n);

    f64 peak_val = cumulative[0];
    result.peak_idx = 0;
    for (std::size_t i = 0; i <= result.trough_idx; ++i) {
        if (cumulative[i] >= peak_val) {
            peak_val = cumulative[i];
            result.peak_idx = i;
        }
    }

    result.recovery_idx = n;
    for (std::size_t i = result.trough_idx + 1; i < n; ++i) {
        if (cumulative[i] >= peak_val) {
            result.recovery_idx = i;
            break;
        }
    }

    if (result.trough_idx > result.peak_idx) {
        result.max_drawdown_duration = static_cast<f64>(result.trough_idx - result.peak_idx);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== Performance metrics ==========

Result<PerformanceResult> performance_metrics(const Vector<f64>& returns,
                                               const PerformanceOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();
    if (n < 10) {
        return Status::error(ErrorCode::InvalidArgument,
                             "performance_metrics: need at least 10 observations");
    }

    f64 fn = static_cast<f64>(n);
    f64 mu = returns.mean();

    f64 var = 0.0;
    f64 downside_var = 0.0;
    std::size_t downside_count = 0;
    f64 gain_sum = 0.0;
    f64 loss_sum = 0.0;
    f64 daily_rf = opts.risk_free_rate / opts.annualization_factor;

    for (std::size_t i = 0; i < n; ++i) {
        f64 d = returns[i] - mu;
        var += d * d;

        f64 excess = returns[i] - opts.target_return / opts.annualization_factor;
        if (excess < 0.0) {
            downside_var += excess * excess;
            ++downside_count;
            loss_sum += std::abs(excess);
        } else {
            gain_sum += excess;
        }
    }

    var /= (fn - 1.0);
    f64 sigma = std::sqrt(var);
    f64 downside_dev = (downside_count > 0) ?
        std::sqrt(downside_var / fn) : sigma;

    PerformanceResult result;
    result.n_obs = n;
    result.annualized_return = mu * opts.annualization_factor;
    result.annualized_volatility = sigma * std::sqrt(opts.annualization_factor);
    result.downside_deviation = downside_dev * std::sqrt(opts.annualization_factor);

    f64 excess_return = mu - daily_rf;
    result.sharpe_ratio = (sigma > 0.0) ?
        (excess_return / sigma) * std::sqrt(opts.annualization_factor) : 0.0;

    result.sortino_ratio = (downside_dev > 0.0) ?
        (excess_return / downside_dev) * std::sqrt(opts.annualization_factor) : 0.0;

    result.omega_ratio = (loss_sum > 0.0) ? (gain_sum / loss_sum) : 1e10;

    auto dd_res = drawdown_analysis(returns);
    if (dd_res) {
        result.max_drawdown = dd_res.value().max_drawdown;
        result.calmar_ratio = (result.max_drawdown > 0.0) ?
            result.annualized_return / result.max_drawdown : 0.0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== Portfolio optimization ==========

Result<PortfolioResult> minimum_variance_portfolio(const Vector<f64>& expected_returns,
                                                     const Matrix<f64>& cov_matrix,
                                                     const PortfolioOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = expected_returns.size();
    if (n < 2 || cov_matrix.rows() != n || cov_matrix.cols() != n) {
        return Status::error(ErrorCode::DimensionMismatch,
                             "minimum_variance_portfolio: dimension mismatch");
    }

    Vector<f64> ones(n, 1.0);
    auto cov_inv = invert_spd(cov_matrix);
    if (!cov_inv) {
        return Status::error(ErrorCode::SingularMatrix,
                             "minimum_variance_portfolio: covariance matrix not invertible");
    }

    auto w_raw = matvec(cov_inv.value(), ones);
    f64 w_sum = w_raw.sum();

    PortfolioResult result;
    result.n_assets = n;
    result.weights.resize(n);

    if (opts.long_only) {
        for (std::size_t i = 0; i < n; ++i) {
            result.weights[i] = std::max(0.0, w_raw[i]);
        }
        f64 pos_sum = result.weights.sum();
        if (pos_sum > 0.0) {
            for (std::size_t i = 0; i < n; ++i) result.weights[i] /= pos_sum;
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            result.weights[i] = w_raw[i] / w_sum;
        }
    }

    result.expected_return = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result.expected_return += result.weights[i] * expected_returns[i];
    }

    auto Sw = matvec(cov_matrix, result.weights);
    f64 port_var = result.weights.dot(Sw);
    result.volatility = std::sqrt(std::max(0.0, port_var));

    f64 excess = result.expected_return - opts.risk_free_rate;
    result.sharpe_ratio = (result.volatility > 0.0) ? excess / result.volatility : 0.0;

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

Result<PortfolioResult> max_sharpe_portfolio(const Vector<f64>& expected_returns,
                                              const Matrix<f64>& cov_matrix,
                                              const PortfolioOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = expected_returns.size();
    if (n < 2 || cov_matrix.rows() != n || cov_matrix.cols() != n) {
        return Status::error(ErrorCode::DimensionMismatch,
                             "max_sharpe_portfolio: dimension mismatch");
    }

    auto cov_inv = invert_spd(cov_matrix);
    if (!cov_inv) {
        return Status::error(ErrorCode::SingularMatrix,
                             "max_sharpe_portfolio: covariance matrix not invertible");
    }

    Vector<f64> excess(n);
    for (std::size_t i = 0; i < n; ++i) {
        excess[i] = expected_returns[i] - opts.risk_free_rate;
    }

    auto w_raw = matvec(cov_inv.value(), excess);

    PortfolioResult result;
    result.n_assets = n;
    result.weights.resize(n);

    if (opts.long_only) {
        for (std::size_t i = 0; i < n; ++i) {
            result.weights[i] = std::max(0.0, w_raw[i]);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            result.weights[i] = w_raw[i];
        }
    }

    f64 w_sum = result.weights.sum();
    if (std::abs(w_sum) > 1e-15) {
        for (std::size_t i = 0; i < n; ++i) result.weights[i] /= w_sum;
    }

    result.expected_return = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        result.expected_return += result.weights[i] * expected_returns[i];
    }

    auto Sw = matvec(cov_matrix, result.weights);
    f64 port_var = result.weights.dot(Sw);
    result.volatility = std::sqrt(std::max(0.0, port_var));

    f64 excess_ret = result.expected_return - opts.risk_free_rate;
    result.sharpe_ratio = (result.volatility > 0.0) ? excess_ret / result.volatility : 0.0;

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

Result<EfficientFrontierResult> efficient_frontier(const Vector<f64>& expected_returns,
                                                     const Matrix<f64>& cov_matrix,
                                                     std::size_t n_points,
                                                     const PortfolioOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = expected_returns.size();
    if (n < 2) {
        return Status::error(ErrorCode::InvalidArgument,
                             "efficient_frontier: need at least 2 assets");
    }

    auto cov_inv = invert_spd(cov_matrix);
    if (!cov_inv) {
        return Status::error(ErrorCode::SingularMatrix,
                             "efficient_frontier: covariance matrix not invertible");
    }

    f64 min_ret = expected_returns[0], max_ret = expected_returns[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (expected_returns[i] < min_ret) min_ret = expected_returns[i];
        if (expected_returns[i] > max_ret) max_ret = expected_returns[i];
    }

    EfficientFrontierResult result;
    result.n_points = n_points;
    result.n_assets = n;

    Vector<f64> ones(n, 1.0);
    auto Sinv_ones = matvec(cov_inv.value(), ones);
    auto Sinv_mu = matvec(cov_inv.value(), expected_returns);

    f64 A = expected_returns.dot(Sinv_ones);
    f64 B = expected_returns.dot(Sinv_mu);
    f64 C = ones.dot(Sinv_ones);
    f64 D = B * C - A * A;

    if (std::abs(D) < 1e-15) {
        return Status::error(ErrorCode::SingularMatrix,
                             "efficient_frontier: degenerate problem");
    }

    for (std::size_t p = 0; p < n_points; ++p) {
        f64 target = min_ret + (max_ret - min_ret) *
                     static_cast<f64>(p) / static_cast<f64>(n_points - 1);

        f64 lam1 = (C * target - A) / D;
        f64 lam2 = (B - A * target) / D;

        PortfolioResult pr;
        pr.n_assets = n;
        pr.weights.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            pr.weights[i] = lam1 * Sinv_mu[i] + lam2 * Sinv_ones[i];
        }

        if (opts.long_only) {
            for (std::size_t i = 0; i < n; ++i) {
                pr.weights[i] = std::max(0.0, pr.weights[i]);
            }
            f64 ws = pr.weights.sum();
            if (ws > 0.0) {
                for (std::size_t i = 0; i < n; ++i) pr.weights[i] /= ws;
            }
        }

        pr.expected_return = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            pr.expected_return += pr.weights[i] * expected_returns[i];
        }

        auto Sw = matvec(cov_matrix, pr.weights);
        f64 pv = pr.weights.dot(Sw);
        pr.volatility = std::sqrt(std::max(0.0, pv));

        f64 ex = pr.expected_return - opts.risk_free_rate;
        pr.sharpe_ratio = (pr.volatility > 0.0) ? ex / pr.volatility : 0.0;

        result.portfolios.push_back(std::move(pr));
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

} // namespace hfm
