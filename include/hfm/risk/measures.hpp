#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"

namespace hfm {

// ========== VaR / CVaR ==========

enum class VaRMethod : u32 {
    Historical,
    Parametric,
    CornishFisher
};

struct VaRResult {
    f64 var = 0.0;
    f64 cvar = 0.0;
    f64 confidence = 0.95;
    VaRMethod method = VaRMethod::Historical;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

Result<VaRResult> value_at_risk(const Vector<f64>& returns,
                                 f64 confidence = 0.95,
                                 VaRMethod method = VaRMethod::Historical);

// ========== Drawdown ==========

struct DrawdownResult {
    f64 max_drawdown = 0.0;
    std::size_t peak_idx = 0;
    std::size_t trough_idx = 0;
    std::size_t recovery_idx = 0;
    Vector<f64> drawdown_series;
    f64 avg_drawdown = 0.0;
    f64 max_drawdown_duration = 0.0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

Result<DrawdownResult> drawdown_analysis(const Vector<f64>& returns);

// ========== Performance ratios ==========

struct PerformanceResult {
    f64 sharpe_ratio = 0.0;
    f64 sortino_ratio = 0.0;
    f64 calmar_ratio = 0.0;
    f64 omega_ratio = 0.0;
    f64 information_ratio = 0.0;
    f64 annualized_return = 0.0;
    f64 annualized_volatility = 0.0;
    f64 max_drawdown = 0.0;
    f64 downside_deviation = 0.0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

struct PerformanceOptions {
    f64 risk_free_rate = 0.0;
    f64 target_return = 0.0;
    f64 annualization_factor = 252.0;

    PerformanceOptions& set_risk_free_rate(f64 r) { risk_free_rate = r; return *this; }
    PerformanceOptions& set_target_return(f64 t) { target_return = t; return *this; }
    PerformanceOptions& set_annualization_factor(f64 a) { annualization_factor = a; return *this; }
};

Result<PerformanceResult> performance_metrics(const Vector<f64>& returns,
                                               const PerformanceOptions& opts = PerformanceOptions{});

// ========== Portfolio optimization ==========

struct PortfolioResult {
    Vector<f64> weights;
    f64 expected_return = 0.0;
    f64 volatility = 0.0;
    f64 sharpe_ratio = 0.0;
    std::size_t n_assets = 0;
    f64 elapsed_ms = 0.0;
};

struct PortfolioOptions {
    f64 risk_free_rate = 0.0;
    f64 target_return = -1.0;
    bool long_only = true;

    PortfolioOptions& set_risk_free_rate(f64 r) { risk_free_rate = r; return *this; }
    PortfolioOptions& set_target_return(f64 t) { target_return = t; return *this; }
    PortfolioOptions& set_long_only(bool l) { long_only = l; return *this; }
};

Result<PortfolioResult> minimum_variance_portfolio(const Vector<f64>& expected_returns,
                                                     const Matrix<f64>& cov_matrix,
                                                     const PortfolioOptions& opts = PortfolioOptions{});

Result<PortfolioResult> max_sharpe_portfolio(const Vector<f64>& expected_returns,
                                              const Matrix<f64>& cov_matrix,
                                              const PortfolioOptions& opts = PortfolioOptions{});

struct EfficientFrontierResult {
    std::vector<PortfolioResult> portfolios;
    std::size_t n_points = 0;
    std::size_t n_assets = 0;
    f64 elapsed_ms = 0.0;
};

Result<EfficientFrontierResult> efficient_frontier(const Vector<f64>& expected_returns,
                                                     const Matrix<f64>& cov_matrix,
                                                     std::size_t n_points = 50,
                                                     const PortfolioOptions& opts = PortfolioOptions{});

} // namespace hfm
