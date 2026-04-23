#include <gtest/gtest.h>
#include "hfm/risk/measures.hpp"
#include <cmath>

using namespace hfm;

namespace {
Vector<f64> generate_returns(std::size_t n, f64 mu, f64 sigma, uint64_t seed) {
    Vector<f64> r(n);
    for (std::size_t i = 0; i < n; i += 2) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u1 = static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u2 = static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31);
        u1 = std::max(u1, 1e-10);
        f64 z1 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        f64 z2 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
        r[i] = mu + sigma * z1;
        if (i + 1 < n) r[i + 1] = mu + sigma * z2;
    }
    return r;
}
} // namespace

// ========== VaR ==========

TEST(RiskTest, HistoricalVaR) {
    auto r = generate_returns(1000, 0.0005, 0.02, 42);
    auto result = value_at_risk(r, 0.95, VaRMethod::Historical);
    ASSERT_TRUE(result.is_ok());
    auto& v = result.value();
    EXPECT_GT(v.var, 0.0);
    EXPECT_GT(v.cvar, v.var);
    EXPECT_EQ(v.n_obs, 1000u);
}

TEST(RiskTest, ParametricVaR) {
    auto r = generate_returns(1000, 0.0005, 0.02, 42);
    auto result = value_at_risk(r, 0.95, VaRMethod::Parametric);
    ASSERT_TRUE(result.is_ok());
    auto& v = result.value();
    EXPECT_GT(v.var, 0.0);
    EXPECT_GT(v.cvar, v.var);
}

TEST(RiskTest, CornishFisherVaR) {
    auto r = generate_returns(1000, 0.0005, 0.02, 42);
    auto result = value_at_risk(r, 0.95, VaRMethod::CornishFisher);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().var, 0.0);
}

// ========== Drawdown ==========

TEST(RiskTest, DrawdownAnalysis) {
    auto r = generate_returns(500, 0.0001, 0.01, 42);
    auto result = drawdown_analysis(r);
    ASSERT_TRUE(result.is_ok());
    auto& dd = result.value();
    EXPECT_GT(dd.max_drawdown, 0.0);
    EXPECT_EQ(dd.drawdown_series.size(), 500u);
    EXPECT_GE(dd.avg_drawdown, 0.0);
    EXPECT_LE(dd.peak_idx, dd.trough_idx);
}

// ========== Performance metrics ==========

TEST(RiskTest, PerformanceMetrics) {
    auto r = generate_returns(500, 0.0003, 0.01, 42);
    PerformanceOptions opts;
    opts.risk_free_rate = 0.02;
    auto result = performance_metrics(r, opts);
    ASSERT_TRUE(result.is_ok());
    auto& pm = result.value();
    EXPECT_NE(pm.sharpe_ratio, 0.0);
    EXPECT_NE(pm.sortino_ratio, 0.0);
    EXPECT_GT(pm.annualized_volatility, 0.0);
    EXPECT_GT(pm.max_drawdown, 0.0);
    EXPECT_GT(pm.omega_ratio, 0.0);
}

// ========== Portfolio ==========

TEST(RiskTest, MinVariancePortfolio) {
    Vector<f64> mu({0.10, 0.15, 0.12});
    Matrix<f64> cov(3, 3, std::vector<f64>{
        0.04, 0.006, 0.002,
        0.006, 0.09, 0.009,
        0.002, 0.009, 0.0225
    });
    auto result = minimum_variance_portfolio(mu, cov);
    ASSERT_TRUE(result.is_ok());
    auto& p = result.value();
    EXPECT_EQ(p.n_assets, 3u);
    f64 wsum = 0.0;
    for (std::size_t i = 0; i < p.weights.size(); ++i) wsum += p.weights[i];
    EXPECT_NEAR(wsum, 1.0, 1e-6);
    EXPECT_GT(p.volatility, 0.0);
}

TEST(RiskTest, MaxSharpePortfolio) {
    Vector<f64> mu({0.10, 0.15, 0.12});
    Matrix<f64> cov(3, 3, std::vector<f64>{
        0.04, 0.006, 0.002,
        0.006, 0.09, 0.009,
        0.002, 0.009, 0.0225
    });
    PortfolioOptions opts;
    opts.risk_free_rate = 0.03;
    auto result = max_sharpe_portfolio(mu, cov, opts);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().sharpe_ratio, 0.0);
}

TEST(RiskTest, EfficientFrontier) {
    Vector<f64> mu({0.10, 0.15, 0.12});
    Matrix<f64> cov(3, 3, std::vector<f64>{
        0.04, 0.006, 0.002,
        0.006, 0.09, 0.009,
        0.002, 0.009, 0.0225
    });
    auto result = efficient_frontier(mu, cov, 20);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(result.value().portfolios.size(), 20u);
}

// ========== Error cases ==========

TEST(RiskTest, TooFewObservations) {
    Vector<f64> r(5, 0.01);
    EXPECT_FALSE(value_at_risk(r).is_ok());
    EXPECT_FALSE(performance_metrics(r).is_ok());
}
