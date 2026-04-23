#include <gtest/gtest.h>
#include "hfm/timeseries/local_projections.hpp"
#include <cmath>

using namespace hfm;

TEST(LocalProjectionsTest, UnivariateLP) {
    std::size_t n = 500;
    Vector<f64> y(n);
    uint64_t seed = 42;
    y[0] = 0.0;
    for (std::size_t t = 1; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 noise = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.5;
        y[t] = 0.5 * y[t-1] + noise;
    }

    auto result = local_projections(y, LPOptions{}.set_max_horizon(8).set_n_lags(2));
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_EQ(res.max_horizon, 8u);
    EXPECT_EQ(res.irf.size(), 9u);         // 0..8
    EXPECT_EQ(res.irf_se.size(), 9u);
    EXPECT_EQ(res.cumulative_irf.size(), 9u);

    // Impact response should be close to 1 (own shock)
    // IRF should decay with horizon
    EXPECT_GT(std::abs(res.irf[0]), 0.01);
}

TEST(LocalProjectionsTest, BivariateLP) {
    std::size_t n = 500;
    Vector<f64> y(n), x(n);
    Matrix<f64> controls(n, 0); // no controls
    uint64_t seed = 99;

    for (std::size_t t = 0; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        x[t] = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 1.0;
    }
    y[0] = 0.0;
    for (std::size_t t = 1; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 noise = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        y[t] = 0.3 * y[t-1] + 0.5 * x[t] + noise;
    }

    auto result = local_projections(y, x, controls, LPOptions{}.set_max_horizon(6));
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    // Impact of x on y at horizon 0 should be ~0.5
    EXPECT_NEAR(res.irf[0], 0.5, 0.3);
    EXPECT_GT(res.n_obs, 0u);
}

TEST(LocalProjectionsTest, ConfidenceBands) {
    std::size_t n = 300;
    Vector<f64> y(n);
    uint64_t seed = 77;
    for (std::size_t t = 0; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        y[t] = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0);
    }

    auto result = local_projections(y, LPOptions{}.set_max_horizon(4));
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    for (std::size_t h = 0; h <= 4; ++h) {
        EXPECT_LE(res.irf_lower[h], res.irf[h]);
        EXPECT_GE(res.irf_upper[h], res.irf[h]);
        EXPECT_GE(res.irf_se[h], 0.0);
    }
}

TEST(LocalProjectionsTest, TooFewObs) {
    Vector<f64> y(5);
    auto result = local_projections(y, LPOptions{}.set_max_horizon(10));
    EXPECT_FALSE(result.is_ok());
}
