#include <gtest/gtest.h>
#include "hfm/hf/realized_measures.hpp"
#include <cmath>

using namespace hfm;

TEST(RealizedMeasuresTest, RealizedVariance) {
    // Known: sum of r_i^2
    Series<f64> r(std::vector<f64>{0.01, -0.02, 0.015, -0.005, 0.03});
    f64 rv = realized_variance(r);
    f64 expected = 0.01*0.01 + 0.02*0.02 + 0.015*0.015 + 0.005*0.005 + 0.03*0.03;
    EXPECT_NEAR(rv, expected, 1e-15);
}

TEST(RealizedMeasuresTest, RealizedVolatility) {
    Series<f64> r(std::vector<f64>{0.01, -0.02, 0.015});
    f64 rv = realized_variance(r);
    f64 rvol = realized_volatility(r);
    EXPECT_NEAR(rvol, std::sqrt(rv), 1e-15);
}

TEST(RealizedMeasuresTest, BipowerVariation) {
    Series<f64> r(std::vector<f64>{0.01, -0.02, 0.015, -0.005});
    f64 bv = bipower_variation(r);
    // BV should be positive and close to RV when no jumps
    EXPECT_GT(bv, 0.0);
}

TEST(RealizedMeasuresTest, Semivariance) {
    Series<f64> r(std::vector<f64>{0.01, -0.02, 0.015, -0.005, 0.03});
    f64 rsv_neg = realized_semivariance(r, false);
    f64 rsv_pos = realized_semivariance(r, true);
    f64 rv = realized_variance(r);
    EXPECT_NEAR(rsv_neg + rsv_pos, rv, 1e-15);
}

TEST(RealizedMeasuresTest, ComputeAll) {
    Series<f64> r(std::vector<f64>{0.01, -0.02, 0.015, -0.005, 0.03});
    auto result = compute_realized_measures(r);
    EXPECT_EQ(result.n_obs, 5u);
    EXPECT_GT(result.realized_variance, 0.0);
    EXPECT_GT(result.realized_volatility, 0.0);
    EXPECT_GT(result.bipower_variation, 0.0);
    EXPECT_GE(result.jump_statistic, 0.0);
}

TEST(RealizedMeasuresTest, EmptyThrows) {
    Series<f64> r;
    EXPECT_THROW(realized_variance(r), std::invalid_argument);
}
