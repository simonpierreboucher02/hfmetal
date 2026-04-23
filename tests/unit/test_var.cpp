#include <gtest/gtest.h>
#include "hfm/timeseries/var.hpp"
#include <cmath>

using namespace hfm;

TEST(VARTest, BivariateLagDesignMatrix) {
    // Y is 5 x 2
    Matrix<f64> Y(5, 2);
    for (std::size_t i = 0; i < 5; ++i) {
        Y(i, 0) = static_cast<f64>(i + 1);
        Y(i, 1) = static_cast<f64>((i + 1) * 10);
    }

    auto X = var_lag_design_matrix(Y, 2, true);
    // T = 5 - 2 = 3, k = 2*2 + 1 = 5 (const, y1_lag1, y2_lag1, y1_lag2, y2_lag2)
    EXPECT_EQ(X.rows(), 3u);
    EXPECT_EQ(X.cols(), 5u);
    EXPECT_DOUBLE_EQ(X(0, 0), 1.0); // intercept
    // First row: t=2, lag1 = row 1 (Y[1,:] = {2, 20}), lag2 = row 0 (Y[0,:] = {1, 10})
    EXPECT_DOUBLE_EQ(X(0, 1), 2.0);  // y1(t-1)
    EXPECT_DOUBLE_EQ(X(0, 2), 20.0); // y2(t-1)
    EXPECT_DOUBLE_EQ(X(0, 3), 1.0);  // y1(t-2)
    EXPECT_DOUBLE_EQ(X(0, 4), 10.0); // y2(t-2)
}

TEST(VARTest, BivariateVAR1) {
    // Generate bivariate VAR(1) process
    std::size_t n = 500;
    Matrix<f64> Y(n, 2, 0.0);
    Y(0, 0) = 1.0;
    Y(0, 1) = 0.5;
    uint64_t seed = 99;
    for (std::size_t t = 1; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 e1 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.2;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 e2 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.2;
        Y(t, 0) = 0.3 + 0.5 * Y(t - 1, 0) + 0.1 * Y(t - 1, 1) + e1;
        Y(t, 1) = 0.1 + 0.2 * Y(t - 1, 0) + 0.4 * Y(t - 1, 1) + e2;
    }

    auto result = var(Y);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_EQ(res.n_vars, 2u);
    EXPECT_EQ(res.p, 1u);
    EXPECT_EQ(res.n_obs, n - 1);
    EXPECT_EQ(res.coefficients.rows(), 2u);
    EXPECT_EQ(res.coefficients.cols(), 3u); // const + 2 vars * 1 lag
    EXPECT_EQ(res.residuals.rows(), n - 1);
    EXPECT_EQ(res.residuals.cols(), 2u);
    EXPECT_EQ(res.sigma_u.rows(), 2u);
    EXPECT_EQ(res.sigma_u.cols(), 2u);
}

TEST(VARTest, InformationCriteria) {
    std::size_t n = 200;
    Matrix<f64> Y(n, 2, 0.0);
    uint64_t seed = 42;
    for (std::size_t t = 1; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 e1 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 e2 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        Y(t, 0) = 0.3 * Y(t - 1, 0) + e1;
        Y(t, 1) = 0.2 * Y(t - 1, 1) + e2;
    }

    auto r1 = var(Y, VAROptions{}.set_p(1));
    auto r2 = var(Y, VAROptions{}.set_p(4));
    ASSERT_TRUE(r1.is_ok());
    ASSERT_TRUE(r2.is_ok());

    // BIC should penalize overparameterized model
    EXPECT_LT(r1.value().bic, r2.value().bic);
}

TEST(VARTest, TooFewObservations) {
    Matrix<f64> Y(3, 2, 1.0);
    auto result = var(Y, VAROptions{}.set_p(5));
    EXPECT_FALSE(result.is_ok());
}
