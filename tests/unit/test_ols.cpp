#include <gtest/gtest.h>
#include "hfm/estimators/ols.hpp"
#include <cmath>

using namespace hfm;

TEST(OLSTest, SimpleRegression) {
    // y = 1 + 2*x, perfect fit
    std::size_t n = 100;
    Matrix<f64> X(n, 2, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        f64 x = static_cast<f64>(i) / 10.0;
        X(i, 0) = 1.0;  // intercept
        X(i, 1) = x;
        y[i] = 1.0 + 2.0 * x;
    }

    auto result = ols(y, X);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_NEAR(res.coefficients()[0], 1.0, 1e-10);
    EXPECT_NEAR(res.coefficients()[1], 2.0, 1e-10);
    EXPECT_NEAR(res.r_squared(), 1.0, 1e-10);
}

TEST(OLSTest, WithNoise) {
    std::size_t n = 1000;
    Matrix<f64> X(n, 2, 0.0);
    Vector<f64> y(n);

    // Deterministic pseudo-noise
    for (std::size_t i = 0; i < n; ++i) {
        f64 x = static_cast<f64>(i) / 100.0;
        f64 noise = std::sin(static_cast<f64>(i) * 0.1) * 0.5;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        y[i] = 3.0 + 1.5 * x + noise;
    }

    auto result = ols(y, X);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_NEAR(res.coefficients()[0], 3.0, 0.5);
    EXPECT_NEAR(res.coefficients()[1], 1.5, 0.1);
    EXPECT_GT(res.r_squared(), 0.9);
}

TEST(OLSTest, WhiteCovariance) {
    std::size_t n = 100;
    Matrix<f64> X(n, 2, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        f64 x = static_cast<f64>(i) / 10.0;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        y[i] = 1.0 + 2.0 * x + std::sin(static_cast<f64>(i)) * 0.1;
    }

    OLSOptions opts;
    opts.covariance = CovarianceType::White;
    auto result = ols(y, X, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    for (std::size_t j = 0; j < res.n_regressors(); ++j) {
        EXPECT_GT(res.std_errors()[j], 0.0);
    }
}

TEST(OLSTest, NeweyWest) {
    std::size_t n = 200;
    Matrix<f64> X(n, 2, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        f64 x = static_cast<f64>(i) / 20.0;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        y[i] = 2.0 + 0.5 * x + std::sin(static_cast<f64>(i) * 0.3) * 0.2;
    }

    OLSOptions opts;
    opts.covariance = CovarianceType::NeweyWest;
    auto result = ols(y, X, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_GT(res.std_errors()[0], 0.0);
    EXPECT_GT(res.std_errors()[1], 0.0);
}

TEST(OLSTest, Summary) {
    Matrix<f64> X(3, 2, 0.0);
    X(0, 0) = 1; X(0, 1) = 1;
    X(1, 0) = 1; X(1, 1) = 2;
    X(2, 0) = 1; X(2, 1) = 3;
    Vector<f64> y{3.0, 5.0, 7.0};

    auto result = ols(y, X);
    ASSERT_TRUE(result.is_ok());
    std::string summary = result.value().summary();
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("OLS"), std::string::npos);
}

TEST(OLSTest, DimensionMismatch) {
    Matrix<f64> X(3, 2, 0.0);
    Vector<f64> y{1.0, 2.0};
    auto result = ols(y, X);
    EXPECT_FALSE(result.is_ok());
}
