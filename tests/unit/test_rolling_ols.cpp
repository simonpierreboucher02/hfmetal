#include <gtest/gtest.h>
#include "hfm/estimators/rolling_ols.hpp"
#include <cmath>

using namespace hfm;

TEST(RollingOLSTest, BasicRolling) {
    std::size_t n = 500;
    std::size_t k = 2;
    Matrix<f64> X(n, k, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        f64 x = static_cast<f64>(i) / 100.0;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        y[i] = 1.0 + 2.0 * x;
    }

    RollingOptions opts;
    opts.window = 100;
    opts.step = 50;
    auto result = rolling_ols(y, X, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_GT(res.n_windows(), 0u);
    // Perfect fit: all betas should be [1, 2]
    for (std::size_t w = 0; w < res.n_windows(); ++w) {
        EXPECT_NEAR(res.betas()(w, 0), 1.0, 1e-8);
        EXPECT_NEAR(res.betas()(w, 1), 2.0, 1e-8);
        EXPECT_NEAR(res.r_squared()[w], 1.0, 1e-8);
    }
}

TEST(RollingOLSTest, StepOne) {
    std::size_t n = 200;
    Matrix<f64> X(n, 2, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i);
        y[i] = 3.0 + 0.5 * static_cast<f64>(i);
    }

    RollingOptions opts;
    opts.window = 50;
    opts.step = 1;
    auto result = rolling_ols(y, X, opts);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(result.value().n_windows(), 151u);
}

TEST(ExpandingOLSTest, Basic) {
    std::size_t n = 300;
    Matrix<f64> X(n, 2, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i) / 100.0;
        y[i] = 2.0 + 1.0 * X(i, 1);
    }

    RollingOptions opts;
    opts.window = 50;
    opts.step = 10;
    auto result = expanding_ols(y, X, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_GT(res.n_windows(), 0u);
    // All windows start at 0
    for (std::size_t w = 0; w < res.n_windows(); ++w) {
        EXPECT_EQ(res.window_starts()[w], 0u);
    }
}

TEST(RollingOLSTest, TooSmallWindow) {
    Matrix<f64> X(10, 5, 0.0);
    Vector<f64> y(10);
    RollingOptions opts;
    opts.window = 3;
    auto result = rolling_ols(y, X, opts);
    EXPECT_FALSE(result.is_ok());
}
