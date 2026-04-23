#include <gtest/gtest.h>
#include "hfm/estimators/batched_ols.hpp"
#include <cmath>

using namespace hfm;

TEST(BatchedOLSTest, SameXMultipleY) {
    std::size_t n = 200;
    std::size_t k = 2;
    std::size_t n_reg = 10;

    Matrix<f64> X(n, k, 0.0);
    Matrix<f64> Y(n, n_reg, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i) / 100.0;
        for (std::size_t r = 0; r < n_reg; ++r) {
            Y(i, r) = static_cast<f64>(r + 1) + 0.5 * X(i, 1);
        }
    }

    auto result = batched_ols(Y, X);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_EQ(res.n_regressions, n_reg);
    EXPECT_EQ(res.n_succeeded, n_reg);

    for (std::size_t r = 0; r < n_reg; ++r) {
        EXPECT_NEAR(res.betas(r, 0), static_cast<f64>(r + 1), 1e-8);
        EXPECT_NEAR(res.betas(r, 1), 0.5, 1e-8);
    }
}

TEST(BatchedOLSTest, WithWhiteSE) {
    std::size_t n = 100;
    std::size_t k = 2;
    std::size_t n_reg = 5;

    Matrix<f64> X(n, k, 0.0);
    Matrix<f64> Y(n, n_reg, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i) / 50.0;
        for (std::size_t r = 0; r < n_reg; ++r) {
            Y(i, r) = 1.0 + 2.0 * X(i, 1) + std::sin(static_cast<f64>(i * r)) * 0.1;
        }
    }

    BatchedOLSOptions opts;
    opts.covariance = CovarianceType::White;
    auto result = batched_ols(Y, X, opts);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(result.value().n_succeeded, n_reg);
}
