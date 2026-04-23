#include <gtest/gtest.h>
#include "hfm/estimators/gls.hpp"
#include "hfm/estimators/ols.hpp"

using namespace hfm;

TEST(GLSTest, HomoskedasticRecoverOLS) {
    // When Omega = sigma^2 * I, GLS should give same results as OLS
    std::size_t n = 100;
    Vector<f64> y(n);
    Matrix<f64> X(n, 2);
    uint64_t seed = 77;
    for (std::size_t i = 0; i < n; ++i) {
        f64 x1 = static_cast<f64>(i) / static_cast<f64>(n);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 noise = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.1;
        X(i, 0) = 1.0;
        X(i, 1) = x1;
        y[i] = 2.0 + 3.0 * x1 + noise;
    }

    Matrix<f64> Omega = Matrix<f64>::identity(n);
    auto gls_result = gls(y, X, Omega);
    ASSERT_TRUE(gls_result.is_ok());

    auto ols_result = ols(y, X);
    ASSERT_TRUE(ols_result.is_ok());

    // Coefficients should match closely
    EXPECT_NEAR(gls_result.value().coefficients()[0], ols_result.value().coefficients()[0], 1e-10);
    EXPECT_NEAR(gls_result.value().coefficients()[1], ols_result.value().coefficients()[1], 1e-10);
}

TEST(GLSTest, DimensionMismatch) {
    Vector<f64> y(10, 1.0);
    Matrix<f64> X(10, 2, 1.0);
    Matrix<f64> Omega(5, 5, 1.0); // wrong size
    auto result = gls(y, X, Omega);
    EXPECT_FALSE(result.is_ok());
}

TEST(GLSTest, FeasibleGLS) {
    std::size_t n = 200;
    Vector<f64> y(n);
    Matrix<f64> X(n, 2);
    uint64_t seed = 42;
    for (std::size_t i = 0; i < n; ++i) {
        f64 x1 = static_cast<f64>(i) / static_cast<f64>(n);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 noise = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * (0.1 + 0.5 * x1);
        X(i, 0) = 1.0;
        X(i, 1) = x1;
        y[i] = 1.0 + 2.0 * x1 + noise;
    }

    auto result = fgls(y, X);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_EQ(res.n_obs(), n);
    EXPECT_EQ(res.n_regressors(), 2u);
    EXPECT_NEAR(res.coefficients()[0], 1.0, 0.5);
    EXPECT_NEAR(res.coefficients()[1], 2.0, 1.0);
}
