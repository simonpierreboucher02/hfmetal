#include <gtest/gtest.h>
#include "hfm/models/garch.hpp"
#include <cmath>

using namespace hfm;

namespace {
Vector<f64> generate_garch_data(std::size_t n, f64 omega, f64 alpha, f64 beta, uint64_t seed) {
    Vector<f64> r(n);
    f64 h = omega / (1.0 - alpha - beta);
    for (std::size_t t = 0; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 1.7;
        f64 sigma = std::sqrt(h);
        r[t] = sigma * z;
        h = omega + alpha * r[t] * r[t] + beta * h;
    }
    return r;
}
}

TEST(GARCHTest, BasicEstimation) {
    auto r = generate_garch_data(2000, 0.00001, 0.08, 0.90, 42);
    auto result = garch(r);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_GT(res.omega, 0.0);
    EXPECT_GT(res.alpha, 0.0);
    EXPECT_GT(res.beta, 0.0);
    EXPECT_LT(res.persistence, 1.0);
    EXPECT_EQ(res.conditional_var.size(), 2000u);
    EXPECT_EQ(res.std_residuals.size(), 2000u);
    EXPECT_TRUE(res.converged);
}

TEST(GARCHTest, ConditionalVarianceSeries) {
    auto r = generate_garch_data(500, 0.00002, 0.1, 0.85, 77);
    auto result = garch(r);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    // All conditional variances should be positive
    for (std::size_t i = 0; i < res.conditional_var.size(); ++i) {
        EXPECT_GT(res.conditional_var[i], 0.0);
    }

    // Standardized residuals should have roughly unit variance
    f64 var_z = 0.0;
    for (std::size_t i = 0; i < res.std_residuals.size(); ++i) {
        var_z += res.std_residuals[i] * res.std_residuals[i];
    }
    var_z /= static_cast<f64>(res.std_residuals.size());
    EXPECT_NEAR(var_z, 1.0, 0.5);
}

TEST(GARCHTest, InformationCriteria) {
    auto r = generate_garch_data(1000, 0.00001, 0.05, 0.93, 99);
    auto result = garch(r);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_LT(res.aic, 0.0); // Typically negative for returns data
    EXPECT_LT(res.bic, 0.0);
    EXPECT_LT(res.aic, res.bic); // AIC < BIC when n is large
}

TEST(GARCHTest, TooFewObservations) {
    Vector<f64> r(5, 0.01);
    auto result = garch(r);
    EXPECT_FALSE(result.is_ok());
}

TEST(GARCHTest, LogLikelihood) {
    Vector<f64> r(100);
    uint64_t seed = 42;
    for (std::size_t i = 0; i < 100; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        r[i] = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.02;
    }

    f64 ll = garch_loglikelihood(r, 0.0, 0.00001, 0.05, 0.90);
    EXPECT_GT(ll, -1e20); // Should be finite
    // Log-likelihood can be positive for small-variance data (high density)
}
