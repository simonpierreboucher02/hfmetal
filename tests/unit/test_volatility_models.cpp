#include <gtest/gtest.h>
#include "hfm/models/egarch.hpp"
#include "hfm/models/gjr_garch.hpp"
#include "hfm/models/garch_t.hpp"
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
} // namespace

// ========== EGARCH ==========

TEST(VolatilityTest, EGARCHBasic) {
    auto r = generate_garch_data(1000, 0.00001, 0.08, 0.90, 42);
    auto result = egarch(r);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_EQ(res.conditional_var.size(), 1000u);
    EXPECT_EQ(res.std_residuals.size(), 1000u);
    for (std::size_t i = 0; i < res.conditional_var.size(); ++i) {
        EXPECT_GT(res.conditional_var[i], 0.0);
    }
}

TEST(VolatilityTest, EGARCHLogLikelihood) {
    auto r = generate_garch_data(500, 0.00001, 0.05, 0.93, 99);
    f64 ll = egarch_loglikelihood(r, 0.0, -0.1, 0.1, -0.05, 0.98);
    EXPECT_GT(ll, -1e20);
}

// ========== GJR-GARCH ==========

TEST(VolatilityTest, GJRGARCHBasic) {
    auto r = generate_garch_data(1000, 0.00001, 0.06, 0.90, 42);
    auto result = gjr_garch(r);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_GT(res.omega, 0.0);
    EXPECT_GE(res.alpha, 0.0);
    EXPECT_GE(res.gamma, 0.0);
    EXPECT_GE(res.beta, 0.0);
    EXPECT_LT(res.persistence, 1.0);
    EXPECT_EQ(res.conditional_var.size(), 1000u);
}

TEST(VolatilityTest, GJRGARCHLogLikelihood) {
    auto r = generate_garch_data(500, 0.00001, 0.05, 0.93, 99);
    f64 ll = gjr_garch_loglikelihood(r, 0.0, 0.00001, 0.05, 0.04, 0.88);
    EXPECT_GT(ll, -1e20);
}

// ========== GARCH-t ==========

TEST(VolatilityTest, GARCHTBasic) {
    auto r = generate_garch_data(1000, 0.00001, 0.08, 0.90, 42);
    auto result = garch_t(r);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_GT(res.omega, 0.0);
    EXPECT_GT(res.alpha, 0.0);
    EXPECT_GT(res.beta, 0.0);
    EXPECT_GT(res.nu, 2.0);
    EXPECT_LT(res.persistence, 1.0);
    EXPECT_EQ(res.conditional_var.size(), 1000u);
}

TEST(VolatilityTest, GARCHTInformationCriteria) {
    auto r = generate_garch_data(500, 0.00001, 0.05, 0.93, 99);
    auto result = garch_t(r);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_LT(res.aic, 0.0);
    EXPECT_LT(res.bic, 0.0);
}

// ========== Error cases ==========

TEST(VolatilityTest, TooFewObservationsEGARCH) {
    Vector<f64> r(5, 0.01);
    EXPECT_FALSE(egarch(r).is_ok());
}

TEST(VolatilityTest, TooFewObservationsGJR) {
    Vector<f64> r(5, 0.01);
    EXPECT_FALSE(gjr_garch(r).is_ok());
}

TEST(VolatilityTest, TooFewObservationsGARCHT) {
    Vector<f64> r(5, 0.01);
    EXPECT_FALSE(garch_t(r).is_ok());
}
