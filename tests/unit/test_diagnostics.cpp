#include <gtest/gtest.h>
#include "hfm/diagnostics/statistical_tests.hpp"
#include <cmath>

using namespace hfm;

namespace {
Vector<f64> generate_normal(std::size_t n, f64 mu, f64 sigma, uint64_t seed) {
    Vector<f64> x(n);
    for (std::size_t i = 0; i < n; i += 2) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u1 = static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u2 = static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31);
        u1 = std::max(u1, 1e-10);
        f64 z1 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        f64 z2 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
        x[i] = mu + sigma * z1;
        if (i + 1 < n) x[i + 1] = mu + sigma * z2;
    }
    return x;
}

Vector<f64> generate_ar1(std::size_t n, f64 phi, uint64_t seed) {
    auto noise = generate_normal(n, 0.0, 1.0, seed);
    Vector<f64> y(n);
    y[0] = noise[0];
    for (std::size_t i = 1; i < n; ++i) {
        y[i] = phi * y[i - 1] + noise[i];
    }
    return y;
}
} // namespace

// ========== Descriptive Stats ==========

TEST(DiagnosticsTest, DescriptiveStats) {
    auto x = generate_normal(1000, 5.0, 2.0, 42);
    auto result = descriptive_stats(x);
    ASSERT_TRUE(result.is_ok());
    auto& ds = result.value();
    EXPECT_NEAR(ds.mean, 5.0, 0.3);
    EXPECT_NEAR(ds.std_dev, 2.0, 0.3);
    EXPECT_NEAR(ds.skewness, 0.0, 0.5);
    EXPECT_NEAR(ds.excess_kurtosis, 0.0, 0.5);
    EXPECT_EQ(ds.n_obs, 1000u);
    EXPECT_LT(ds.min, ds.q25);
    EXPECT_LT(ds.q25, ds.median);
    EXPECT_LT(ds.median, ds.q75);
    EXPECT_LT(ds.q75, ds.max);
}

// ========== Jarque-Bera ==========

TEST(DiagnosticsTest, JarqueBeraOnNormal) {
    auto x = generate_normal(2000, 0.0, 1.0, 42);
    auto result = jarque_bera(x);
    ASSERT_TRUE(result.is_ok());
    auto& jb = result.value();
    EXPECT_GT(jb.p_value, 0.01);
    EXPECT_NEAR(jb.skewness, 0.0, 0.3);
    EXPECT_NEAR(jb.excess_kurtosis, 0.0, 0.5);
}

TEST(DiagnosticsTest, JarqueBeraOnSkewed) {
    Vector<f64> x(500);
    uint64_t seed = 77;
    for (std::size_t i = 0; i < 500; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u = static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31);
        x[i] = std::exp(u * 3.0);
    }
    auto result = jarque_bera(x);
    ASSERT_TRUE(result.is_ok());
    EXPECT_LT(result.value().p_value, 0.05);
}

// ========== Durbin-Watson ==========

TEST(DiagnosticsTest, DurbinWatsonNoAutocorr) {
    auto x = generate_normal(500, 0.0, 1.0, 42);
    auto result = durbin_watson(x);
    ASSERT_TRUE(result.is_ok());
    EXPECT_NEAR(result.value().statistic, 2.0, 0.3);
}

TEST(DiagnosticsTest, DurbinWatsonPositiveAutocorr) {
    auto y = generate_ar1(500, 0.8, 42);
    Vector<f64> residuals(499);
    for (std::size_t i = 1; i < 500; ++i) {
        residuals[i - 1] = y[i] - 0.8 * y[i - 1];
    }
    // Use the autocorrelated y directly
    auto result = durbin_watson(y);
    ASSERT_TRUE(result.is_ok());
    EXPECT_LT(result.value().statistic, 1.5);
}

// ========== Ljung-Box ==========

TEST(DiagnosticsTest, LjungBoxOnWhiteNoise) {
    auto x = generate_normal(500, 0.0, 1.0, 42);
    auto result = ljung_box(x, 10);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().p_value, 0.01);
}

TEST(DiagnosticsTest, LjungBoxOnAR1) {
    auto x = generate_ar1(500, 0.7, 42);
    auto result = ljung_box(x, 10);
    ASSERT_TRUE(result.is_ok());
    EXPECT_LT(result.value().p_value, 0.05);
}

// ========== Autocorrelation ==========

TEST(DiagnosticsTest, ACF) {
    auto x = generate_ar1(1000, 0.5, 42);
    auto acf = autocorrelation(x, 5);
    EXPECT_EQ(acf.size(), 5u);
    EXPECT_GT(acf[0], 0.3);
    EXPECT_GT(acf[0], acf[1]);
}

// ========== Breusch-Pagan ==========

TEST(DiagnosticsTest, BreuschPaganHomoskedastic) {
    auto x = generate_normal(200, 0.0, 1.0, 42);
    auto e = generate_normal(200, 0.0, 1.0, 99);
    Matrix<f64> X(200, 1);
    for (std::size_t i = 0; i < 200; ++i) X(i, 0) = x[i];
    auto result = breusch_pagan(e, X);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().p_value, 0.01);
}

// ========== ARCH-LM ==========

TEST(DiagnosticsTest, ArchLMNoEffects) {
    auto x = generate_normal(300, 0.0, 1.0, 42);
    auto result = arch_lm(x, 5);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().p_value, 0.01);
}

// ========== ADF ==========

TEST(DiagnosticsTest, ADFOnStationary) {
    auto y = generate_ar1(500, 0.5, 42);
    auto result = adf_test(y);
    ASSERT_TRUE(result.is_ok());
    EXPECT_LT(result.value().statistic, result.value().critical_5pct);
}

TEST(DiagnosticsTest, ADFOnRandomWalk) {
    Vector<f64> y(500);
    y[0] = 0.0;
    uint64_t seed = 42;
    for (std::size_t i = 1; i < 500; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 2.0;
        y[i] = y[i - 1] + z;
    }
    auto result = adf_test(y);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().statistic, result.value().critical_5pct);
}

// ========== KPSS ==========

TEST(DiagnosticsTest, KPSSOnStationary) {
    auto y = generate_ar1(500, 0.3, 42);
    auto result = kpss_test(y);
    ASSERT_TRUE(result.is_ok());
    EXPECT_LT(result.value().statistic, result.value().critical_5pct);
}

// ========== White Test ==========

TEST(DiagnosticsTest, WhiteTestHomoskedastic) {
    auto x = generate_normal(200, 0.0, 1.0, 42);
    auto e = generate_normal(200, 0.0, 1.0, 99);
    Matrix<f64> X(200, 1);
    for (std::size_t i = 0; i < 200; ++i) X(i, 0) = x[i];
    auto result = white_test(e, X);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().p_value, 0.01);
}

// ========== Distribution utilities ==========

TEST(DiagnosticsTest, Chi2CDF) {
    EXPECT_NEAR(chi2_cdf(3.841, 1.0), 0.95, 0.01);
    EXPECT_NEAR(chi2_cdf(5.991, 2.0), 0.95, 0.01);
    EXPECT_NEAR(chi2_sf(3.841, 1.0), 0.05, 0.01);
}

TEST(DiagnosticsTest, NormalCDF) {
    EXPECT_NEAR(normal_cdf(0.0), 0.5, 1e-10);
    EXPECT_NEAR(normal_cdf(1.96), 0.975, 0.001);
    EXPECT_NEAR(normal_sf(1.96), 0.025, 0.001);
}

// ========== Error cases ==========

TEST(DiagnosticsTest, TooFewObservations) {
    Vector<f64> x(3, 1.0);
    EXPECT_FALSE(descriptive_stats(x).is_ok());
    EXPECT_FALSE(jarque_bera(x).is_ok());
}
