#include <gtest/gtest.h>
#include "hfm/estimators/iv.hpp"
#include <cmath>

using namespace hfm;

TEST(IVTest, JustIdentified) {
    // y = 2 + 3*x + e, x is endogenous, z is instrument
    std::size_t n = 500;
    Vector<f64> y(n), z_raw(n);
    Matrix<f64> X(n, 2), Z(n, 2);

    uint64_t seed = 42;
    for (std::size_t i = 0; i < n; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        f64 x = 0.5 * z + u; // x correlated with u (endogenous)
        y[i] = 2.0 + 3.0 * x + u;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        Z(i, 0) = 1.0;
        Z(i, 1) = z;
    }

    auto result = iv_2sls(y, X, Z);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_EQ(res.n_obs(), n);
    EXPECT_EQ(res.n_regressors(), 2u);
    EXPECT_EQ(res.n_instruments(), 2u);
    EXPECT_NEAR(res.coefficients()[0], 2.0, 1.0);
    EXPECT_NEAR(res.coefficients()[1], 3.0, 1.0);
    EXPECT_FALSE(res.overidentified());
}

TEST(IVTest, OverIdentified) {
    std::size_t n = 500;
    Vector<f64> y(n);
    Matrix<f64> X(n, 2), Z(n, 3);

    uint64_t seed = 99;
    for (std::size_t i = 0; i < n; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z1 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z2 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        f64 x = 0.4 * z1 + 0.3 * z2 + u;
        y[i] = 1.0 + 2.0 * x + u;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        Z(i, 0) = 1.0;
        Z(i, 1) = z1;
        Z(i, 2) = z2;
    }

    auto result = iv_2sls(y, X, Z);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_TRUE(res.overidentified());
    EXPECT_GT(res.sargan_stat(), 0.0);
    EXPECT_EQ(res.n_instruments(), 3u);
}

TEST(IVTest, WeakInstrumentWarning) {
    std::size_t n = 200;
    Vector<f64> y(n);
    Matrix<f64> X(n, 2), Z(n, 2);

    uint64_t seed = 77;
    for (std::size_t i = 0; i < n; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0);
        f64 x = 0.01 * z + u; // very weak instrument
        y[i] = 1.0 + x + u;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        Z(i, 0) = 1.0;
        Z(i, 1) = z;
    }

    auto result = iv_2sls(y, X, Z);
    ASSERT_TRUE(result.is_ok());
    EXPECT_TRUE(result.value().first_stage().weak_instrument);
}

TEST(IVTest, UnderIdentified) {
    Vector<f64> y(50, 1.0);
    Matrix<f64> X(50, 3, 1.0); // 3 regressors
    Matrix<f64> Z(50, 2, 1.0); // only 2 instruments
    auto result = iv_2sls(y, X, Z);
    EXPECT_FALSE(result.is_ok());
}

TEST(IVTest, RobustSE) {
    std::size_t n = 300;
    Vector<f64> y(n);
    Matrix<f64> X(n, 2), Z(n, 2);

    uint64_t seed = 55;
    for (std::size_t i = 0; i < n; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        f64 x = 0.6 * z + u;
        y[i] = 1.0 + 2.0 * x + u;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        Z(i, 0) = 1.0;
        Z(i, 1) = z;
    }

    auto res_classical = iv_2sls(y, X, Z);
    auto res_robust = iv_2sls(y, X, Z, IVOptions{}.set_covariance(CovarianceType::White));
    ASSERT_TRUE(res_classical.is_ok());
    ASSERT_TRUE(res_robust.is_ok());

    // Both should give same coefficients
    EXPECT_NEAR(res_classical.value().coefficients()[0],
                res_robust.value().coefficients()[0], 1e-10);
}
