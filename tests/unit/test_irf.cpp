#include <gtest/gtest.h>
#include "hfm/timeseries/irf.hpp"
#include "hfm/timeseries/var.hpp"
#include <cmath>

using namespace hfm;

namespace {
Matrix<f64> generate_var_data(std::size_t n, uint64_t seed) {
    Matrix<f64> Y(n, 2);
    Y(0, 0) = 0.0;
    Y(0, 1) = 0.0;
    for (std::size_t t = 1; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 e1 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 2.0;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 e2 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 2.0;
        Y(t, 0) = 0.5 * Y(t - 1, 0) + 0.1 * Y(t - 1, 1) + e1;
        Y(t, 1) = 0.2 * Y(t - 1, 0) + 0.3 * Y(t - 1, 1) + e2;
    }
    return Y;
}
} // namespace

TEST(IRFTest, BasicIRF) {
    auto Y = generate_var_data(500, 42);
    VAROptions opts;
    opts.p = 1;
    auto var_res = var(Y, opts);
    ASSERT_TRUE(var_res.is_ok());

    auto irf = var_irf(var_res.value(), 10);
    ASSERT_TRUE(irf.is_ok());
    EXPECT_EQ(irf.value().n_horizons, 10u);
    EXPECT_EQ(irf.value().n_vars, 2u);
    EXPECT_EQ(irf.value().irf.size(), 11u);

    // At horizon 0, IRF should be close to Cholesky of Sigma_u
    // (identity shock → impulse = Cholesky factor)
    auto& irf0 = irf.value().irf[0];
    EXPECT_EQ(irf0.rows(), 2u);
    EXPECT_EQ(irf0.cols(), 2u);
}

TEST(IRFTest, IRFDecays) {
    auto Y = generate_var_data(500, 42);
    VAROptions opts;
    opts.p = 1;
    auto var_res = var(Y, opts);
    ASSERT_TRUE(var_res.is_ok());

    auto irf = var_irf(var_res.value(), 20);
    ASSERT_TRUE(irf.is_ok());

    // IRF should decay for stationary VAR
    f64 norm_0 = 0.0, norm_20 = 0.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) {
            norm_0 += irf.value().irf[0](i, j) * irf.value().irf[0](i, j);
            norm_20 += irf.value().irf[20](i, j) * irf.value().irf[20](i, j);
        }
    EXPECT_LT(norm_20, norm_0);
}

TEST(IRFTest, FEVD) {
    auto Y = generate_var_data(500, 42);
    VAROptions opts;
    opts.p = 1;
    auto var_res = var(Y, opts);
    ASSERT_TRUE(var_res.is_ok());

    auto fevd = var_fevd(var_res.value(), 10);
    ASSERT_TRUE(fevd.is_ok());
    EXPECT_EQ(fevd.value().n_horizons, 10u);

    // Each row should sum to 1.0
    for (std::size_t h = 0; h <= 10; ++h) {
        for (std::size_t i = 0; i < 2; ++i) {
            f64 row_sum = 0.0;
            for (std::size_t j = 0; j < 2; ++j) {
                row_sum += fevd.value().fevd[h](i, j);
            }
            EXPECT_NEAR(row_sum, 1.0, 1e-10);
        }
    }
}

TEST(IRFTest, ForecastEval) {
    Vector<f64> actual({1.0, 2.0, 3.0, 4.0, 5.0});
    Vector<f64> forecast({1.1, 1.9, 3.2, 3.8, 5.1});
    auto result = forecast_eval(actual, forecast);
    ASSERT_TRUE(result.is_ok());
    auto& r = result.value();
    EXPECT_GT(r.rmse, 0.0);
    EXPECT_GT(r.mae, 0.0);
    EXPECT_LT(r.rmse, 1.0);
    EXPECT_GT(r.r_squared, 0.9);
}
