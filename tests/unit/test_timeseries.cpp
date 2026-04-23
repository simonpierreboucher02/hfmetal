#include <gtest/gtest.h>
#include "hfm/timeseries/ar.hpp"
#include "hfm/timeseries/har.hpp"
#include "hfm/timeseries/rolling.hpp"
#include <cmath>

using namespace hfm;

TEST(ARTest, AR1) {
    // Generate AR(1): y_t = 0.5 + 0.7 * y_{t-1} + noise
    // Use a simple LCG for reproducible pseudo-random noise
    std::size_t n = 1000;
    Vector<f64> y(n);
    y[0] = 1.0;
    uint64_t seed = 42;
    for (std::size_t t = 1; t < n; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 noise = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        y[t] = 0.5 + 0.7 * y[t - 1] + noise;
    }

    AROptions opts;
    opts.p = 1;
    auto result = ar(y, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_EQ(res.p, 1u);
    EXPECT_NEAR(res.coefficients[1], 0.7, 0.15);
    EXPECT_GT(res.r_squared, 0.4);
}

TEST(ARTest, AR2) {
    std::size_t n = 1000;
    Vector<f64> y(n);
    y[0] = 1.0; y[1] = 1.5;
    for (std::size_t t = 2; t < n; ++t) {
        y[t] = 0.3 + 0.5 * y[t - 1] + 0.2 * y[t - 2] +
               std::sin(static_cast<f64>(t) * 0.5) * 0.05;
    }

    AROptions opts;
    opts.p = 2;
    auto result = ar(y, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_EQ(res.p, 2u);
    EXPECT_EQ(res.n_obs, n - 2);
}

TEST(ARTest, LagDesignMatrix) {
    Vector<f64> y{1.0, 2.0, 3.0, 4.0, 5.0};
    auto X = lag_design_matrix(y, 2, true);
    EXPECT_EQ(X.rows(), 3u);  // T = 5-2 = 3
    EXPECT_EQ(X.cols(), 3u);  // const + 2 lags
    EXPECT_DOUBLE_EQ(X(0, 0), 1.0);  // intercept
    EXPECT_DOUBLE_EQ(X(0, 1), 2.0);  // y(t-1) when y(t)=y[2]=3
    EXPECT_DOUBLE_EQ(X(0, 2), 1.0);  // y(t-2) when y(t)=y[2]=3
}

TEST(HARTest, Basic) {
    // Generate RV-like data with heterogeneous persistence
    std::size_t n = 500;
    std::vector<f64> rv(n);
    uint64_t seed = 123;
    rv[0] = 0.0002;
    for (std::size_t i = 1; i < n; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 noise = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.00005;
        rv[i] = 0.00005 + 0.4 * rv[i - 1] + noise;
        if (rv[i] < 1e-6) rv[i] = 1e-6;
    }
    Series<f64> daily_rv(rv);

    auto result = har_rv(daily_rv);
    ASSERT_TRUE(result.is_ok()) << "HAR failed to estimate";
    auto& res = result.value();
    EXPECT_GT(res.n_obs, 0u);
    EXPECT_EQ(res.std_errors.size(), 4u);  // const + 3 betas
}

TEST(RollingTest, RollingMean) {
    Series<f64> data(std::vector<f64>{1.0, 2.0, 3.0, 4.0, 5.0});
    auto rm = rolling_mean(data, 3);
    EXPECT_EQ(rm.size(), 3u);
    EXPECT_DOUBLE_EQ(rm[0], 2.0);  // mean(1,2,3)
    EXPECT_DOUBLE_EQ(rm[1], 3.0);  // mean(2,3,4)
    EXPECT_DOUBLE_EQ(rm[2], 4.0);  // mean(3,4,5)
}

TEST(RollingTest, RollingSum) {
    Series<f64> data(std::vector<f64>{1.0, 2.0, 3.0, 4.0, 5.0});
    auto rs = rolling_sum(data, 3);
    EXPECT_EQ(rs.size(), 3u);
    EXPECT_DOUBLE_EQ(rs[0], 6.0);
    EXPECT_DOUBLE_EQ(rs[1], 9.0);
    EXPECT_DOUBLE_EQ(rs[2], 12.0);
}

TEST(RollingTest, RollingVariance) {
    Series<f64> data(std::vector<f64>{1.0, 2.0, 3.0, 4.0, 5.0});
    auto rv = rolling_variance(data, 3);
    EXPECT_EQ(rv.size(), 3u);
    EXPECT_NEAR(rv[0], 1.0, 1e-10);  // var(1,2,3) = 1
    EXPECT_NEAR(rv[1], 1.0, 1e-10);
    EXPECT_NEAR(rv[2], 1.0, 1e-10);
}

TEST(RollingTest, RollingMinMax) {
    Series<f64> data(std::vector<f64>{3.0, 1.0, 4.0, 1.0, 5.0});
    auto rmin = rolling_min(data, 3);
    auto rmax = rolling_max(data, 3);
    EXPECT_EQ(rmin.size(), 3u);
    EXPECT_DOUBLE_EQ(rmin[0], 1.0);
    EXPECT_DOUBLE_EQ(rmax[0], 4.0);
    EXPECT_DOUBLE_EQ(rmin[2], 1.0);
    EXPECT_DOUBLE_EQ(rmax[2], 5.0);
}

TEST(RollingTest, EmptyOnSmallData) {
    Series<f64> data(std::vector<f64>{1.0, 2.0});
    auto rm = rolling_mean(data, 5);
    EXPECT_EQ(rm.size(), 0u);
}
