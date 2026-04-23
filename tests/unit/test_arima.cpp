#include <gtest/gtest.h>
#include "hfm/timeseries/arima.hpp"
#include <cmath>

using namespace hfm;

namespace {
Vector<f64> generate_ar1(std::size_t n, f64 phi, uint64_t seed) {
    Vector<f64> y(n);
    y[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 2.0;
        y[i] = phi * y[i - 1] + z;
    }
    return y;
}
} // namespace

TEST(ARIMATest, Differencing) {
    Vector<f64> y({1.0, 3.0, 6.0, 10.0, 15.0});
    auto dy = difference(y, 1);
    EXPECT_EQ(dy.size(), 4u);
    EXPECT_NEAR(dy[0], 2.0, 1e-10);
    EXPECT_NEAR(dy[1], 3.0, 1e-10);
    EXPECT_NEAR(dy[2], 4.0, 1e-10);
    EXPECT_NEAR(dy[3], 5.0, 1e-10);
}

TEST(ARIMATest, DoubleDifferencing) {
    Vector<f64> y({1.0, 3.0, 6.0, 10.0, 15.0});
    auto dy = difference(y, 2);
    EXPECT_EQ(dy.size(), 3u);
    EXPECT_NEAR(dy[0], 1.0, 1e-10);
    EXPECT_NEAR(dy[1], 1.0, 1e-10);
    EXPECT_NEAR(dy[2], 1.0, 1e-10);
}

TEST(ARIMATest, AREstimation) {
    auto y = generate_ar1(500, 0.6, 42);
    ARIMAOptions opts;
    opts.p = 1;
    opts.d = 0;
    opts.q = 0;
    auto result = arima(y, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_NEAR(res.ar_coefficients[0], 0.6, 0.15);
    EXPECT_GT(res.residuals.size(), 0u);
}

TEST(ARIMATest, ARIMAWithDifferencing) {
    // Generate random walk (unit root)
    Vector<f64> rw(300);
    rw[0] = 100.0;
    uint64_t seed = 42;
    for (std::size_t i = 1; i < 300; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 2.0;
        rw[i] = rw[i - 1] + z;
    }
    ARIMAOptions opts;
    opts.p = 1;
    opts.d = 1;
    opts.q = 0;
    auto result = arima(rw, opts);
    ASSERT_TRUE(result.is_ok());
}

TEST(ARIMATest, Forecast) {
    auto y = generate_ar1(500, 0.5, 42);
    ARIMAOptions opts;
    opts.p = 1;
    auto model = arima(y, opts);
    ASSERT_TRUE(model.is_ok());
    auto forecast = arima_forecast(y, model.value(), 10, 0.95);
    ASSERT_TRUE(forecast.is_ok());
    auto& f = forecast.value();
    EXPECT_EQ(f.forecast.size(), 10u);
    EXPECT_EQ(f.lower.size(), 10u);
    EXPECT_EQ(f.upper.size(), 10u);
    for (std::size_t h = 0; h < 10; ++h) {
        EXPECT_LT(f.lower[h], f.forecast[h]);
        EXPECT_LT(f.forecast[h], f.upper[h]);
    }
}

TEST(ARIMATest, TooFewObservations) {
    Vector<f64> y(5, 1.0);
    EXPECT_FALSE(arima(y).is_ok());
}
