#include <gtest/gtest.h>
#include "hfm/timeseries/granger.hpp"
#include <cmath>

using namespace hfm;

namespace {
void generate_pair(Vector<f64>& y, Vector<f64>& x, std::size_t n,
                   f64 phi_y, f64 phi_xy, uint64_t seed) {
    y.resize(n);
    x.resize(n);
    y[0] = 0.0;
    x[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 ex = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 2.0;
        x[i] = 0.5 * x[i - 1] + ex;

        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 ey = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 2.0;
        y[i] = phi_y * y[i - 1] + phi_xy * x[i - 1] + ey;
    }
}
} // namespace

TEST(GrangerTest, CausalRelationship) {
    Vector<f64> y, x;
    generate_pair(y, x, 500, 0.3, 0.5, 42);
    auto result = granger_causality(y, x, 4);
    ASSERT_TRUE(result.is_ok());
    EXPECT_LT(result.value().p_value, 0.05);
    EXPECT_GT(result.value().f_statistic, 0.0);
}

TEST(GrangerTest, NoCausalRelationship) {
    Vector<f64> y, x;
    generate_pair(y, x, 500, 0.5, 0.0, 42);
    auto result = granger_causality(y, x, 4);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().p_value, 0.01);
}

TEST(GrangerTest, DimensionMismatch) {
    Vector<f64> y(100, 1.0), x(50, 1.0);
    EXPECT_FALSE(granger_causality(y, x, 4).is_ok());
}
