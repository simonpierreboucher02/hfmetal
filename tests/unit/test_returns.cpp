#include <gtest/gtest.h>
#include "hfm/hf/returns.hpp"
#include <cmath>

using namespace hfm;

TEST(ReturnsTest, LogReturns) {
    Series<f64> prices(std::vector<f64>{100.0, 110.0, 105.0, 115.0});
    auto r = log_returns(prices);
    EXPECT_EQ(r.size(), 3u);
    EXPECT_NEAR(r[0], std::log(110.0 / 100.0), 1e-12);
    EXPECT_NEAR(r[1], std::log(105.0 / 110.0), 1e-12);
    EXPECT_NEAR(r[2], std::log(115.0 / 105.0), 1e-12);
}

TEST(ReturnsTest, SimpleReturns) {
    Series<f64> prices(std::vector<f64>{100.0, 110.0, 105.0});
    auto r = simple_returns(prices);
    EXPECT_EQ(r.size(), 2u);
    EXPECT_NEAR(r[0], 0.10, 1e-12);
    EXPECT_NEAR(r[1], -5.0 / 110.0, 1e-12);
}

TEST(ReturnsTest, LogReturnsPreservesTimestamps) {
    std::vector<f64> vals{100.0, 110.0, 120.0};
    std::vector<Timestamp> ts{Timestamp(1000), Timestamp(2000), Timestamp(3000)};
    Series<f64> prices(vals, ts);
    auto r = log_returns(prices);
    EXPECT_TRUE(r.has_timestamps());
    EXPECT_EQ(r.timestamps()[0].microseconds(), 2000);
}

TEST(ReturnsTest, TooFewPrices) {
    Series<f64> prices(std::vector<f64>{100.0});
    EXPECT_THROW(log_returns(prices), std::invalid_argument);
}

TEST(ReturnsTest, ComputeReturnsDispatch) {
    Series<f64> prices(std::vector<f64>{100.0, 110.0, 120.0});
    auto r_log = compute_returns(prices, ReturnType::Log);
    auto r_simple = compute_returns(prices, ReturnType::Simple);
    EXPECT_NEAR(r_log[0], std::log(1.1), 1e-12);
    EXPECT_NEAR(r_simple[0], 0.1, 1e-12);
}
