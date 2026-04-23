#include <gtest/gtest.h>
#include "hfm/data/series.hpp"

using namespace hfm;

TEST(SeriesTest, Construction) {
    Series<f64> s(5, 1.0);
    EXPECT_EQ(s.size(), 5u);
    EXPECT_DOUBLE_EQ(s[0], 1.0);
}

TEST(SeriesTest, VectorConstruction) {
    Series<f64> s(std::vector<f64>{1.0, 2.0, 3.0});
    EXPECT_EQ(s.size(), 3u);
    EXPECT_DOUBLE_EQ(s[1], 2.0);
}

TEST(SeriesTest, WithTimestamps) {
    std::vector<f64> vals{1.0, 2.0, 3.0};
    std::vector<Timestamp> ts{Timestamp(1000), Timestamp(2000), Timestamp(3000)};
    Series<f64> s(vals, ts);
    EXPECT_TRUE(s.has_timestamps());
    EXPECT_EQ(s.timestamps()[0].microseconds(), 1000);
}

TEST(SeriesTest, Slice) {
    Series<f64> s(std::vector<f64>{1.0, 2.0, 3.0, 4.0, 5.0});
    auto sub = s.slice(1, 4);
    EXPECT_EQ(sub.size(), 3u);
    EXPECT_DOUBLE_EQ(sub[0], 2.0);
    EXPECT_DOUBLE_EQ(sub[2], 4.0);
}

TEST(SeriesTest, PushBack) {
    Series<f64> s;
    s.push_back(1.0);
    s.push_back(2.0);
    EXPECT_EQ(s.size(), 2u);
    EXPECT_DOUBLE_EQ(s[1], 2.0);
}
