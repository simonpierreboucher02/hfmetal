#include <gtest/gtest.h>
#include "hfm/data/timestamp.hpp"

using namespace hfm;

TEST(TimestampTest, Microseconds) {
    Timestamp ts(1000000);  // 1 second
    EXPECT_EQ(ts.microseconds(), 1000000);
    EXPECT_DOUBLE_EQ(ts.seconds(), 1.0);
}

TEST(TimestampTest, FromSeconds) {
    auto ts = Timestamp::from_seconds(1.5);
    EXPECT_EQ(ts.microseconds(), 1500000);
}

TEST(TimestampTest, FromMillis) {
    auto ts = Timestamp::from_millis(1500);
    EXPECT_EQ(ts.microseconds(), 1500000);
}

TEST(TimestampTest, Comparison) {
    Timestamp a(1000);
    Timestamp b(2000);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a != b);
}

TEST(TimestampTest, SecondsSince) {
    Timestamp a(0);
    Timestamp b(5000000);
    EXPECT_DOUBLE_EQ(b.seconds_since(a), 5.0);
}

TEST(TimeRangeTest, Contains) {
    TimeRange range{Timestamp(1000), Timestamp(5000)};
    EXPECT_TRUE(range.contains(Timestamp(3000)));
    EXPECT_FALSE(range.contains(Timestamp(5000)));
    EXPECT_FALSE(range.contains(Timestamp(500)));
}
