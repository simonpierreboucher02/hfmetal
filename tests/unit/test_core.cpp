#include <gtest/gtest.h>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/core/numeric_traits.hpp"

using namespace hfm;

TEST(StatusTest, DefaultIsOk) {
    Status s;
    EXPECT_TRUE(s.is_ok());
    EXPECT_EQ(s.code(), ErrorCode::Ok);
}

TEST(StatusTest, ErrorStatus) {
    auto s = Status::error(ErrorCode::DimensionMismatch, "bad dims");
    EXPECT_FALSE(s.is_ok());
    EXPECT_EQ(s.code(), ErrorCode::DimensionMismatch);
    EXPECT_EQ(s.message(), "bad dims");
}

TEST(ResultTest, ValueResult) {
    Result<int> r(42);
    EXPECT_TRUE(r.is_ok());
    EXPECT_EQ(r.value(), 42);
}

TEST(ResultTest, ErrorResult) {
    Result<int> r(Status::error(ErrorCode::InvalidArgument, "oops"));
    EXPECT_FALSE(r.is_ok());
    EXPECT_THROW(r.value(), std::runtime_error);
}

TEST(NumericTraitsTest, ApproxEqual) {
    EXPECT_TRUE(NumericTraits<f64>::approx_equal(1.0, 1.0 + 1e-14));
    EXPECT_FALSE(NumericTraits<f64>::approx_equal(1.0, 2.0));
}

TEST(NumericTraitsTest, NanHandling) {
    EXPECT_TRUE(NumericTraits<f64>::is_nan(NumericTraits<f64>::nan()));
    EXPECT_TRUE(NumericTraits<f64>::approx_equal(
        NumericTraits<f64>::nan(), NumericTraits<f64>::nan()));
}
