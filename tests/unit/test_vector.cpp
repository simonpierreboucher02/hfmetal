#include <gtest/gtest.h>
#include "hfm/linalg/vector.hpp"

using namespace hfm;

TEST(VectorTest, Construction) {
    Vector<f64> v(5, 1.0);
    EXPECT_EQ(v.size(), 5u);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[4], 1.0);
}

TEST(VectorTest, InitializerList) {
    Vector<f64> v{1.0, 2.0, 3.0};
    EXPECT_EQ(v.size(), 3u);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
}

TEST(VectorTest, Dot) {
    Vector<f64> a{1.0, 2.0, 3.0};
    Vector<f64> b{4.0, 5.0, 6.0};
    EXPECT_DOUBLE_EQ(a.dot(b), 32.0);
}

TEST(VectorTest, Norm) {
    Vector<f64> v{3.0, 4.0};
    EXPECT_DOUBLE_EQ(v.norm(), 5.0);
}

TEST(VectorTest, Sum) {
    Vector<f64> v{1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(v.sum(), 10.0);
}

TEST(VectorTest, Mean) {
    Vector<f64> v{1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(v.mean(), 2.5);
}

TEST(VectorTest, Add) {
    Vector<f64> a{1.0, 2.0};
    Vector<f64> b{3.0, 4.0};
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c[0], 4.0);
    EXPECT_DOUBLE_EQ(c[1], 6.0);
}

TEST(VectorTest, Sub) {
    Vector<f64> a{5.0, 3.0};
    Vector<f64> b{2.0, 1.0};
    auto c = a - b;
    EXPECT_DOUBLE_EQ(c[0], 3.0);
    EXPECT_DOUBLE_EQ(c[1], 2.0);
}

TEST(VectorTest, ScalarMul) {
    Vector<f64> v{1.0, 2.0, 3.0};
    auto r = v * 2.0;
    EXPECT_DOUBLE_EQ(r[0], 2.0);
    EXPECT_DOUBLE_EQ(r[2], 6.0);
}

TEST(VectorFloat, DotF32) {
    Vector<f32> a{1.0f, 2.0f, 3.0f};
    Vector<f32> b{4.0f, 5.0f, 6.0f};
    EXPECT_FLOAT_EQ(a.dot(b), 32.0f);
}
