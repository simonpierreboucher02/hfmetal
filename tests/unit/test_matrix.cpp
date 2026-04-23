#include <gtest/gtest.h>
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/solver.hpp"

using namespace hfm;

TEST(MatrixTest, Construction) {
    Matrix<f64> m(3, 4, 1.0);
    EXPECT_EQ(m.rows(), 3u);
    EXPECT_EQ(m.cols(), 4u);
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(2, 3), 1.0);
}

TEST(MatrixTest, Identity) {
    auto I = Matrix<f64>::identity(3);
    EXPECT_DOUBLE_EQ(I(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(I(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(I(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(I(2, 2), 1.0);
}

TEST(MatrixTest, Transpose) {
    Matrix<f64> m(2, 3, 0.0);
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    auto t = m.transpose();
    EXPECT_EQ(t.rows(), 3u);
    EXPECT_EQ(t.cols(), 2u);
    EXPECT_DOUBLE_EQ(t(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t(2, 1), 6.0);
}

TEST(MatrixTest, Matmul) {
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    Matrix<f64> A(2, 2, 0.0);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    Matrix<f64> B(2, 2, 0.0);
    B(0, 0) = 5; B(0, 1) = 6;
    B(1, 0) = 7; B(1, 1) = 8;
    auto C = matmul(A, B);
    EXPECT_DOUBLE_EQ(C(0, 0), 19.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 22.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 43.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 50.0);
}

TEST(MatrixTest, Matvec) {
    Matrix<f64> A(2, 2, 0.0);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    Vector<f64> x{5.0, 6.0};
    auto y = matvec(A, x);
    EXPECT_DOUBLE_EQ(y[0], 17.0);
    EXPECT_DOUBLE_EQ(y[1], 39.0);
}

TEST(SolverTest, LeastSquares) {
    // y = 2*x + 1, three points
    Matrix<f64> X(3, 2, 0.0);
    X(0, 0) = 1; X(0, 1) = 1;
    X(1, 0) = 1; X(1, 1) = 2;
    X(2, 0) = 1; X(2, 1) = 3;
    Vector<f64> y{3.0, 5.0, 7.0};

    auto result = solve_least_squares(X, y);
    ASSERT_TRUE(result.is_ok());
    auto& beta = result.value();
    EXPECT_NEAR(beta[0], 1.0, 1e-10);
    EXPECT_NEAR(beta[1], 2.0, 1e-10);
}

TEST(SolverTest, Cholesky) {
    // SPD matrix: [4 2; 2 3]
    Matrix<f64> A(2, 2, 0.0);
    A(0, 0) = 4; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 3;
    auto result = cholesky(A);
    ASSERT_TRUE(result.is_ok());
    auto& L = result.value();
    EXPECT_NEAR(L(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(L(1, 0), 1.0, 1e-10);
}

TEST(SolverTest, InvertSPD) {
    Matrix<f64> A(2, 2, 0.0);
    A(0, 0) = 4; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 3;
    auto result = invert_spd(A);
    ASSERT_TRUE(result.is_ok());
    // A^{-1} = [3/8 -1/4; -1/4 1/2]
    auto& Ainv = result.value();
    EXPECT_NEAR(Ainv(0, 0), 0.375, 1e-10);
    EXPECT_NEAR(Ainv(0, 1), -0.25, 1e-10);
    EXPECT_NEAR(Ainv(1, 0), -0.25, 1e-10);
    EXPECT_NEAR(Ainv(1, 1), 0.5, 1e-10);
}
