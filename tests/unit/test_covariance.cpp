#include <gtest/gtest.h>
#include "hfm/covariance/covariance.hpp"
#include "hfm/linalg/solver.hpp"

using namespace hfm;

TEST(CovarianceTest, ClassicalCovariance) {
    // Simple case: 3 obs, 2 regressors, known residuals
    Matrix<f64> X(3, 2, 0.0);
    X(0, 0) = 1; X(0, 1) = 1;
    X(1, 0) = 1; X(1, 1) = 2;
    X(2, 0) = 1; X(2, 1) = 3;

    Vector<f64> residuals{0.1, -0.2, 0.1};
    auto cov = classical_covariance(X, residuals, 3, 2);

    EXPECT_EQ(cov.rows(), 2u);
    EXPECT_EQ(cov.cols(), 2u);
    EXPECT_GT(cov(0, 0), 0.0);
    EXPECT_GT(cov(1, 1), 0.0);
}

TEST(CovarianceTest, WhiteCovariance) {
    Matrix<f64> X(4, 2, 0.0);
    X(0, 0) = 1; X(0, 1) = 1;
    X(1, 0) = 1; X(1, 1) = 2;
    X(2, 0) = 1; X(2, 1) = 3;
    X(3, 0) = 1; X(3, 1) = 4;

    Vector<f64> residuals{0.1, -0.3, 0.2, -0.1};
    Matrix<f64> XtX = matmul_AtB(X, X);
    auto XtX_inv = invert_spd(XtX);
    ASSERT_TRUE(XtX_inv.is_ok());

    auto cov = white_covariance(X, residuals, XtX_inv.value());
    EXPECT_GT(cov(0, 0), 0.0);
    EXPECT_GT(cov(1, 1), 0.0);
}

TEST(CovarianceTest, NeweyWestAutoLag) {
    EXPECT_EQ(newey_west_auto_lag(100), 4);
    EXPECT_EQ(newey_west_auto_lag(1000), 9);   // floor(1000^(1/3)) = 9
    EXPECT_EQ(newey_west_auto_lag(27), 3);
}
