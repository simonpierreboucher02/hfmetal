#include <gtest/gtest.h>
#include "hfm/simulation/bootstrap.hpp"
#include "hfm/estimators/ols.hpp"
#include "hfm/linalg/solver.hpp"
#include <cmath>

using namespace hfm;

TEST(BootstrapTest, MeanEstimator) {
    Vector<f64> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    auto mean_fn = [](const Vector<f64>& d) -> Vector<f64> {
        return Vector<f64>{d.mean()};
    };

    BootstrapOptions opts;
    opts.n_bootstrap = 2000;
    opts.seed = 123;

    auto result = bootstrap(data, mean_fn, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_EQ(res.n_bootstrap, 2000u);
    EXPECT_NEAR(res.estimate[0], 5.5, 1e-10);
    EXPECT_NEAR(res.mean[0], 5.5, 0.5);
    EXPECT_GT(res.std_error[0], 0.0);
    EXPECT_LT(res.ci_lower[0], res.ci_upper[0]);
}

TEST(BootstrapTest, BlockBootstrap) {
    Vector<f64> data(100);
    for (std::size_t i = 0; i < 100; ++i) {
        data[i] = std::sin(static_cast<f64>(i) * 0.1) + 1.0;
    }

    auto mean_fn = [](const Vector<f64>& d) -> Vector<f64> {
        return Vector<f64>{d.mean()};
    };

    BootstrapOptions opts;
    opts.n_bootstrap = 500;
    opts.type = BootstrapType::Block;
    opts.block_size = 10;
    opts.seed = 42;

    auto result = bootstrap(data, mean_fn, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_GT(res.std_error[0], 0.0);
}

TEST(BootstrapTest, PairsBootstrapOLS) {
    std::size_t n = 100;
    Vector<f64> y(n);
    Matrix<f64> X(n, 2, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i) / 50.0;
        y[i] = 3.0 + 2.0 * X(i, 1) + std::sin(static_cast<f64>(i)) * 0.3;
    }

    auto ols_fn = [](const Vector<f64>& yb, const Matrix<f64>& Xb) -> Vector<f64> {
        auto res = solve_least_squares(Xb, yb);
        if (res) return res.value();
        return Vector<f64>(Xb.cols(), 0.0);
    };

    BootstrapOptions opts;
    opts.n_bootstrap = 500;
    opts.seed = 99;

    auto result = bootstrap_pairs(y, X, ols_fn, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_NEAR(res.estimate[0], 3.0, 0.5);
    EXPECT_NEAR(res.estimate[1], 2.0, 0.5);
    EXPECT_GT(res.std_error[0], 0.0);
    EXPECT_GT(res.std_error[1], 0.0);
}
