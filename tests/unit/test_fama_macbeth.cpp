#include <gtest/gtest.h>
#include "hfm/models/fama_macbeth.hpp"
#include <cmath>

using namespace hfm;

TEST(FamaMacBethTest, Basic) {
    // 10 entities, 50 time periods
    std::size_t n_entities = 10;
    std::size_t n_periods = 50;
    std::size_t n = n_entities * n_periods;

    Vector<f64> y(n);
    Matrix<f64> X(n, 2, 0.0);  // const + one factor
    Vector<i64> time_ids(n);

    f64 true_gamma0 = 0.01;
    f64 true_gamma1 = 0.5;

    for (std::size_t t = 0; t < n_periods; ++t) {
        for (std::size_t e = 0; e < n_entities; ++e) {
            std::size_t idx = t * n_entities + e;
            f64 beta = static_cast<f64>(e + 1) / 5.0;
            X(idx, 0) = 1.0;
            X(idx, 1) = beta;
            y[idx] = true_gamma0 + true_gamma1 * beta +
                     std::sin(static_cast<f64>(idx) * 0.1) * 0.01;
            time_ids[idx] = static_cast<i64>(t);
        }
    }

    auto result = fama_macbeth(y, X, time_ids);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_EQ(res.n_periods, n_periods);
    EXPECT_NEAR(res.gamma[0], true_gamma0, 0.05);
    EXPECT_NEAR(res.gamma[1], true_gamma1, 0.1);
    EXPECT_GT(res.std_errors[0], 0.0);
    EXPECT_GT(res.std_errors[1], 0.0);
}

TEST(FamaMacBethTest, NoNWCorrection) {
    std::size_t n_entities = 5;
    std::size_t n_periods = 30;
    std::size_t n = n_entities * n_periods;

    Vector<f64> y(n);
    Matrix<f64> X(n, 2, 0.0);
    Vector<i64> time_ids(n);

    for (std::size_t t = 0; t < n_periods; ++t) {
        for (std::size_t e = 0; e < n_entities; ++e) {
            std::size_t idx = t * n_entities + e;
            X(idx, 0) = 1.0;
            X(idx, 1) = static_cast<f64>(e + 1);
            y[idx] = 0.02 + 0.1 * X(idx, 1);
            time_ids[idx] = static_cast<i64>(t);
        }
    }

    FamaMacBethOptions opts;
    opts.newey_west_correction = false;
    auto result = fama_macbeth(y, X, time_ids, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_NEAR(res.gamma[1], 0.1, 1e-8);
}
