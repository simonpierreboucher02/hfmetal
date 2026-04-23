#include <gtest/gtest.h>
#include "hfm/panel/fixed_effects.hpp"
#include <cmath>

using namespace hfm;

TEST(PanelTest, EntityFixedEffects) {
    // 3 entities, 50 periods each
    std::size_t n_entities = 3;
    std::size_t n_periods = 50;
    std::size_t n = n_entities * n_periods;

    Vector<f64> y(n);
    Matrix<f64> X(n, 1, 0.0);
    Vector<i64> entity_ids(n);
    Vector<i64> time_ids(n);

    f64 true_beta = 2.5;
    f64 entity_effects[] = {1.0, -0.5, 0.3};

    for (std::size_t e = 0; e < n_entities; ++e) {
        for (std::size_t t = 0; t < n_periods; ++t) {
            std::size_t idx = e * n_periods + t;
            f64 x = static_cast<f64>(t) / 10.0;
            X(idx, 0) = x;
            y[idx] = entity_effects[e] + true_beta * x;
            entity_ids[idx] = static_cast<i64>(e);
            time_ids[idx] = static_cast<i64>(t);
        }
    }

    PanelOptions opts;
    opts.effect = PanelEffect::Entity;
    opts.cluster_entity = false;

    auto result = fixed_effects(y, X, entity_ids, time_ids, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_NEAR(res.coefficients()[0], true_beta, 1e-8);
    EXPECT_NEAR(res.r_squared_within(), 1.0, 1e-8);
    EXPECT_EQ(res.n_groups(), 3u);
}

TEST(PanelTest, ClusteredSE) {
    std::size_t n_entities = 5;
    std::size_t n_periods = 100;
    std::size_t n = n_entities * n_periods;

    Vector<f64> y(n);
    Matrix<f64> X(n, 1, 0.0);
    Vector<i64> entity_ids(n);
    Vector<i64> time_ids(n);

    for (std::size_t e = 0; e < n_entities; ++e) {
        for (std::size_t t = 0; t < n_periods; ++t) {
            std::size_t idx = e * n_periods + t;
            f64 x = static_cast<f64>(t) / 50.0;
            X(idx, 0) = x;
            y[idx] = static_cast<f64>(e) + 1.5 * x +
                     std::sin(static_cast<f64>(t) * 0.1 + static_cast<f64>(e)) * 0.3;
            entity_ids[idx] = static_cast<i64>(e);
            time_ids[idx] = static_cast<i64>(t);
        }
    }

    PanelOptions opts;
    opts.cluster_entity = true;
    auto result = fixed_effects(y, X, entity_ids, time_ids, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_GT(res.std_errors()[0], 0.0);
    EXPECT_NEAR(res.coefficients()[0], 1.5, 0.2);
}

TEST(PanelTest, TwoWayCluster) {
    std::size_t n_entities = 5;
    std::size_t n_periods = 30;
    std::size_t n = n_entities * n_periods;

    Vector<f64> y(n);
    Matrix<f64> X(n, 1, 0.0);
    Vector<i64> entity_ids(n);
    Vector<i64> time_ids(n);

    for (std::size_t e = 0; e < n_entities; ++e) {
        for (std::size_t t = 0; t < n_periods; ++t) {
            std::size_t idx = e * n_periods + t;
            X(idx, 0) = static_cast<f64>(t) / 10.0;
            y[idx] = static_cast<f64>(e) + 2.0 * X(idx, 0);
            entity_ids[idx] = static_cast<i64>(e);
            time_ids[idx] = static_cast<i64>(t);
        }
    }

    PanelOptions opts;
    opts.cluster_entity = true;
    opts.cluster_time = true;
    auto result = fixed_effects(y, X, entity_ids, time_ids, opts);
    ASSERT_TRUE(result.is_ok());
    EXPECT_GT(result.value().std_errors()[0], 0.0);
}

TEST(PanelTest, Summary) {
    std::size_t n = 100;
    Vector<f64> y(n);
    Matrix<f64> X(n, 1, 0.0);
    Vector<i64> entity_ids(n);
    Vector<i64> time_ids(n);

    for (std::size_t i = 0; i < n; ++i) {
        entity_ids[i] = static_cast<i64>(i % 5);
        time_ids[i] = static_cast<i64>(i / 5);
        X(i, 0) = static_cast<f64>(i) / 50.0;
        y[i] = static_cast<f64>(i % 5) + 1.0 * X(i, 0);
    }

    auto result = fixed_effects(y, X, entity_ids, time_ids);
    ASSERT_TRUE(result.is_ok());
    auto summary = result.value().summary();
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("Panel"), std::string::npos);
}
