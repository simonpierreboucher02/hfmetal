#include <gtest/gtest.h>
#include "hfm/models/logit_probit.hpp"
#include <cmath>

using namespace hfm;

namespace {
void generate_binary_data(std::size_t n, Vector<f64>& y, Matrix<f64>& X, uint64_t seed) {
    y = Vector<f64>(n);
    X = Matrix<f64>(n, 2);
    for (std::size_t i = 0; i < n; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 x1 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 2.0;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 x2 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 2.0;
        X(i, 0) = x1;
        X(i, 1) = x2;
        f64 z = -0.5 + 1.5 * x1 - 0.8 * x2;
        f64 prob = 1.0 / (1.0 + std::exp(-z));
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 u = static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31);
        y[i] = (u < prob) ? 1.0 : 0.0;
    }
}
}

TEST(LogitTest, BasicEstimation) {
    Vector<f64> y;
    Matrix<f64> X;
    generate_binary_data(500, y, X, 42);

    auto result = logit(y, X);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.n_obs, 500u);
    EXPECT_EQ(res.n_regressors, 3u); // 2 + intercept
    EXPECT_EQ(res.type, BinaryModelType::Logit);
    EXPECT_GT(res.pseudo_r_squared, 0.0);
    EXPECT_LT(res.pseudo_r_squared, 1.0);
    EXPECT_GT(res.log_likelihood, res.log_likelihood_null);
}

TEST(LogitTest, PredictedProbabilities) {
    Vector<f64> y;
    Matrix<f64> X;
    generate_binary_data(300, y, X, 77);

    auto result = logit(y, X);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    for (std::size_t i = 0; i < res.predicted_prob.size(); ++i) {
        EXPECT_GE(res.predicted_prob[i], 0.0);
        EXPECT_LE(res.predicted_prob[i], 1.0);
    }
}

TEST(LogitTest, MarginalEffects) {
    Vector<f64> y;
    Matrix<f64> X;
    generate_binary_data(500, y, X, 99);

    auto result = logit(y, X);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(result.value().marginal_effects.size(), 3u);
}

TEST(ProbitTest, BasicEstimation) {
    Vector<f64> y;
    Matrix<f64> X;
    generate_binary_data(500, y, X, 42);

    auto result = probit(y, X);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.type, BinaryModelType::Probit);
    EXPECT_GT(res.pseudo_r_squared, 0.0);
}

TEST(LogitTest, InvalidY) {
    Vector<f64> y(10, 0.5); // not 0/1
    Matrix<f64> X(10, 2, 1.0);
    auto result = logit(y, X);
    EXPECT_FALSE(result.is_ok());
}

TEST(LogitTest, InformationCriteria) {
    Vector<f64> y;
    Matrix<f64> X;
    generate_binary_data(500, y, X, 123);

    auto result = logit(y, X);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();

    EXPECT_GT(res.aic, 0.0);
    EXPECT_GT(res.bic, 0.0);
    EXPECT_LT(res.aic, res.bic); // AIC < BIC for large n
}
