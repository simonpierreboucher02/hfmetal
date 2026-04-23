#include <benchmark/benchmark.h>
#include "hfm/estimators/ols.hpp"
#include <cmath>

using namespace hfm;

static void BM_OLS(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::size_t k = 5;

    Matrix<f64> X(n, k, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        for (std::size_t j = 1; j < k; ++j) {
            X(i, j) = std::sin(static_cast<f64>(i * j) * 0.01);
        }
        y[i] = 1.0 + 0.5 * X(i, 1) - 0.3 * X(i, 2);
    }

    for (auto _ : state) {
        auto result = ols(y, X);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BM_OLS)->RangeMultiplier(4)->Range(100, 100'000);

static void BM_OLS_White(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::size_t k = 3;

    Matrix<f64> X(n, k, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        for (std::size_t j = 1; j < k; ++j) {
            X(i, j) = std::sin(static_cast<f64>(i * j) * 0.01);
        }
        y[i] = 1.0 + 0.5 * X(i, 1);
    }

    OLSOptions opts;
    opts.covariance = CovarianceType::White;

    for (auto _ : state) {
        auto result = ols(y, X, opts);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BM_OLS_White)->RangeMultiplier(4)->Range(100, 100'000);

static void BM_OLS_NeweyWest(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::size_t k = 3;

    Matrix<f64> X(n, k, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        for (std::size_t j = 1; j < k; ++j) {
            X(i, j) = std::sin(static_cast<f64>(i * j) * 0.01);
        }
        y[i] = 1.0 + 0.5 * X(i, 1);
    }

    OLSOptions opts;
    opts.covariance = CovarianceType::NeweyWest;

    for (auto _ : state) {
        auto result = ols(y, X, opts);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BM_OLS_NeweyWest)->RangeMultiplier(4)->Range(100, 100'000);
