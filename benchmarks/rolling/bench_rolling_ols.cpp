#include <benchmark/benchmark.h>
#include "hfm/estimators/rolling_ols.hpp"
#include "hfm/estimators/batched_ols.hpp"
#include <cmath>

using namespace hfm;

static void BM_RollingOLS(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::size_t k = 3;
    Matrix<f64> X(n, k, 0.0);
    Vector<f64> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = std::sin(static_cast<f64>(i) * 0.01);
        X(i, 2) = std::cos(static_cast<f64>(i) * 0.01);
        y[i] = 1.0 + 0.5 * X(i, 1) - 0.3 * X(i, 2);
    }

    RollingOptions opts;
    opts.window = 250;
    opts.step = 1;

    for (auto _ : state) {
        auto result = rolling_ols(y, X, opts);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n));
}

BENCHMARK(BM_RollingOLS)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000)->Arg(10000);

static void BM_BatchedOLS(benchmark::State& state) {
    auto n_reg = static_cast<std::size_t>(state.range(0));
    std::size_t n = 200;
    std::size_t k = 3;
    Matrix<f64> X(n, k, 0.0);
    Matrix<f64> Y(n, n_reg, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i) / 100.0;
        X(i, 2) = std::sin(static_cast<f64>(i) * 0.05);
        for (std::size_t r = 0; r < n_reg; ++r) {
            Y(i, r) = 1.0 + 0.5 * X(i, 1) + std::sin(static_cast<f64>(i * r)) * 0.01;
        }
    }

    for (auto _ : state) {
        auto result = batched_ols(Y, X);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n_reg));
}

BENCHMARK(BM_BatchedOLS)->Arg(10)->Arg(50)->Arg(100)->Arg(500)->Arg(1000);
