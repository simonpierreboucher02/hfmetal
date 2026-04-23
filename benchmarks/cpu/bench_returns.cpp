#include <benchmark/benchmark.h>
#include "hfm/hf/returns.hpp"
#include <cmath>

using namespace hfm;

static void BM_LogReturns(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::vector<f64> prices(n);
    prices[0] = 100.0;
    for (std::size_t i = 1; i < n; ++i) {
        prices[i] = prices[i - 1] * (1.0 + std::sin(static_cast<f64>(i) * 0.01) * 0.001);
    }
    Series<f64> price_series(prices);

    for (auto _ : state) {
        auto r = log_returns(price_series);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n));
}

BENCHMARK(BM_LogReturns)->RangeMultiplier(10)->Range(100, 10'000'000);

static void BM_SimpleReturns(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::vector<f64> prices(n);
    prices[0] = 100.0;
    for (std::size_t i = 1; i < n; ++i) {
        prices[i] = prices[i - 1] * (1.0 + std::sin(static_cast<f64>(i) * 0.01) * 0.001);
    }
    Series<f64> price_series(prices);

    for (auto _ : state) {
        auto r = simple_returns(price_series);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n));
}

BENCHMARK(BM_SimpleReturns)->RangeMultiplier(10)->Range(100, 10'000'000);
