#include <benchmark/benchmark.h>
#include "hfm/hf/realized_measures.hpp"
#include <cmath>

using namespace hfm;

static void BM_RealizedVariance(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::vector<f64> returns(n);
    for (std::size_t i = 0; i < n; ++i) {
        returns[i] = std::sin(static_cast<f64>(i) * 0.01) * 0.01;
    }
    Series<f64> ret_series(returns);

    for (auto _ : state) {
        auto rv = realized_variance(ret_series);
        benchmark::DoNotOptimize(rv);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n));
}

BENCHMARK(BM_RealizedVariance)->RangeMultiplier(10)->Range(100, 10'000'000);

static void BM_BipowerVariation(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::vector<f64> returns(n);
    for (std::size_t i = 0; i < n; ++i) {
        returns[i] = std::sin(static_cast<f64>(i) * 0.01) * 0.01;
    }
    Series<f64> ret_series(returns);

    for (auto _ : state) {
        auto bv = bipower_variation(ret_series);
        benchmark::DoNotOptimize(bv);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n));
}

BENCHMARK(BM_BipowerVariation)->RangeMultiplier(10)->Range(100, 10'000'000);

static void BM_ComputeAllMeasures(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    std::vector<f64> returns(n);
    for (std::size_t i = 0; i < n; ++i) {
        returns[i] = std::sin(static_cast<f64>(i) * 0.01) * 0.01;
    }
    Series<f64> ret_series(returns);

    for (auto _ : state) {
        auto result = compute_realized_measures(ret_series);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n));
}

BENCHMARK(BM_ComputeAllMeasures)->RangeMultiplier(10)->Range(100, 10'000'000);
