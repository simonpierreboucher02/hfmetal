#include <benchmark/benchmark.h>
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"
#include <cmath>

using namespace hfm;

static void BM_Matmul(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    Matrix<f64> A(n, n, 1.0);
    Matrix<f64> B(n, n, 1.0);

    for (auto _ : state) {
        auto C = matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n * n * n));
}

BENCHMARK(BM_Matmul)->RangeMultiplier(2)->Range(16, 1024);

static void BM_VectorDot(benchmark::State& state) {
    auto n = static_cast<std::size_t>(state.range(0));
    Vector<f64> a(n, 1.0);
    Vector<f64> b(n, 1.0);

    for (auto _ : state) {
        auto d = a.dot(b);
        benchmark::DoNotOptimize(d);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n));
}

BENCHMARK(BM_VectorDot)->RangeMultiplier(10)->Range(100, 10'000'000);
