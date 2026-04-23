#pragma once

#include <vector>
#include <functional>
#include <random>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"

namespace hfm {

enum class BootstrapType : u32 {
    IID,
    Block,
    Circular
};

struct BootstrapOptions {
    std::size_t n_bootstrap = 1000;
    std::size_t block_size = 1;
    BootstrapType type = BootstrapType::IID;
    u64 seed = 42;
    f64 confidence_level = 0.95;
    Backend backend = Backend::Auto;

    BootstrapOptions& set_n_bootstrap(std::size_t n) { n_bootstrap = n; return *this; }
    BootstrapOptions& set_block_size(std::size_t b) { block_size = b; return *this; }
    BootstrapOptions& set_type(BootstrapType t) { type = t; return *this; }
    BootstrapOptions& set_seed(u64 s) { seed = s; return *this; }
    BootstrapOptions& set_confidence(f64 cl) { confidence_level = cl; return *this; }
};

struct BootstrapResult {
    Vector<f64> estimate;              // original statistic
    Matrix<f64> bootstrap_samples;     // n_bootstrap x dim
    Vector<f64> mean;                  // bootstrap mean
    Vector<f64> std_error;             // bootstrap SE
    Vector<f64> ci_lower;
    Vector<f64> ci_upper;
    Vector<f64> p_value;               // two-sided test H0: stat = 0
    f64 confidence_level = 0.95;
    std::size_t n_bootstrap = 0;
    f64 elapsed_ms = 0.0;
};

using StatisticFn = std::function<Vector<f64>(const Vector<f64>& data)>;
using PairedStatisticFn = std::function<Vector<f64>(const Vector<f64>& y, const Matrix<f64>& X)>;

Result<BootstrapResult> bootstrap(const Vector<f64>& data,
                                   const StatisticFn& statistic,
                                   const BootstrapOptions& opts = BootstrapOptions{});

Result<BootstrapResult> bootstrap_pairs(const Vector<f64>& y,
                                         const Matrix<f64>& X,
                                         const PairedStatisticFn& statistic,
                                         const BootstrapOptions& opts = BootstrapOptions{});

std::vector<std::size_t> generate_block_indices(std::size_t n, std::size_t block_size,
                                                 std::mt19937_64& rng);

} // namespace hfm
