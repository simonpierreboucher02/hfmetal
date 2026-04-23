#pragma once

#include "hfm/core/types.hpp"
#include "hfm/data/series.hpp"

namespace hfm {

enum class ReturnType : u32 {
    Log,
    Simple
};

Series<f64> log_returns(const Series<f64>& prices);
Series<f64> simple_returns(const Series<f64>& prices);
Series<f64> compute_returns(const Series<f64>& prices, ReturnType type);

void log_returns_inplace(const f64* prices, f64* out, std::size_t n);
void simple_returns_inplace(const f64* prices, f64* out, std::size_t n);

} // namespace hfm
