#pragma once

#include "hfm/core/types.hpp"
#include "hfm/data/series.hpp"

namespace hfm {

struct RealizedMeasuresResult {
    f64 realized_variance = 0.0;
    f64 realized_volatility = 0.0;
    f64 bipower_variation = 0.0;
    f64 jump_statistic = 0.0;
    std::size_t n_obs = 0;
};

f64 realized_variance(const Series<f64>& returns);
f64 realized_variance(const f64* returns, std::size_t n);

f64 realized_volatility(const Series<f64>& returns);

f64 bipower_variation(const Series<f64>& returns);
f64 bipower_variation(const f64* returns, std::size_t n);

f64 realized_semivariance(const Series<f64>& returns, bool positive = false);

RealizedMeasuresResult compute_realized_measures(const Series<f64>& returns);

} // namespace hfm
