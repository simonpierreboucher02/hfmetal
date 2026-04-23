#include "hfm/hf/realized_measures.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace hfm {

f64 realized_variance(const f64* returns, std::size_t n) {
    f64 rv = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        rv += returns[i] * returns[i];
    }
    return rv;
}

f64 realized_variance(const Series<f64>& returns) {
    if (returns.empty()) {
        throw std::invalid_argument("realized_variance: empty series");
    }
    return realized_variance(returns.data(), returns.size());
}

f64 realized_volatility(const Series<f64>& returns) {
    return std::sqrt(realized_variance(returns));
}

f64 bipower_variation(const f64* returns, std::size_t n) {
    if (n < 2) {
        throw std::invalid_argument("bipower_variation: need at least 2 returns");
    }
    f64 bv = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        bv += std::abs(returns[i]) * std::abs(returns[i - 1]);
    }
    // Scale factor: (pi/2) * (n/(n-1))
    bv *= (std::numbers::pi / 2.0) * (static_cast<f64>(n) / static_cast<f64>(n - 1));
    return bv;
}

f64 bipower_variation(const Series<f64>& returns) {
    if (returns.size() < 2) {
        throw std::invalid_argument("bipower_variation: need at least 2 returns");
    }
    return bipower_variation(returns.data(), returns.size());
}

f64 realized_semivariance(const Series<f64>& returns, bool positive) {
    f64 rsv = 0.0;
    for (std::size_t i = 0; i < returns.size(); ++i) {
        f64 r = returns[i];
        if (positive ? (r > 0.0) : (r < 0.0)) {
            rsv += r * r;
        }
    }
    return rsv;
}

RealizedMeasuresResult compute_realized_measures(const Series<f64>& returns) {
    RealizedMeasuresResult result;
    result.n_obs = returns.size();
    result.realized_variance = realized_variance(returns);
    result.realized_volatility = std::sqrt(result.realized_variance);
    if (returns.size() >= 2) {
        result.bipower_variation = bipower_variation(returns);
        f64 rv = result.realized_variance;
        f64 bv = result.bipower_variation;
        result.jump_statistic = (rv > bv) ? (rv - bv) : 0.0;
    }
    return result;
}

} // namespace hfm
