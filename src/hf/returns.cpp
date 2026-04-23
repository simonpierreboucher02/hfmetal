#include "hfm/hf/returns.hpp"
#include <cmath>
#include <stdexcept>

namespace hfm {

void log_returns_inplace(const f64* prices, f64* out, std::size_t n) {
    for (std::size_t i = 1; i < n; ++i) {
        out[i - 1] = std::log(prices[i] / prices[i - 1]);
    }
}

void simple_returns_inplace(const f64* prices, f64* out, std::size_t n) {
    for (std::size_t i = 1; i < n; ++i) {
        out[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
    }
}

Series<f64> log_returns(const Series<f64>& prices) {
    if (prices.size() < 2) {
        throw std::invalid_argument("log_returns: need at least 2 prices");
    }
    std::vector<f64> ret(prices.size() - 1);
    log_returns_inplace(prices.data(), ret.data(), prices.size());

    if (prices.has_timestamps()) {
        auto ts = std::vector<Timestamp>(
            prices.timestamps().begin() + 1,
            prices.timestamps().end());
        return Series<f64>(std::move(ret), std::move(ts));
    }
    return Series<f64>(std::move(ret));
}

Series<f64> simple_returns(const Series<f64>& prices) {
    if (prices.size() < 2) {
        throw std::invalid_argument("simple_returns: need at least 2 prices");
    }
    std::vector<f64> ret(prices.size() - 1);
    simple_returns_inplace(prices.data(), ret.data(), prices.size());

    if (prices.has_timestamps()) {
        auto ts = std::vector<Timestamp>(
            prices.timestamps().begin() + 1,
            prices.timestamps().end());
        return Series<f64>(std::move(ret), std::move(ts));
    }
    return Series<f64>(std::move(ret));
}

Series<f64> compute_returns(const Series<f64>& prices, ReturnType type) {
    switch (type) {
        case ReturnType::Log: return log_returns(prices);
        case ReturnType::Simple: return simple_returns(prices);
    }
    throw std::invalid_argument("unknown return type");
}

} // namespace hfm
