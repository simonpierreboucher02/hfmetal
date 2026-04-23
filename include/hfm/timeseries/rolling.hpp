#pragma once

#include <functional>
#include <vector>
#include "hfm/core/types.hpp"
#include "hfm/data/series.hpp"

namespace hfm {

template <typename T>
Series<T> rolling_apply(const Series<T>& data, std::size_t window,
                         std::function<T(const T*, std::size_t)> func) {
    if (data.size() < window) return Series<T>();
    std::size_t n_out = data.size() - window + 1;
    std::vector<T> out(n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
        out[i] = func(data.data() + i, window);
    }
    return Series<T>(std::move(out));
}

template <typename T>
Series<T> expanding_apply(const Series<T>& data, std::size_t min_obs,
                           std::function<T(const T*, std::size_t)> func) {
    if (data.size() < min_obs) return Series<T>();
    std::size_t n_out = data.size() - min_obs + 1;
    std::vector<T> out(n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
        out[i] = func(data.data(), min_obs + i);
    }
    return Series<T>(std::move(out));
}

Series<f64> rolling_mean(const Series<f64>& data, std::size_t window);
Series<f64> rolling_variance(const Series<f64>& data, std::size_t window);
Series<f64> rolling_sum(const Series<f64>& data, std::size_t window);
Series<f64> rolling_min(const Series<f64>& data, std::size_t window);
Series<f64> rolling_max(const Series<f64>& data, std::size_t window);

} // namespace hfm
