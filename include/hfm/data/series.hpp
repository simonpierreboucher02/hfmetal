#pragma once

#include <vector>
#include <span>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include "hfm/core/types.hpp"
#include "hfm/data/timestamp.hpp"

namespace hfm {

template <typename T>
class Series {
public:
    Series() = default;

    explicit Series(std::size_t n, T fill = T{})
        : values_(n, fill) {}

    Series(std::vector<T> values)
        : values_(std::move(values)) {}

    Series(std::vector<T> values, std::vector<Timestamp> times)
        : values_(std::move(values)), timestamps_(std::move(times)) {
        if (!timestamps_.empty() && timestamps_.size() != values_.size()) {
            throw std::invalid_argument("values and timestamps size mismatch");
        }
    }

    std::size_t size() const { return values_.size(); }
    bool empty() const { return values_.empty(); }
    bool has_timestamps() const { return !timestamps_.empty(); }

    T& operator[](std::size_t i) { return values_[i]; }
    const T& operator[](std::size_t i) const { return values_[i]; }

    T* data() { return values_.data(); }
    const T* data() const { return values_.data(); }

    std::span<T> span() { return values_; }
    std::span<const T> span() const { return values_; }

    const std::vector<T>& values() const { return values_; }
    std::vector<T>& values() { return values_; }

    const std::vector<Timestamp>& timestamps() const { return timestamps_; }

    void push_back(T val) { values_.push_back(val); }
    void push_back(T val, Timestamp ts) {
        values_.push_back(val);
        timestamps_.push_back(ts);
    }

    void reserve(std::size_t n) {
        values_.reserve(n);
        if (has_timestamps()) timestamps_.reserve(n);
    }

    Series<T> slice(std::size_t start, std::size_t end) const {
        auto vals = std::vector<T>(values_.begin() + static_cast<std::ptrdiff_t>(start),
                                   values_.begin() + static_cast<std::ptrdiff_t>(end));
        if (has_timestamps()) {
            auto ts = std::vector<Timestamp>(
                timestamps_.begin() + static_cast<std::ptrdiff_t>(start),
                timestamps_.begin() + static_cast<std::ptrdiff_t>(end));
            return Series<T>(std::move(vals), std::move(ts));
        }
        return Series<T>(std::move(vals));
    }

    auto begin() { return values_.begin(); }
    auto end() { return values_.end(); }
    auto begin() const { return values_.begin(); }
    auto end() const { return values_.end(); }

private:
    std::vector<T> values_;
    std::vector<Timestamp> timestamps_;
};

using DoubleSeries = Series<f64>;
using FloatSeries = Series<f32>;

} // namespace hfm
