#pragma once

#include <cmath>
#include <limits>
#include "hfm/core/types.hpp"

namespace hfm {

template <typename T>
struct NumericTraits {
    static constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }
    static constexpr T min() { return std::numeric_limits<T>::min(); }
    static constexpr T max() { return std::numeric_limits<T>::max(); }
    static constexpr T nan() { return std::numeric_limits<T>::quiet_NaN(); }
    static constexpr T infinity() { return std::numeric_limits<T>::infinity(); }

    static bool is_nan(T v) { return std::isnan(v); }
    static bool is_finite(T v) { return std::isfinite(v); }

    static constexpr T tolerance() {
        if constexpr (std::is_same_v<T, f32>) return 1e-6f;
        else return 1e-12;
    }

    static bool approx_equal(T a, T b, T tol = tolerance()) {
        if (is_nan(a) && is_nan(b)) return true;
        return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
    }
};

} // namespace hfm
