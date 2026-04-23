#pragma once

#include <cstdint>
#include <chrono>
#include <compare>
#include <string>
#include "hfm/core/types.hpp"

namespace hfm {

class Timestamp {
public:
    using duration_type = std::chrono::microseconds;
    using clock_type = std::chrono::system_clock;
    using time_point = std::chrono::time_point<clock_type, duration_type>;

    Timestamp() : tp_{} {}
    explicit Timestamp(i64 microseconds_since_epoch)
        : tp_(duration_type(microseconds_since_epoch)) {}
    explicit Timestamp(time_point tp) : tp_(tp) {}

    static Timestamp from_seconds(f64 epoch_seconds) {
        return Timestamp(static_cast<i64>(epoch_seconds * 1'000'000));
    }

    static Timestamp from_millis(i64 millis) {
        return Timestamp(millis * 1000);
    }

    i64 microseconds() const {
        return tp_.time_since_epoch().count();
    }

    f64 seconds() const {
        return static_cast<f64>(microseconds()) / 1'000'000.0;
    }

    time_point time_point_value() const { return tp_; }

    auto operator<=>(const Timestamp&) const = default;

    f64 seconds_since(const Timestamp& other) const {
        return static_cast<f64>(microseconds() - other.microseconds()) / 1'000'000.0;
    }

    Timestamp operator+(duration_type d) const {
        return Timestamp(tp_ + d);
    }

    Timestamp operator-(duration_type d) const {
        return Timestamp(tp_ - d);
    }

    duration_type operator-(const Timestamp& other) const {
        return std::chrono::duration_cast<duration_type>(tp_ - other.tp_);
    }

    std::string to_string() const;

private:
    time_point tp_;
};

struct TimeRange {
    Timestamp start;
    Timestamp end;

    bool contains(const Timestamp& ts) const {
        return ts >= start && ts < end;
    }

    f64 duration_seconds() const {
        return end.seconds_since(start);
    }
};

} // namespace hfm
