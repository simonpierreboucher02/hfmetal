#pragma once

#include <vector>
#include <span>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <Accelerate/Accelerate.h>
#include "hfm/core/types.hpp"
#include "hfm/core/assert.hpp"

namespace hfm {

template <typename T>
class Vector {
public:
    Vector() = default;
    explicit Vector(std::size_t n, T fill = T{}) : data_(n, fill) {}
    Vector(std::vector<T> data) : data_(std::move(data)) {}
    Vector(std::initializer_list<T> init) : data_(init) {}

    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    std::span<T> span() { return data_; }
    std::span<const T> span() const { return data_; }

    const std::vector<T>& storage() const { return data_; }

    void resize(std::size_t n, T fill = T{}) { data_.resize(n, fill); }
    void reserve(std::size_t n) { data_.reserve(n); }
    void push_back(T val) { data_.push_back(val); }

    T dot(const Vector& other) const;
    T norm() const;
    T sum() const;
    T mean() const;

    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator*(T scalar) const;

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

private:
    std::vector<T> data_;
};

// Accelerate-backed specializations for f64
template <>
inline f64 Vector<f64>::dot(const Vector<f64>& other) const {
    HFM_ASSERT(size() == other.size(), "dot: dimension mismatch");
    return cblas_ddot(static_cast<int>(size()), data(), 1, other.data(), 1);
}

template <>
inline f64 Vector<f64>::norm() const {
    return cblas_dnrm2(static_cast<int>(size()), data(), 1);
}

template <>
inline f64 Vector<f64>::sum() const {
    f64 result = 0.0;
    vDSP_sveD(data(), 1, &result, static_cast<vDSP_Length>(size()));
    return result;
}

template <>
inline f64 Vector<f64>::mean() const {
    f64 result = 0.0;
    vDSP_meanvD(data(), 1, &result, static_cast<vDSP_Length>(size()));
    return result;
}

template <>
inline Vector<f64> Vector<f64>::operator+(const Vector<f64>& other) const {
    HFM_ASSERT(size() == other.size(), "add: dimension mismatch");
    Vector<f64> result(size());
    vDSP_vaddD(data(), 1, other.data(), 1, result.data(), 1,
               static_cast<vDSP_Length>(size()));
    return result;
}

template <>
inline Vector<f64> Vector<f64>::operator-(const Vector<f64>& other) const {
    HFM_ASSERT(size() == other.size(), "sub: dimension mismatch");
    Vector<f64> result(size());
    vDSP_vsubD(other.data(), 1, data(), 1, result.data(), 1,
               static_cast<vDSP_Length>(size()));
    return result;
}

template <>
inline Vector<f64> Vector<f64>::operator*(f64 scalar) const {
    Vector<f64> result(size());
    vDSP_vsmulD(data(), 1, &scalar, result.data(), 1,
                static_cast<vDSP_Length>(size()));
    return result;
}

// f32 specializations
template <>
inline f32 Vector<f32>::dot(const Vector<f32>& other) const {
    HFM_ASSERT(size() == other.size(), "dot: dimension mismatch");
    return cblas_sdot(static_cast<int>(size()), data(), 1, other.data(), 1);
}

template <>
inline f32 Vector<f32>::norm() const {
    return cblas_snrm2(static_cast<int>(size()), data(), 1);
}

template <>
inline f32 Vector<f32>::sum() const {
    f32 result = 0.0f;
    vDSP_sve(data(), 1, &result, static_cast<vDSP_Length>(size()));
    return result;
}

template <>
inline f32 Vector<f32>::mean() const {
    f32 result = 0.0f;
    vDSP_meanv(data(), 1, &result, static_cast<vDSP_Length>(size()));
    return result;
}

} // namespace hfm
