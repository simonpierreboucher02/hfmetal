#pragma once

#include <vector>
#include <span>
#include <cstring>
#include <Accelerate/Accelerate.h>
#include "hfm/core/types.hpp"
#include "hfm/core/assert.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

template <typename T>
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(std::size_t rows, std::size_t cols, T fill = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    Matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        HFM_ASSERT(data_.size() == rows * cols, "matrix data size mismatch");
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }

    T& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    const T& operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    std::span<T> row(std::size_t r) {
        return std::span<T>(data_.data() + r * cols_, cols_);
    }
    std::span<const T> row(std::size_t r) const {
        return std::span<const T>(data_.data() + r * cols_, cols_);
    }

    Vector<T> col(std::size_t c) const {
        Vector<T> result(rows_);
        for (std::size_t r = 0; r < rows_; ++r) {
            result[r] = (*this)(r, c);
        }
        return result;
    }

    void set_col(std::size_t c, const Vector<T>& v) {
        HFM_ASSERT(v.size() == rows_, "set_col: dimension mismatch");
        for (std::size_t r = 0; r < rows_; ++r) {
            (*this)(r, c) = v[r];
        }
    }

    Matrix<T> transpose() const {
        Matrix<T> result(cols_, rows_);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                result(c, r) = (*this)(r, c);
            }
        }
        return result;
    }

    static Matrix<T> identity(std::size_t n) {
        Matrix<T> m(n, n, T{0});
        for (std::size_t i = 0; i < n; ++i) m(i, i) = T{1};
        return m;
    }

    static Matrix<T> zeros(std::size_t rows, std::size_t cols) {
        return Matrix<T>(rows, cols, T{0});
    }

    static Matrix<T> ones(std::size_t rows, std::size_t cols) {
        return Matrix<T>(rows, cols, T{1});
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};

// Accelerate-backed matrix multiply: C = A * B (row-major)
inline Matrix<f64> matmul(const Matrix<f64>& A, const Matrix<f64>& B) {
    HFM_ASSERT(A.cols() == B.rows(), "matmul: dimension mismatch");
    Matrix<f64> C(A.rows(), B.cols(), 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(A.rows()), static_cast<int>(B.cols()),
                static_cast<int>(A.cols()),
                1.0, A.data(), static_cast<int>(A.cols()),
                B.data(), static_cast<int>(B.cols()),
                0.0, C.data(), static_cast<int>(C.cols()));
    return C;
}

// C = A^T * B
inline Matrix<f64> matmul_AtB(const Matrix<f64>& A, const Matrix<f64>& B) {
    HFM_ASSERT(A.rows() == B.rows(), "matmul_AtB: dimension mismatch");
    Matrix<f64> C(A.cols(), B.cols(), 0.0);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(A.cols()), static_cast<int>(B.cols()),
                static_cast<int>(A.rows()),
                1.0, A.data(), static_cast<int>(A.cols()),
                B.data(), static_cast<int>(B.cols()),
                0.0, C.data(), static_cast<int>(C.cols()));
    return C;
}

// y = A * x
inline Vector<f64> matvec(const Matrix<f64>& A, const Vector<f64>& x) {
    HFM_ASSERT(A.cols() == x.size(), "matvec: dimension mismatch");
    Vector<f64> y(A.rows(), 0.0);
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(A.rows()), static_cast<int>(A.cols()),
                1.0, A.data(), static_cast<int>(A.cols()),
                x.data(), 1,
                0.0, y.data(), 1);
    return y;
}

} // namespace hfm
