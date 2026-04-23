#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"

namespace hfm {

// ========== SVD ==========

struct SVDResult {
    Matrix<f64> U;       // m × min(m,n)
    Vector<f64> S;       // min(m,n) singular values
    Matrix<f64> Vt;      // min(m,n) × n
};

Result<SVDResult> svd(const Matrix<f64>& A);

f64 condition_number(const Matrix<f64>& A);

// ========== Eigenvalue decomposition (symmetric) ==========

struct EigenResult {
    Vector<f64> eigenvalues;    // n eigenvalues (ascending)
    Matrix<f64> eigenvectors;   // n × n (columns are eigenvectors)
};

Result<EigenResult> eigen_symmetric(const Matrix<f64>& A);

// ========== PCA ==========

struct PCAResult {
    Matrix<f64> components;        // n_obs × n_components (scores)
    Matrix<f64> loadings;          // n_vars × n_components
    Vector<f64> explained_variance;
    Vector<f64> explained_ratio;
    f64 total_variance = 0.0;
    std::size_t n_components = 0;
    f64 elapsed_ms = 0.0;
};

Result<PCAResult> pca(const Matrix<f64>& X, std::size_t n_components = 0);

} // namespace hfm
