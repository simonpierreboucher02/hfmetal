#include "hfm/linalg/solver.hpp"
#include <cstring>
#include <algorithm>

// Accelerate new LAPACK uses __LAPACK_int
using lapack_int = __LAPACK_int;

namespace hfm {

Result<Vector<f64>> solve_least_squares(const Matrix<f64>& A, const Vector<f64>& b) {
    if (A.rows() != b.size()) {
        return Status::error(ErrorCode::DimensionMismatch, "A.rows != b.size");
    }
    if (A.rows() < A.cols()) {
        return Status::error(ErrorCode::InvalidArgument, "underdetermined system");
    }

    auto m = static_cast<lapack_int>(A.rows());
    auto n = static_cast<lapack_int>(A.cols());
    lapack_int nrhs = 1;
    lapack_int lda = n;
    lapack_int ldb = 1;

    // LAPACK dgels works with column-major, so transpose A
    std::vector<f64> At(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (std::size_t r = 0; r < A.rows(); ++r) {
        for (std::size_t c = 0; c < A.cols(); ++c) {
            At[c * static_cast<std::size_t>(m) + r] = A(r, c);
        }
    }
    lda = m;

    std::vector<f64> rhs(static_cast<std::size_t>(m));
    std::copy(b.data(), b.data() + b.size(), rhs.begin());
    ldb = m;

    lapack_int lwork = -1;
    f64 work_query = 0.0;
    lapack_int info = 0;
    char trans = 'N';

    dgels_(&trans, &m, &n, &nrhs, At.data(), &lda,
           rhs.data(), &ldb, &work_query, &lwork, &info);

    lwork = static_cast<lapack_int>(work_query);
    std::vector<f64> work(static_cast<std::size_t>(lwork));

    dgels_(&trans, &m, &n, &nrhs, At.data(), &lda,
           rhs.data(), &ldb, work.data(), &lwork, &info);

    if (info != 0) {
        return Status::error(ErrorCode::SingularMatrix,
                             "dgels failed with info=" + std::to_string(info));
    }

    Vector<f64> result(static_cast<std::size_t>(n));
    std::copy(rhs.begin(), rhs.begin() + n, result.data());
    return result;
}

Result<Matrix<f64>> cholesky(const Matrix<f64>& A) {
    if (A.rows() != A.cols()) {
        return Status::error(ErrorCode::DimensionMismatch, "cholesky: not square");
    }

    auto n = static_cast<lapack_int>(A.rows());
    std::vector<f64> L(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));

    // Copy to column-major lower triangle
    for (std::size_t r = 0; r < A.rows(); ++r) {
        for (std::size_t c = 0; c <= r; ++c) {
            L[c * static_cast<std::size_t>(n) + r] = A(r, c);
        }
    }

    char uplo = 'L';
    lapack_int info = 0;
    dpotrf_(&uplo, &n, L.data(), &n, &info);

    if (info != 0) {
        return Status::error(ErrorCode::SingularMatrix,
                             "cholesky failed, not SPD, info=" + std::to_string(info));
    }

    Matrix<f64> result(static_cast<std::size_t>(n), static_cast<std::size_t>(n), 0.0);
    for (std::size_t r = 0; r < result.rows(); ++r) {
        for (std::size_t c = 0; c <= r; ++c) {
            result(r, c) = L[c * static_cast<std::size_t>(n) + r];
        }
    }
    return result;
}

Result<Matrix<f64>> invert_spd(const Matrix<f64>& A) {
    if (A.rows() != A.cols()) {
        return Status::error(ErrorCode::DimensionMismatch, "invert_spd: not square");
    }

    auto n = static_cast<lapack_int>(A.rows());
    std::vector<f64> colmaj(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));

    for (std::size_t r = 0; r < A.rows(); ++r) {
        for (std::size_t c = 0; c < A.cols(); ++c) {
            colmaj[c * static_cast<std::size_t>(n) + r] = A(r, c);
        }
    }

    char uplo = 'L';
    lapack_int info = 0;

    dpotrf_(&uplo, &n, colmaj.data(), &n, &info);
    if (info != 0) {
        return Status::error(ErrorCode::SingularMatrix, "invert_spd: cholesky failed");
    }

    dpotri_(&uplo, &n, colmaj.data(), &n, &info);
    if (info != 0) {
        return Status::error(ErrorCode::SingularMatrix, "invert_spd: inversion failed");
    }

    Matrix<f64> result(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    for (std::size_t r = 0; r < result.rows(); ++r) {
        for (std::size_t c = 0; c < result.cols(); ++c) {
            if (c <= r) {
                result(r, c) = colmaj[c * static_cast<std::size_t>(n) + r];
            } else {
                result(r, c) = colmaj[r * static_cast<std::size_t>(n) + c];
            }
        }
    }
    return result;
}

Result<Vector<f64>> solve_spd(const Matrix<f64>& A, const Vector<f64>& b) {
    if (A.rows() != A.cols() || A.rows() != b.size()) {
        return Status::error(ErrorCode::DimensionMismatch, "solve_spd: dimension mismatch");
    }

    auto n = static_cast<lapack_int>(A.rows());
    lapack_int nrhs = 1;

    std::vector<f64> colmaj(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (std::size_t r = 0; r < A.rows(); ++r) {
        for (std::size_t c = 0; c < A.cols(); ++c) {
            colmaj[c * static_cast<std::size_t>(n) + r] = A(r, c);
        }
    }

    std::vector<f64> x(b.data(), b.data() + b.size());

    char uplo = 'L';
    lapack_int info = 0;
    dposv_(&uplo, &n, &nrhs, colmaj.data(), &n, x.data(), &n, &info);

    if (info != 0) {
        return Status::error(ErrorCode::SingularMatrix, "solve_spd failed");
    }

    return Vector<f64>(std::move(x));
}

} // namespace hfm
