#include "hfm/linalg/decompositions.hpp"
#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cmath>
#include <algorithm>

using lapack_int = __LAPACK_int;

namespace hfm {

Result<SVDResult> svd(const Matrix<f64>& A) {
    std::size_t m = A.rows();
    std::size_t n = A.cols();
    std::size_t mn = std::min(m, n);

    std::vector<f64> a_copy(A.data(), A.data() + m * n);
    // LAPACK expects column-major, transpose
    std::vector<f64> a_col(m * n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a_col[j * m + i] = a_copy[i * n + j];

    std::vector<f64> s(mn);
    std::vector<f64> u(m * mn);
    std::vector<f64> vt(mn * n);

    lapack_int M = static_cast<lapack_int>(m);
    lapack_int N = static_cast<lapack_int>(n);
    lapack_int lda = M;
    lapack_int ldu = M;
    lapack_int ldvt = static_cast<lapack_int>(mn);

    lapack_int lwork = -1;
    lapack_int info = 0;
    f64 work_query = 0.0;
    char jobu = 'S';
    char jobvt = 'S';

    dgesvd_(&jobu, &jobvt, &M, &N, a_col.data(), &lda,
            s.data(), u.data(), &ldu, vt.data(), &ldvt,
            &work_query, &lwork, &info);

    lwork = static_cast<lapack_int>(work_query);
    std::vector<f64> work(static_cast<std::size_t>(lwork));

    dgesvd_(&jobu, &jobvt, &M, &N, a_col.data(), &lda,
            s.data(), u.data(), &ldu, vt.data(), &ldvt,
            work.data(), &lwork, &info);

    if (info != 0) {
        return Status::error(ErrorCode::NotConverged, "svd: LAPACK dgesvd failed");
    }

    SVDResult result;
    result.S = Vector<f64>(std::vector<f64>(s.begin(), s.end()));

    result.U = Matrix<f64>(m, mn);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < mn; ++j)
            result.U(i, j) = u[j * m + i];

    result.Vt = Matrix<f64>(mn, n);
    for (std::size_t i = 0; i < mn; ++i)
        for (std::size_t j = 0; j < n; ++j)
            result.Vt(i, j) = vt[j * mn + i];

    return result;
}

f64 condition_number(const Matrix<f64>& A) {
    auto res = svd(A);
    if (!res) return 1e20;
    auto& sv = res.value().S;
    if (sv.size() == 0) return 1e20;
    f64 s_max = sv[0];
    f64 s_min = sv[sv.size() - 1];
    if (s_min < 1e-30) return 1e20;
    return s_max / s_min;
}

Result<EigenResult> eigen_symmetric(const Matrix<f64>& A) {
    std::size_t n = A.rows();
    if (n != A.cols()) {
        return Status::error(ErrorCode::DimensionMismatch,
                             "eigen_symmetric: matrix must be square");
    }

    // LAPACK expects column-major, symmetric → transpose is same
    std::vector<f64> a_col(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a_col[j * n + i] = A(i, j);

    std::vector<f64> w(n);

    lapack_int N = static_cast<lapack_int>(n);
    lapack_int lda = N;
    lapack_int lwork = -1;
    lapack_int info = 0;
    f64 work_query = 0.0;
    char jobz = 'V';
    char uplo = 'U';

    dsyev_(&jobz, &uplo, &N, a_col.data(), &lda, w.data(),
           &work_query, &lwork, &info);

    lwork = static_cast<lapack_int>(work_query);
    std::vector<f64> work(static_cast<std::size_t>(lwork));

    dsyev_(&jobz, &uplo, &N, a_col.data(), &lda, w.data(),
           work.data(), &lwork, &info);

    if (info != 0) {
        return Status::error(ErrorCode::NotConverged,
                             "eigen_symmetric: LAPACK dsyev failed");
    }

    EigenResult result;
    result.eigenvalues = Vector<f64>(std::vector<f64>(w.begin(), w.end()));

    result.eigenvectors = Matrix<f64>(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            result.eigenvectors(i, j) = a_col[j * n + i];

    return result;
}

Result<PCAResult> pca(const Matrix<f64>& X, std::size_t n_components) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = X.rows();
    std::size_t p = X.cols();

    if (n < 2 || p < 1) {
        return Status::error(ErrorCode::InvalidArgument, "pca: insufficient data");
    }

    if (n_components == 0 || n_components > p) n_components = p;

    // Center
    Vector<f64> means(p, 0.0);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i) means[j] += X(i, j);
        means[j] /= static_cast<f64>(n);
    }

    Matrix<f64> Xc(n, p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            Xc(i, j) = X(i, j) - means[j];

    // Covariance = X'X / (n-1)
    Matrix<f64> cov = matmul_AtB(Xc, Xc);
    f64 scale = 1.0 / static_cast<f64>(n - 1);
    for (std::size_t i = 0; i < p * p; ++i) cov.data()[i] *= scale;

    auto eig = eigen_symmetric(cov);
    if (!eig) {
        return Status::error(eig.status().code(), "pca: eigendecomposition failed");
    }

    auto& eigenvalues = eig.value().eigenvalues;
    auto& eigenvectors = eig.value().eigenvectors;

    PCAResult result;
    result.n_components = n_components;
    result.total_variance = 0.0;
    for (std::size_t j = 0; j < p; ++j) result.total_variance += eigenvalues[j];

    // eigenvalues in ascending order — reverse for PCA
    result.explained_variance.resize(n_components);
    result.explained_ratio.resize(n_components);
    result.loadings = Matrix<f64>(p, n_components);

    for (std::size_t c = 0; c < n_components; ++c) {
        std::size_t idx = p - 1 - c;
        result.explained_variance[c] = eigenvalues[idx];
        result.explained_ratio[c] = (result.total_variance > 0.0) ?
            eigenvalues[idx] / result.total_variance : 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            result.loadings(j, c) = eigenvectors(j, idx);
        }
    }

    // Scores = Xc × loadings
    result.components = matmul(Xc, result.loadings);

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

} // namespace hfm
