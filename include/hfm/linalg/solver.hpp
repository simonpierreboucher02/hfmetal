#pragma once

#include <Accelerate/Accelerate.h>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

struct QRDecomposition {
    Matrix<f64> Q;
    Matrix<f64> R;
};

Result<Vector<f64>> solve_least_squares(const Matrix<f64>& A, const Vector<f64>& b);
Result<QRDecomposition> qr_decompose(const Matrix<f64>& A);
Result<Matrix<f64>> cholesky(const Matrix<f64>& A);
Result<Vector<f64>> solve_spd(const Matrix<f64>& A, const Vector<f64>& b);
Result<Matrix<f64>> invert_spd(const Matrix<f64>& A);

} // namespace hfm
