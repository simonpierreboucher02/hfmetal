#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/hf/returns.hpp"
#include "hfm/hf/realized_measures.hpp"
#include "hfm/hf/event_study.hpp"
#include "hfm/estimators/ols.hpp"
#include "hfm/estimators/rolling_ols.hpp"
#include "hfm/estimators/batched_ols.hpp"
#include "hfm/estimators/gls.hpp"
#include "hfm/covariance/covariance.hpp"
#include "hfm/timeseries/ar.hpp"
#include "hfm/timeseries/var.hpp"
#include "hfm/timeseries/har.hpp"
#include "hfm/data/series.hpp"
#include "hfm/panel/fixed_effects.hpp"
#include "hfm/models/fama_macbeth.hpp"
#include "hfm/models/garch.hpp"
#include "hfm/models/logit_probit.hpp"
#include "hfm/estimators/iv.hpp"
#include "hfm/timeseries/local_projections.hpp"
#include "hfm/simulation/bootstrap.hpp"
#include "hfm/simulation/mcmc.hpp"

namespace py = pybind11;
using namespace hfm;

namespace {

Vector<f64> numpy_to_vector(py::array_t<f64> arr) {
    auto buf = arr.request();
    auto* ptr = static_cast<f64*>(buf.ptr);
    std::size_t n = static_cast<std::size_t>(buf.size);
    Vector<f64> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = ptr[i];
    return v;
}

Vector<i64> numpy_to_ivector(py::array_t<int64_t> arr) {
    auto buf = arr.request();
    auto* ptr = static_cast<int64_t*>(buf.ptr);
    std::size_t n = static_cast<std::size_t>(buf.size);
    Vector<i64> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = ptr[i];
    return v;
}

Matrix<f64> numpy_to_matrix(py::array_t<f64, py::array::c_style> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
    auto rows = static_cast<std::size_t>(buf.shape[0]);
    auto cols = static_cast<std::size_t>(buf.shape[1]);
    auto* ptr = static_cast<f64*>(buf.ptr);
    Matrix<f64> m(rows, cols);
    for (std::size_t i = 0; i < rows * cols; ++i) m.data()[i] = ptr[i];
    return m;
}

py::array_t<f64> vector_to_numpy(const Vector<f64>& v) {
    py::array_t<f64> result(static_cast<py::ssize_t>(v.size()));
    auto buf = result.request();
    auto* ptr = static_cast<f64*>(buf.ptr);
    for (std::size_t i = 0; i < v.size(); ++i) ptr[i] = v[i];
    return result;
}

py::array_t<f64> matrix_to_numpy(const Matrix<f64>& m) {
    py::array_t<f64> result({static_cast<py::ssize_t>(m.rows()),
                              static_cast<py::ssize_t>(m.cols())});
    auto buf = result.request();
    auto* ptr = static_cast<f64*>(buf.ptr);
    for (std::size_t i = 0; i < m.rows() * m.cols(); ++i) ptr[i] = m.data()[i];
    return result;
}

Series<f64> numpy_to_series(py::array_t<f64> arr) {
    auto buf = arr.request();
    auto* ptr = static_cast<f64*>(buf.ptr);
    std::size_t n = static_cast<std::size_t>(buf.size);
    std::vector<f64> v(ptr, ptr + n);
    return Series<f64>(v);
}

CovarianceType parse_covariance(const std::string& s) {
    if (s == "white") return CovarianceType::White;
    if (s == "newey_west" || s == "hac") return CovarianceType::NeweyWest;
    if (s == "clustered") return CovarianceType::ClusteredOneWay;
    return CovarianceType::Classical;
}

} // namespace

PYBIND11_MODULE(_hfmetal, m) {
    m.doc() = "HFMetal: High-frequency econometrics engine for Apple Silicon";

    // ========== Enums ==========
    py::enum_<Backend>(m, "Backend")
        .value("Auto", Backend::Auto)
        .value("CPU", Backend::CPU)
        .value("Accelerate", Backend::Accelerate)
        .value("Metal", Backend::Metal);

    py::enum_<CovarianceType>(m, "CovarianceType")
        .value("Classical", CovarianceType::Classical)
        .value("White", CovarianceType::White)
        .value("NeweyWest", CovarianceType::NeweyWest)
        .value("ClusteredOneWay", CovarianceType::ClusteredOneWay)
        .value("ClusteredTwoWay", CovarianceType::ClusteredTwoWay);

    // ========== OLSResult ==========
    py::class_<OLSResult>(m, "OLSResult")
        .def_property_readonly("coefficients", [](const OLSResult& r) { return vector_to_numpy(r.coefficients()); })
        .def_property_readonly("residuals", [](const OLSResult& r) { return vector_to_numpy(r.residuals()); })
        .def_property_readonly("fitted_values", [](const OLSResult& r) { return vector_to_numpy(r.fitted_values()); })
        .def_property_readonly("covariance_matrix", [](const OLSResult& r) { return matrix_to_numpy(r.covariance_matrix()); })
        .def_property_readonly("std_errors", [](const OLSResult& r) { return vector_to_numpy(r.std_errors()); })
        .def_property_readonly("t_stats", [](const OLSResult& r) { return vector_to_numpy(r.t_stats()); })
        .def_property_readonly("p_values", [](const OLSResult& r) { return vector_to_numpy(r.p_values()); })
        .def_property_readonly("n_obs", &OLSResult::n_obs)
        .def_property_readonly("n_regressors", &OLSResult::n_regressors)
        .def_property_readonly("r_squared", &OLSResult::r_squared)
        .def_property_readonly("adj_r_squared", &OLSResult::adj_r_squared)
        .def_property_readonly("sigma", &OLSResult::sigma)
        .def_property_readonly("elapsed_ms", &OLSResult::elapsed_ms)
        .def("summary", &OLSResult::summary);

    // ========== RollingOLSResult ==========
    py::class_<RollingOLSResult>(m, "RollingOLSResult")
        .def_property_readonly("betas", [](const RollingOLSResult& r) { return matrix_to_numpy(r.betas()); })
        .def_property_readonly("std_errors", [](const RollingOLSResult& r) { return matrix_to_numpy(r.std_errors()); })
        .def_property_readonly("r_squared", [](const RollingOLSResult& r) { return vector_to_numpy(r.r_squared()); })
        .def_property_readonly("n_windows", &RollingOLSResult::n_windows)
        .def_property_readonly("elapsed_ms", &RollingOLSResult::elapsed_ms);

    // ========== GLSResult ==========
    py::class_<GLSResult>(m, "GLSResult")
        .def_property_readonly("coefficients", [](const GLSResult& r) { return vector_to_numpy(r.coefficients()); })
        .def_property_readonly("residuals", [](const GLSResult& r) { return vector_to_numpy(r.residuals()); })
        .def_property_readonly("std_errors", [](const GLSResult& r) { return vector_to_numpy(r.std_errors()); })
        .def_property_readonly("t_stats", [](const GLSResult& r) { return vector_to_numpy(r.t_stats()); })
        .def_property_readonly("p_values", [](const GLSResult& r) { return vector_to_numpy(r.p_values()); })
        .def_property_readonly("r_squared", &GLSResult::r_squared)
        .def_property_readonly("elapsed_ms", &GLSResult::elapsed_ms);

    // ========== PanelResult ==========
    py::class_<PanelResult>(m, "PanelResult")
        .def_property_readonly("coefficients", [](const PanelResult& r) { return vector_to_numpy(r.coefficients()); })
        .def_property_readonly("std_errors", [](const PanelResult& r) { return vector_to_numpy(r.std_errors()); })
        .def_property_readonly("t_stats", [](const PanelResult& r) { return vector_to_numpy(r.t_stats()); })
        .def_property_readonly("p_values", [](const PanelResult& r) { return vector_to_numpy(r.p_values()); })
        .def_property_readonly("r_squared_within", &PanelResult::r_squared_within)
        .def_property_readonly("n_obs", &PanelResult::n_obs)
        .def_property_readonly("n_groups", &PanelResult::n_groups)
        .def("summary", &PanelResult::summary);

    // ========== FamaMacBethResult ==========
    py::class_<FamaMacBethResult>(m, "FamaMacBethResult")
        .def_property_readonly("gamma", [](const FamaMacBethResult& r) { return vector_to_numpy(r.gamma); })
        .def_property_readonly("std_errors", [](const FamaMacBethResult& r) { return vector_to_numpy(r.std_errors); })
        .def_property_readonly("t_stats", [](const FamaMacBethResult& r) { return vector_to_numpy(r.t_stats); })
        .def_property_readonly("n_periods", [](const FamaMacBethResult& r) { return r.n_periods; })
        .def_property_readonly("n_factors", [](const FamaMacBethResult& r) { return r.n_factors; });

    // ========== ARResult ==========
    py::class_<ARResult>(m, "ARResult")
        .def_property_readonly("coefficients", [](const ARResult& r) { return vector_to_numpy(r.coefficients); })
        .def_property_readonly("std_errors", [](const ARResult& r) { return vector_to_numpy(r.std_errors); })
        .def_property_readonly("t_stats", [](const ARResult& r) { return vector_to_numpy(r.t_stats); })
        .def_property_readonly("r_squared", [](const ARResult& r) { return r.r_squared; })
        .def_property_readonly("aic", [](const ARResult& r) { return r.aic; })
        .def_property_readonly("bic", [](const ARResult& r) { return r.bic; })
        .def_property_readonly("n_obs", [](const ARResult& r) { return r.n_obs; });

    // ========== VARResult ==========
    py::class_<VARResult>(m, "VARResult")
        .def_property_readonly("coefficients", [](const VARResult& r) { return matrix_to_numpy(r.coefficients); })
        .def_property_readonly("residuals", [](const VARResult& r) { return matrix_to_numpy(r.residuals); })
        .def_property_readonly("sigma_u", [](const VARResult& r) { return matrix_to_numpy(r.sigma_u); })
        .def_property_readonly("aic", [](const VARResult& r) { return r.aic; })
        .def_property_readonly("bic", [](const VARResult& r) { return r.bic; })
        .def_property_readonly("n_vars", [](const VARResult& r) { return r.n_vars; })
        .def_property_readonly("n_obs", [](const VARResult& r) { return r.n_obs; });

    // ========== RealizedMeasuresResult ==========
    py::class_<RealizedMeasuresResult>(m, "RealizedMeasuresResult")
        .def_readonly("realized_variance", &RealizedMeasuresResult::realized_variance)
        .def_readonly("realized_volatility", &RealizedMeasuresResult::realized_volatility)
        .def_readonly("bipower_variation", &RealizedMeasuresResult::bipower_variation)
        .def_readonly("jump_statistic", &RealizedMeasuresResult::jump_statistic)
        .def_readonly("n_obs", &RealizedMeasuresResult::n_obs);

    // ========== EventStudyResult ==========
    py::class_<EventStudyResult>(m, "EventStudyResult")
        .def_property_readonly("abnormal_returns", [](const EventStudyResult& r) { return matrix_to_numpy(r.abnormal_returns); })
        .def_property_readonly("cumulative_ar", [](const EventStudyResult& r) { return matrix_to_numpy(r.cumulative_ar); })
        .def_property_readonly("mean_car", [](const EventStudyResult& r) { return vector_to_numpy(r.mean_car); })
        .def_property_readonly("n_events", [](const EventStudyResult& r) { return r.n_events; })
        .def_property_readonly("elapsed_ms", [](const EventStudyResult& r) { return r.elapsed_ms; });

    // ========== BootstrapResult ==========
    py::class_<BootstrapResult>(m, "BootstrapResult")
        .def_property_readonly("estimate", [](const BootstrapResult& r) { return vector_to_numpy(r.estimate); })
        .def_property_readonly("mean", [](const BootstrapResult& r) { return vector_to_numpy(r.mean); })
        .def_property_readonly("std_error", [](const BootstrapResult& r) { return vector_to_numpy(r.std_error); })
        .def_property_readonly("ci_lower", [](const BootstrapResult& r) { return vector_to_numpy(r.ci_lower); })
        .def_property_readonly("ci_upper", [](const BootstrapResult& r) { return vector_to_numpy(r.ci_upper); })
        .def_property_readonly("p_value", [](const BootstrapResult& r) { return vector_to_numpy(r.p_value); })
        .def_property_readonly("n_bootstrap", [](const BootstrapResult& r) { return r.n_bootstrap; });

    // ========== Free functions ==========

    // Returns
    m.def("log_returns", [](py::array_t<f64> prices) {
        auto s = numpy_to_series(prices);
        auto r = log_returns(s);
        py::array_t<f64> out(static_cast<py::ssize_t>(r.size()));
        auto buf = out.request();
        auto* ptr = static_cast<f64*>(buf.ptr);
        for (std::size_t i = 0; i < r.size(); ++i) ptr[i] = r[i];
        return out;
    }, py::arg("prices"), "Compute log returns from prices");

    m.def("simple_returns", [](py::array_t<f64> prices) {
        auto s = numpy_to_series(prices);
        auto r = simple_returns(s);
        py::array_t<f64> out(static_cast<py::ssize_t>(r.size()));
        auto buf = out.request();
        auto* ptr = static_cast<f64*>(buf.ptr);
        for (std::size_t i = 0; i < r.size(); ++i) ptr[i] = r[i];
        return out;
    }, py::arg("prices"), "Compute simple returns from prices");

    // Realized measures
    m.def("realized_variance", [](py::array_t<f64> returns) {
        auto s = numpy_to_series(returns);
        return realized_variance(s);
    }, py::arg("returns"), "Compute realized variance");

    m.def("realized_volatility", [](py::array_t<f64> returns) {
        auto s = numpy_to_series(returns);
        return realized_volatility(s);
    }, py::arg("returns"), "Compute realized volatility");

    m.def("bipower_variation", [](py::array_t<f64> returns) {
        auto s = numpy_to_series(returns);
        return bipower_variation(s);
    }, py::arg("returns"), "Compute bipower variation");

    m.def("compute_realized_measures", [](py::array_t<f64> returns) {
        auto s = numpy_to_series(returns);
        return compute_realized_measures(s);
    }, py::arg("returns"), "Compute all realized measures");

    // OLS
    m.def("ols", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                     const std::string& covariance, int hac_lag, bool add_intercept) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        OLSOptions opts;
        opts.covariance = parse_covariance(covariance);
        opts.hac_lag = hac_lag;
        opts.add_intercept = add_intercept;
        auto result = ols(y, X, opts);
        if (!result) throw std::runtime_error("OLS failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("covariance") = "classical",
       py::arg("hac_lag") = -1, py::arg("add_intercept") = false,
       "Ordinary least squares regression");

    // Rolling OLS
    m.def("rolling_ols", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                             std::size_t window, std::size_t step, const std::string& covariance) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        RollingOptions opts;
        opts.window = window;
        opts.step = step;
        opts.covariance = parse_covariance(covariance);
        auto result = rolling_ols(y, X, opts);
        if (!result) throw std::runtime_error("Rolling OLS failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("window") = 60,
       py::arg("step") = 1, py::arg("covariance") = "classical",
       "Rolling window OLS regression");

    // GLS
    m.def("gls", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                     py::array_t<f64, py::array::c_style> Omega_arr, bool add_intercept) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        auto Omega = numpy_to_matrix(Omega_arr);
        GLSOptions opts;
        opts.add_intercept = add_intercept;
        auto result = gls(y, X, Omega, opts);
        if (!result) throw std::runtime_error("GLS failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("Omega"), py::arg("add_intercept") = false,
       "Generalized least squares");

    m.def("fgls", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                      bool add_intercept) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        GLSOptions opts;
        opts.add_intercept = add_intercept;
        auto result = fgls(y, X, opts);
        if (!result) throw std::runtime_error("FGLS failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("add_intercept") = false,
       "Feasible generalized least squares");

    // AR
    m.def("ar", [](py::array_t<f64> y_arr, std::size_t p, const std::string& covariance) {
        auto y = numpy_to_vector(y_arr);
        AROptions opts;
        opts.p = p;
        opts.covariance = parse_covariance(covariance);
        auto result = ar(y, opts);
        if (!result) throw std::runtime_error("AR failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("p") = 1, py::arg("covariance") = "newey_west",
       "Autoregressive model");

    // VAR
    m.def("var", [](py::array_t<f64, py::array::c_style> Y_arr, std::size_t p,
                     const std::string& covariance) {
        auto Y = numpy_to_matrix(Y_arr);
        VAROptions opts;
        opts.p = p;
        opts.covariance = parse_covariance(covariance);
        auto result = var(Y, opts);
        if (!result) throw std::runtime_error("VAR failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("Y"), py::arg("p") = 1, py::arg("covariance") = "newey_west",
       "Vector autoregressive model");

    // Event study
    m.def("event_study", [](py::array_t<f64> returns_arr,
                             py::array_t<int64_t> event_indices_arr,
                             int64_t left, int64_t right) {
        auto returns = numpy_to_series(returns_arr);
        auto idx_buf = event_indices_arr.request();
        auto* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        std::size_t n_events = static_cast<std::size_t>(idx_buf.size);

        std::vector<std::size_t> event_indices(n_events);
        for (std::size_t i = 0; i < n_events; ++i)
            event_indices[i] = static_cast<std::size_t>(idx_ptr[i]);

        EventStudyOptions opts;
        opts.window.left = left;
        opts.window.right = right;

        auto result = event_study(returns, event_indices, opts);
        if (!result) throw std::runtime_error("Event study failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("returns"), py::arg("event_indices"),
       py::arg("left") = -60, py::arg("right") = 60,
       "Event study analysis");

    // Fixed effects
    m.def("fixed_effects", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                               py::array_t<int64_t> entity_ids_arr, py::array_t<int64_t> time_ids_arr,
                               bool cluster_entity, bool cluster_time) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        auto entity_ids = numpy_to_ivector(entity_ids_arr);
        auto time_ids = numpy_to_ivector(time_ids_arr);
        PanelOptions opts;
        opts.cluster_entity = cluster_entity;
        opts.cluster_time = cluster_time;
        auto result = fixed_effects(y, X, entity_ids, time_ids, opts);
        if (!result) throw std::runtime_error("Fixed effects failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("entity_ids"), py::arg("time_ids"),
       py::arg("cluster_entity") = true, py::arg("cluster_time") = false,
       "Panel fixed effects estimator");

    // Fama-MacBeth
    m.def("fama_macbeth", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                              py::array_t<int64_t> time_ids_arr, bool newey_west) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        auto time_ids = numpy_to_ivector(time_ids_arr);
        FamaMacBethOptions opts;
        opts.newey_west_correction = newey_west;
        auto result = fama_macbeth(y, X, time_ids, opts);
        if (!result) throw std::runtime_error("Fama-MacBeth failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("time_ids"), py::arg("newey_west") = true,
       "Fama-MacBeth cross-sectional regression");

    // Bootstrap
    m.def("bootstrap", [](py::array_t<f64> data_arr, py::function stat_fn,
                           std::size_t n_bootstrap, std::size_t block_size,
                           const std::string& type, uint64_t seed) {
        auto data = numpy_to_vector(data_arr);
        StatisticFn cpp_fn = [&stat_fn](const Vector<f64>& d) -> Vector<f64> {
            py::array_t<f64> arr(static_cast<py::ssize_t>(d.size()));
            auto buf = arr.request();
            auto* ptr = static_cast<f64*>(buf.ptr);
            for (std::size_t i = 0; i < d.size(); ++i) ptr[i] = d[i];
            py::object result = stat_fn(arr);
            auto res_arr = result.cast<py::array_t<f64>>();
            return numpy_to_vector(res_arr);
        };
        BootstrapOptions opts;
        opts.n_bootstrap = n_bootstrap;
        opts.block_size = block_size;
        opts.seed = seed;
        if (type == "block") opts.type = BootstrapType::Block;
        else if (type == "circular") opts.type = BootstrapType::Circular;
        else opts.type = BootstrapType::IID;
        auto result = bootstrap(data, cpp_fn, opts);
        if (!result) throw std::runtime_error("Bootstrap failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("data"), py::arg("statistic"), py::arg("n_bootstrap") = 1000,
       py::arg("block_size") = 1, py::arg("type") = "iid", py::arg("seed") = 42,
       "Bootstrap inference");

    // HAR-RV
    m.def("har_rv", [](py::array_t<f64> rv_arr) {
        auto s = numpy_to_series(rv_arr);
        auto result = har_rv(s);
        if (!result) throw std::runtime_error("HAR-RV failed: " + result.status().message());
        auto& res = result.value();
        py::dict d;
        d["alpha"] = res.alpha;
        d["beta_d"] = res.beta_d;
        d["beta_w"] = res.beta_w;
        d["beta_m"] = res.beta_m;
        d["r_squared"] = res.r_squared;
        d["n_obs"] = res.n_obs;
        d["std_errors"] = vector_to_numpy(res.std_errors);
        d["t_stats"] = vector_to_numpy(res.t_stats);
        return d;
    }, py::arg("daily_rv"), "HAR-RV model for realized volatility");

    // ========== GARCH ==========
    py::class_<GARCHResult>(m, "GARCHResult")
        .def_readonly("omega", &GARCHResult::omega)
        .def_readonly("alpha", &GARCHResult::alpha)
        .def_readonly("beta", &GARCHResult::beta)
        .def_readonly("persistence", &GARCHResult::persistence)
        .def_readonly("unconditional_var", &GARCHResult::unconditional_var)
        .def_property_readonly("conditional_var", [](const GARCHResult& r) { return vector_to_numpy(r.conditional_var); })
        .def_property_readonly("std_residuals", [](const GARCHResult& r) { return vector_to_numpy(r.std_residuals); })
        .def_readonly("log_likelihood", &GARCHResult::log_likelihood)
        .def_readonly("aic", &GARCHResult::aic)
        .def_readonly("bic", &GARCHResult::bic)
        .def_property_readonly("std_errors", [](const GARCHResult& r) { return vector_to_numpy(r.std_errors); })
        .def_readonly("n_obs", &GARCHResult::n_obs)
        .def_readonly("n_iter", &GARCHResult::n_iter)
        .def_readonly("converged", &GARCHResult::converged)
        .def_readonly("elapsed_ms", &GARCHResult::elapsed_ms);

    m.def("garch", [](py::array_t<f64> returns_arr, std::size_t max_iter, f64 tol) {
        auto r = numpy_to_vector(returns_arr);
        GARCHOptions opts;
        opts.max_iter = max_iter;
        opts.tol = tol;
        auto result = garch(r, opts);
        if (!result) throw std::runtime_error("GARCH failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("returns"), py::arg("max_iter") = 500, py::arg("tol") = 1e-8,
       "GARCH(1,1) model estimation");

    // ========== Logit/Probit ==========
    py::class_<BinaryModelResult>(m, "BinaryModelResult")
        .def_property_readonly("coefficients", [](const BinaryModelResult& r) { return vector_to_numpy(r.coefficients); })
        .def_property_readonly("std_errors", [](const BinaryModelResult& r) { return vector_to_numpy(r.std_errors); })
        .def_property_readonly("t_stats", [](const BinaryModelResult& r) { return vector_to_numpy(r.t_stats); })
        .def_property_readonly("p_values", [](const BinaryModelResult& r) { return vector_to_numpy(r.p_values); })
        .def_property_readonly("marginal_effects", [](const BinaryModelResult& r) { return vector_to_numpy(r.marginal_effects); })
        .def_property_readonly("predicted_prob", [](const BinaryModelResult& r) { return vector_to_numpy(r.predicted_prob); })
        .def_readonly("log_likelihood", &BinaryModelResult::log_likelihood)
        .def_readonly("pseudo_r_squared", &BinaryModelResult::pseudo_r_squared)
        .def_readonly("aic", &BinaryModelResult::aic)
        .def_readonly("bic", &BinaryModelResult::bic)
        .def_readonly("n_obs", &BinaryModelResult::n_obs)
        .def_readonly("converged", &BinaryModelResult::converged);

    m.def("logit", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                       bool add_intercept) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        BinaryModelOptions opts;
        opts.add_intercept = add_intercept;
        auto result = logit(y, X, opts);
        if (!result) throw std::runtime_error("Logit failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("add_intercept") = true,
       "Logistic regression");

    m.def("probit", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                        bool add_intercept) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        BinaryModelOptions opts;
        opts.type = BinaryModelType::Probit;
        opts.add_intercept = add_intercept;
        auto result = probit(y, X, opts);
        if (!result) throw std::runtime_error("Probit failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("add_intercept") = true,
       "Probit regression");

    // ========== IV/2SLS ==========
    py::class_<IVResult>(m, "IVResult")
        .def_property_readonly("coefficients", [](const IVResult& r) { return vector_to_numpy(r.coefficients()); })
        .def_property_readonly("residuals", [](const IVResult& r) { return vector_to_numpy(r.residuals()); })
        .def_property_readonly("std_errors", [](const IVResult& r) { return vector_to_numpy(r.std_errors()); })
        .def_property_readonly("t_stats", [](const IVResult& r) { return vector_to_numpy(r.t_stats()); })
        .def_property_readonly("p_values", [](const IVResult& r) { return vector_to_numpy(r.p_values()); })
        .def_property_readonly("r_squared", &IVResult::r_squared)
        .def_property_readonly("n_obs", &IVResult::n_obs)
        .def_property_readonly("n_instruments", &IVResult::n_instruments)
        .def_property_readonly("sargan_stat", &IVResult::sargan_stat)
        .def_property_readonly("sargan_pvalue", &IVResult::sargan_pvalue)
        .def_property_readonly("overidentified", &IVResult::overidentified)
        .def_property_readonly("elapsed_ms", &IVResult::elapsed_ms);

    m.def("iv_2sls", [](py::array_t<f64> y_arr, py::array_t<f64, py::array::c_style> X_arr,
                         py::array_t<f64, py::array::c_style> Z_arr,
                         const std::string& covariance, bool add_intercept) {
        auto y = numpy_to_vector(y_arr);
        auto X = numpy_to_matrix(X_arr);
        auto Z = numpy_to_matrix(Z_arr);
        IVOptions opts;
        opts.covariance = parse_covariance(covariance);
        opts.add_intercept = add_intercept;
        auto result = iv_2sls(y, X, Z, opts);
        if (!result) throw std::runtime_error("IV/2SLS failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("X"), py::arg("Z"),
       py::arg("covariance") = "classical", py::arg("add_intercept") = false,
       "Two-stage least squares (IV) estimation");

    // ========== Local Projections ==========
    py::class_<LPResult>(m, "LPResult")
        .def_property_readonly("irf", [](const LPResult& r) { return vector_to_numpy(r.irf); })
        .def_property_readonly("irf_se", [](const LPResult& r) { return vector_to_numpy(r.irf_se); })
        .def_property_readonly("irf_lower", [](const LPResult& r) { return vector_to_numpy(r.irf_lower); })
        .def_property_readonly("irf_upper", [](const LPResult& r) { return vector_to_numpy(r.irf_upper); })
        .def_property_readonly("cumulative_irf", [](const LPResult& r) { return vector_to_numpy(r.cumulative_irf); })
        .def_readonly("max_horizon", &LPResult::max_horizon)
        .def_readonly("n_obs", &LPResult::n_obs)
        .def_readonly("elapsed_ms", &LPResult::elapsed_ms);

    // ========== MCMCChain ==========
    py::class_<MCMCChain>(m, "MCMCChain")
        .def_property_readonly("samples", [](const MCMCChain& c) { return matrix_to_numpy(c.samples); })
        .def_property_readonly("log_posterior", [](const MCMCChain& c) { return vector_to_numpy(c.log_posterior); })
        .def_readonly("n_dim", &MCMCChain::n_dim)
        .def_readonly("n_kept", &MCMCChain::n_kept)
        .def_readonly("acceptance_rate", &MCMCChain::acceptance_rate)
        .def_property_readonly("mean", [](const MCMCChain& c) { return vector_to_numpy(c.mean); })
        .def_property_readonly("std_dev", [](const MCMCChain& c) { return vector_to_numpy(c.std_dev); })
        .def_property_readonly("median", [](const MCMCChain& c) { return vector_to_numpy(c.median); })
        .def_property_readonly("ci_lower", [](const MCMCChain& c) { return vector_to_numpy(c.ci_lower); })
        .def_property_readonly("ci_upper", [](const MCMCChain& c) { return vector_to_numpy(c.ci_upper); })
        .def_property_readonly("posterior_cov", [](const MCMCChain& c) { return matrix_to_numpy(c.posterior_cov); })
        .def_property_readonly("ess", [](const MCMCChain& c) { return vector_to_numpy(c.ess); })
        .def_property_readonly("autocorr_lag1", [](const MCMCChain& c) { return vector_to_numpy(c.autocorr_lag1); })
        .def_property_readonly("geweke_z", [](const MCMCChain& c) { return vector_to_numpy(c.geweke_z); })
        .def_readonly("sampler", &MCMCChain::sampler)
        .def_readonly("elapsed_ms", &MCMCChain::elapsed_ms);

    // ========== BayesGARCHResult ==========
    py::class_<BayesGARCHResult>(m, "BayesGARCHResult")
        .def_readonly("chain", &BayesGARCHResult::chain)
        .def_readonly("omega_mean", &BayesGARCHResult::omega_mean)
        .def_readonly("alpha_mean", &BayesGARCHResult::alpha_mean)
        .def_readonly("beta_mean", &BayesGARCHResult::beta_mean)
        .def_readonly("persistence_mean", &BayesGARCHResult::persistence_mean)
        .def_readonly("omega_ci_lower", &BayesGARCHResult::omega_ci_lower)
        .def_readonly("omega_ci_upper", &BayesGARCHResult::omega_ci_upper)
        .def_readonly("alpha_ci_lower", &BayesGARCHResult::alpha_ci_lower)
        .def_readonly("alpha_ci_upper", &BayesGARCHResult::alpha_ci_upper)
        .def_readonly("beta_ci_lower", &BayesGARCHResult::beta_ci_lower)
        .def_readonly("beta_ci_upper", &BayesGARCHResult::beta_ci_upper)
        .def_property_readonly("conditional_var", [](const BayesGARCHResult& r) { return vector_to_numpy(r.conditional_var); })
        .def_readonly("dic", &BayesGARCHResult::dic)
        .def_readonly("waic", &BayesGARCHResult::waic)
        .def_readonly("marginal_likelihood", &BayesGARCHResult::marginal_likelihood)
        .def_readonly("elapsed_ms", &BayesGARCHResult::elapsed_ms);

    // ========== BayesGJRResult ==========
    py::class_<BayesGJRResult>(m, "BayesGJRResult")
        .def_readonly("chain", &BayesGJRResult::chain)
        .def_readonly("omega_mean", &BayesGJRResult::omega_mean)
        .def_readonly("alpha_mean", &BayesGJRResult::alpha_mean)
        .def_readonly("gamma_mean", &BayesGJRResult::gamma_mean)
        .def_readonly("beta_mean", &BayesGJRResult::beta_mean)
        .def_readonly("nu_mean", &BayesGJRResult::nu_mean)
        .def_readonly("persistence_mean", &BayesGJRResult::persistence_mean)
        .def_readonly("omega_ci_lower", &BayesGJRResult::omega_ci_lower)
        .def_readonly("omega_ci_upper", &BayesGJRResult::omega_ci_upper)
        .def_readonly("alpha_ci_lower", &BayesGJRResult::alpha_ci_lower)
        .def_readonly("alpha_ci_upper", &BayesGJRResult::alpha_ci_upper)
        .def_readonly("gamma_ci_lower", &BayesGJRResult::gamma_ci_lower)
        .def_readonly("gamma_ci_upper", &BayesGJRResult::gamma_ci_upper)
        .def_readonly("beta_ci_lower", &BayesGJRResult::beta_ci_lower)
        .def_readonly("beta_ci_upper", &BayesGJRResult::beta_ci_upper)
        .def_readonly("nu_ci_lower", &BayesGJRResult::nu_ci_lower)
        .def_readonly("nu_ci_upper", &BayesGJRResult::nu_ci_upper)
        .def_property_readonly("conditional_var", [](const BayesGJRResult& r) { return vector_to_numpy(r.conditional_var); })
        .def_readonly("waic", &BayesGJRResult::waic)
        .def_readonly("elapsed_ms", &BayesGJRResult::elapsed_ms);

    // ========== PredictiveResult ==========
    py::class_<PredictiveResult>(m, "PredictiveResult")
        .def_property_readonly("simulated_returns", [](const PredictiveResult& r) { return matrix_to_numpy(r.simulated_returns); })
        .def_property_readonly("simulated_vol", [](const PredictiveResult& r) { return matrix_to_numpy(r.simulated_vol); })
        .def_property_readonly("mean_forecast", [](const PredictiveResult& r) { return vector_to_numpy(r.mean_forecast); })
        .def_property_readonly("vol_forecast", [](const PredictiveResult& r) { return vector_to_numpy(r.vol_forecast); })
        .def_property_readonly("vol_ci_lower", [](const PredictiveResult& r) { return vector_to_numpy(r.vol_ci_lower); })
        .def_property_readonly("vol_ci_upper", [](const PredictiveResult& r) { return vector_to_numpy(r.vol_ci_upper); })
        .def_readonly("n_ahead", &PredictiveResult::n_ahead)
        .def_readonly("n_posterior", &PredictiveResult::n_posterior);

    // Metropolis-Hastings (general)
    m.def("metropolis_hastings", [](py::array_t<f64> initial_arr,
                                     py::function log_density_fn,
                                     std::size_t n_samples, std::size_t burn_in,
                                     std::size_t thin, py::array_t<f64> proposal_scale_arr,
                                     bool adapt, uint64_t seed) {
        auto initial = numpy_to_vector(initial_arr);
        LogDensityFn cpp_fn = [&log_density_fn](const Vector<f64>& theta) -> f64 {
            auto arr = vector_to_numpy(theta);
            py::object result = log_density_fn(arr);
            return result.cast<f64>();
        };
        MHOptions opts;
        opts.n_samples = n_samples;
        opts.burn_in = burn_in;
        opts.thin = thin;
        opts.adapt = adapt;
        opts.seed = seed;
        if (proposal_scale_arr.size() > 0) {
            opts.proposal_scale = numpy_to_vector(proposal_scale_arr);
        }
        auto result = metropolis_hastings(initial, cpp_fn, opts);
        if (!result) throw std::runtime_error("MH failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("initial"), py::arg("log_density"),
       py::arg("n_samples") = 10000, py::arg("burn_in") = 2000,
       py::arg("thin") = 1, py::arg("proposal_scale") = py::array_t<f64>(),
       py::arg("adapt") = true, py::arg("seed") = 42,
       "Metropolis-Hastings MCMC sampler");

    // Bayesian GARCH
    m.def("bayesian_garch", [](py::array_t<f64> returns_arr,
                                std::size_t n_samples, std::size_t burn_in,
                                std::size_t thin, uint64_t seed) {
        auto r = numpy_to_vector(returns_arr);
        BayesGARCHOptions opts;
        opts.n_samples = n_samples;
        opts.burn_in = burn_in;
        opts.thin = thin;
        opts.seed = seed;
        auto result = bayesian_garch(r, opts);
        if (!result) throw std::runtime_error("Bayesian GARCH failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("returns"), py::arg("n_samples") = 10000,
       py::arg("burn_in") = 5000, py::arg("thin") = 1, py::arg("seed") = 42,
       "Bayesian GARCH(1,1) via Adaptive Metropolis");

    // Bayesian GJR-GARCH with Student-t
    m.def("bayesian_gjr_garch", [](py::array_t<f64> returns_arr,
                                    std::size_t n_samples, std::size_t burn_in,
                                    std::size_t thin, bool estimate_nu, f64 nu_init, uint64_t seed) {
        auto r = numpy_to_vector(returns_arr);
        BayesGJROptions opts;
        opts.n_samples = n_samples;
        opts.burn_in = burn_in;
        opts.thin = thin;
        opts.seed = seed;
        opts.estimate_nu = estimate_nu;
        opts.nu_init = nu_init;
        auto result = bayesian_gjr_garch(r, opts);
        if (!result) throw std::runtime_error("Bayesian GJR failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("returns"), py::arg("n_samples") = 10000,
       py::arg("burn_in") = 5000, py::arg("thin") = 1,
       py::arg("estimate_nu") = true, py::arg("nu_init") = 8.0,
       py::arg("seed") = 42,
       "Bayesian GJR-GARCH(1,1) with Student-t errors");

    // GARCH posterior predictive
    m.def("garch_predictive", [](py::array_t<f64> returns_arr,
                                   const BayesGARCHResult& posterior,
                                   std::size_t n_ahead, std::size_t n_posterior, uint64_t seed) {
        auto r = numpy_to_vector(returns_arr);
        auto result = garch_predictive(r, posterior, n_ahead, n_posterior, seed);
        if (!result) throw std::runtime_error("Predictive failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("returns"), py::arg("posterior"),
       py::arg("n_ahead") = 10, py::arg("n_posterior") = 1000, py::arg("seed") = 42,
       "GARCH posterior predictive simulation");

    // GJR posterior predictive
    m.def("gjr_predictive", [](py::array_t<f64> returns_arr,
                                 const BayesGJRResult& posterior,
                                 std::size_t n_ahead, std::size_t n_posterior, uint64_t seed) {
        auto r = numpy_to_vector(returns_arr);
        auto result = gjr_predictive(r, posterior, n_ahead, n_posterior, seed);
        if (!result) throw std::runtime_error("GJR Predictive failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("returns"), py::arg("posterior"),
       py::arg("n_ahead") = 10, py::arg("n_posterior") = 1000, py::arg("seed") = 42,
       "GJR-GARCH posterior predictive simulation");

    m.def("local_projections", [](py::array_t<f64> y_arr, py::array_t<f64> x_arr,
                                   std::size_t max_horizon, std::size_t n_lags,
                                   const std::string& covariance) {
        auto y = numpy_to_vector(y_arr);
        auto x = numpy_to_vector(x_arr);
        Matrix<f64> controls(y.size(), 0);
        LPOptions opts;
        opts.max_horizon = max_horizon;
        opts.n_lags = n_lags;
        opts.covariance = parse_covariance(covariance);
        auto result = local_projections(y, x, controls, opts);
        if (!result) throw std::runtime_error("LP failed: " + result.status().message());
        return std::move(result).value();
    }, py::arg("y"), py::arg("x"), py::arg("max_horizon") = 12,
       py::arg("n_lags") = 4, py::arg("covariance") = "newey_west",
       "Local projections impulse response");
}
