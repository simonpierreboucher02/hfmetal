#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"

namespace hfm {

// ========== Test result structs ==========

struct JarqueBeraResult {
    f64 statistic = 0.0;
    f64 p_value = 1.0;
    f64 skewness = 0.0;
    f64 excess_kurtosis = 0.0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

struct DurbinWatsonResult {
    f64 statistic = 0.0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

struct LjungBoxResult {
    f64 statistic = 0.0;
    f64 p_value = 1.0;
    std::size_t n_lags = 0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

struct BreuschPaganResult {
    f64 statistic = 0.0;
    f64 p_value = 1.0;
    std::size_t df = 0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

struct WhiteTestResult {
    f64 statistic = 0.0;
    f64 p_value = 1.0;
    std::size_t df = 0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

struct ArchLMResult {
    f64 statistic = 0.0;
    f64 p_value = 1.0;
    std::size_t n_lags = 0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

struct ADFResult {
    f64 statistic = 0.0;
    f64 p_value = 1.0;
    std::size_t n_lags = 0;
    std::size_t n_obs = 0;
    f64 critical_1pct = 0.0;
    f64 critical_5pct = 0.0;
    f64 critical_10pct = 0.0;
    bool has_intercept = true;
    bool has_trend = false;
    f64 elapsed_ms = 0.0;
};

struct KPSSResult {
    f64 statistic = 0.0;
    f64 p_value = 1.0;
    std::size_t n_lags = 0;
    std::size_t n_obs = 0;
    f64 critical_1pct = 0.0;
    f64 critical_5pct = 0.0;
    f64 critical_10pct = 0.0;
    bool trend = false;
    f64 elapsed_ms = 0.0;
};

// ========== Descriptive statistics ==========

struct DescriptiveStats {
    f64 mean = 0.0;
    f64 variance = 0.0;
    f64 std_dev = 0.0;
    f64 skewness = 0.0;
    f64 excess_kurtosis = 0.0;
    f64 min = 0.0;
    f64 max = 0.0;
    f64 median = 0.0;
    f64 q25 = 0.0;
    f64 q75 = 0.0;
    std::size_t n_obs = 0;
    f64 elapsed_ms = 0.0;
};

Result<DescriptiveStats> descriptive_stats(const Vector<f64>& x);

// ========== Normality tests ==========

Result<JarqueBeraResult> jarque_bera(const Vector<f64>& x);

// ========== Autocorrelation tests ==========

Result<DurbinWatsonResult> durbin_watson(const Vector<f64>& residuals);

Result<LjungBoxResult> ljung_box(const Vector<f64>& x, std::size_t n_lags = 10);

Vector<f64> autocorrelation(const Vector<f64>& x, std::size_t max_lag);

// ========== Heteroskedasticity tests ==========

Result<BreuschPaganResult> breusch_pagan(const Vector<f64>& residuals,
                                          const Matrix<f64>& X);

Result<WhiteTestResult> white_test(const Vector<f64>& residuals,
                                    const Matrix<f64>& X);

Result<ArchLMResult> arch_lm(const Vector<f64>& residuals, std::size_t n_lags = 5);

// ========== Stationarity / Unit root tests ==========

enum class ADFType : u32 {
    None,
    Intercept,
    InterceptAndTrend
};

Result<ADFResult> adf_test(const Vector<f64>& y, std::size_t max_lag = 0,
                            ADFType type = ADFType::Intercept);

Result<KPSSResult> kpss_test(const Vector<f64>& y, bool trend = false,
                              std::size_t n_lags = 0);

// ========== Distribution utilities ==========

f64 chi2_cdf(f64 x, f64 df);
f64 chi2_sf(f64 x, f64 df);
f64 normal_cdf(f64 x);
f64 normal_sf(f64 x);

} // namespace hfm
