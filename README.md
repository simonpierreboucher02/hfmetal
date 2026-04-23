<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++20"/>
  <img src="https://img.shields.io/badge/Apple_Silicon-Native-black?style=for-the-badge&logo=apple&logoColor=white" alt="Apple Silicon"/>
  <img src="https://img.shields.io/badge/Metal-GPU_Compute-gray?style=for-the-badge&logo=apple&logoColor=white" alt="Metal"/>
  <img src="https://img.shields.io/badge/Accelerate-BLAS%2FLAPACK-orange?style=for-the-badge&logo=apple&logoColor=white" alt="Accelerate"/>
  <img src="https://img.shields.io/badge/Python-Bindings-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Tests-184_passing-brightgreen?style=for-the-badge" alt="Tests"/>
  <img src="https://img.shields.io/badge/macOS-15%2B-000000?style=for-the-badge&logo=macos&logoColor=white" alt="macOS"/>
</p>

<h1 align="center">HFMetal</h1>

<p align="center">
  <strong>High-Frequency Econometrics Engine for Apple Silicon</strong><br/>
  <em>Production-grade C++20 framework with Metal GPU acceleration, Accelerate BLAS/LAPACK, and Python bindings</em>
</p>

<p align="center">
  <a href="#features">Features</a> &nbsp;&middot;&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;&middot;&nbsp;
  <a href="#api-reference">API Reference</a> &nbsp;&middot;&nbsp;
  <a href="#benchmarks">Benchmarks</a> &nbsp;&middot;&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;&middot;&nbsp;
  <a href="#python-bindings">Python</a> &nbsp;&middot;&nbsp;
  <a href="#citation">Citation</a>
</p>

---

## Author

**Simon-Pierre Boucher** &nbsp;|&nbsp; Universit&eacute; Laval, Qu&eacute;bec, Canada &nbsp;|&nbsp; [spbou4@ulaval.ca](mailto:spbou4@ulaval.ca)

---

## Overview

**HFMetal** is a numerically serious, performance-first econometrics framework built from the ground up for Apple Silicon. It provides a hybrid CPU/GPU runtime that automatically dispatches workloads to the optimal backend:

| Backend | Use Case | Technology |
|---------|----------|------------|
| **CPU** | Control flow, small problems | C++20 threads |
| **Accelerate** | Dense linear algebra, solvers | Apple vDSP, BLAS, LAPACK |
| **Metal** | Massive parallel loops, batched estimation | Metal Shading Language |

Built for empirical finance and economics researchers who need high-frequency data processing at scale, correct inference with robust standard errors, and GPU acceleration where it matters.

---

## Features

### Estimators

| Estimator | Covariance Options | GPU |
|-----------|-------------------|:---:|
| **OLS** | Classical, White, Newey-West HAC | Y |
| **Rolling OLS** | All covariance types | Y |
| **Expanding OLS** | All covariance types | |
| **Batched OLS** | Classical, White | Y |
| **GLS / FGLS** | GLS covariance | |
| **IV / 2SLS** | Classical, White, HAC + Sargan test | |
| **Logit** | MLE via IRLS, marginal effects | |
| **Probit** | MLE via IRLS, marginal effects | |

### High-Frequency Finance

| Function | Description |
|----------|-------------|
| `log_returns()` / `simple_returns()` | Logarithmic and arithmetic returns |
| `realized_variance()` / `realized_volatility()` | Sum of squared intraday returns |
| `bipower_variation()` | Jump-robust volatility measure (Barndorff-Nielsen & Shephard) |
| `realized_semivariance()` | Upside/downside RV decomposition |
| `compute_realized_measures()` | All measures + jump statistic in one call |
| `event_study()` | Abnormal returns, CAR, volatility response |

### Volatility Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| **GARCH(1,1)** | &omega;, &alpha;, &beta; | Standard conditional variance model |
| **EGARCH(1,1)** | &omega;, &alpha;, &gamma;, &beta; | Log-variance with leverage effect |
| **GJR-GARCH(1,1)** | &omega;, &alpha;, &gamma;, &beta; | Asymmetric response to negative shocks |
| **GARCH(1,1)-t** | &omega;, &alpha;, &beta;, &nu; | Student-t innovations for heavy tails |
| **Bayesian GARCH** | posterior samples | Adaptive Metropolis MCMC estimation |
| **Bayesian GJR-GARCH-t** | posterior samples | Full Bayesian with leverage + Student-t |

All frequentist models estimated via MLE (Nelder-Mead with transformed parameters). Each returns conditional variance series, standardized residuals, AIC/BIC, and numerical standard errors.

### Time Series

| Model | Features |
|-------|----------|
| **AR(p)** | OLS-based, AIC/BIC lag selection, HAC standard errors |
| **ARIMA(p,d,q)** | Differencing, iterative CLS for MA, forecasting with confidence bands |
| **VAR(p)** | Per-equation covariance, residual cross-covariance &Sigma;<sub>u</sub>, AIC/BIC |
| **HAR-RV** | Heterogeneous Autoregressive for realized volatility (daily/weekly/monthly) |
| **Local Projections** | Jord&agrave; LP-IRF, cumulative IRF, HAC confidence bands |
| **Granger Causality** | F-test of predictive content (restricted vs. unrestricted) |
| **Impulse Response (IRF)** | Orthogonalized via Cholesky decomposition of &Sigma;<sub>u</sub> |
| **FEVD** | Forecast error variance decomposition from VAR |
| **Forecast Evaluation** | MAE, RMSE, MAPE, MSE, Theil-U, R&sup2; |

### Statistical Diagnostics

| Test | Null Hypothesis | Statistic |
|------|----------------|-----------|
| **Jarque-Bera** | Normality | &chi;&sup2;(2) |
| **Durbin-Watson** | No first-order autocorrelation | DW &asymp; 2 |
| **Ljung-Box** | No autocorrelation up to lag *m* | Q ~ &chi;&sup2;(m) |
| **Breusch-Pagan** | Homoskedasticity | nR&sup2; ~ &chi;&sup2;(k) |
| **White** | Homoskedasticity (with cross-terms) | nR&sup2; ~ &chi;&sup2;(p) |
| **ARCH-LM** | No ARCH effects | TR&sup2; ~ &chi;&sup2;(q) |
| **ADF** | Unit root (non-stationary) | Dickey-Fuller &tau; |
| **KPSS** | Stationarity | LM statistic |

Plus: `autocorrelation()` (ACF), `descriptive_stats()` (mean, variance, skewness, kurtosis, quantiles), `chi2_cdf()`, `normal_cdf()`.

### Risk & Portfolio

| Function | Description |
|----------|-------------|
| `value_at_risk()` | Historical, Parametric, Cornish-Fisher VaR + CVaR (Expected Shortfall) |
| `drawdown_analysis()` | Maximum drawdown, peak/trough indices, drawdown series, duration |
| `performance_metrics()` | Sharpe, Sortino, Calmar, Omega ratios; annualized return/vol |
| `minimum_variance_portfolio()` | Markowitz minimum-variance weights |
| `max_sharpe_portfolio()` | Tangency portfolio (maximum Sharpe ratio) |
| `efficient_frontier()` | Full efficient frontier with *n* points |

### Panel Data

| Method | Features |
|--------|----------|
| **Fixed Effects** | Entity, time, two-way demeaning |
| **Clustered SE** | One-way, two-way (Cameron-Gelbach-Miller) |
| **Fama-MacBeth** | Cross-sectional regressions with Newey-West corrected SE |

### Simulation & Inference

| Method | Description |
|--------|-------------|
| **Bootstrap** | IID, block, circular block |
| **Pairs Bootstrap** | For regression coefficients (resample y,X jointly) |
| **Metropolis-Hastings** | General MCMC with adaptive proposal |
| **Adaptive Metropolis** | Haario et al. (2001) learned covariance |
| **Posterior Predictive** | GARCH / GJR-GARCH forward simulation |

### Linear Algebra & Numerical

| Function | Backend | Description |
|----------|---------|-------------|
| `solve_least_squares()` | LAPACK dgels | QR-based least squares |
| `cholesky()` | LAPACK dpotrf | Cholesky factorization |
| `invert_spd()` | LAPACK dpotri | SPD matrix inversion |
| `svd()` | LAPACK dgesvd | Singular value decomposition |
| `eigen_symmetric()` | LAPACK dsyev | Eigenvalues & eigenvectors (symmetric) |
| `condition_number()` | via SVD | Ratio &sigma;<sub>max</sub>/&sigma;<sub>min</sub> |
| `pca()` | eigen + matmul | Principal Component Analysis (scores, loadings, explained variance) |
| `matmul()` / `matmul_AtB()` | BLAS dgemm | Accelerate-backed matrix multiply |
| `matvec()` | BLAS dgemv | Matrix-vector product |

### Metal GPU Kernels

10 Metal compute shaders providing GPU-accelerated:

| Shader | Operations |
|--------|-----------|
| `returns.metal` | Log returns, simple returns |
| `reductions.metal` | Sum, sum-of-squares, partial reductions |
| `rolling.metal` | Rolling sum, mean, variance |
| `realized_measures.metal` | Realized variance, bipower variation |
| `event_study.metal` | Window extraction, CAR computation |
| `bootstrap.metal` | Index generation, resample gathering |
| `garch.metal` | Batched log-likelihood evaluation |
| `batched_ols.metal` | X'X, X'y accumulation |
| `matvec.metal` | Parallel matrix-vector operations |
| `likelihoods.metal` | General log-likelihood kernels |

---

## Quickstart

### Prerequisites

- macOS 15+ on Apple Silicon (M1/M2/M3/M4/M5)
- CMake 3.24+
- Apple Clang (Xcode Command Line Tools)
- Python 3.9+ (optional, for Python bindings)

### Build

```bash
git clone https://github.com/simonpierreboucher02/hfmetal.git
cd hfmetal
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHFM_BUILD_PYTHON=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Run Tests

```bash
cd build && ctest --output-on-failure
```

### C++ Usage

```cpp
#include "hfm/hfm.hpp"
using namespace hfm;

int main() {
    // --- Returns & Realized Measures ---
    Series<f64> prices({100.0, 101.5, 99.8, 102.3, 103.1});
    auto returns = log_returns(prices);
    auto rv = realized_variance(returns);

    // --- OLS with Newey-West HAC ---
    Vector<f64> y = /* ... */;
    Matrix<f64> X = /* ... */;
    auto res = ols(y, X, OLSOptions{}
        .set_covariance(CovarianceType::NeweyWest));
    std::cout << res.value().summary();

    // --- GARCH Family ---
    auto g = garch(returns).value();           // GARCH(1,1)
    auto eg = egarch(returns).value();         // EGARCH (leverage)
    auto gjr = gjr_garch(returns).value();     // GJR-GARCH (asymmetry)
    auto gt = garch_t(returns).value();        // GARCH-t (heavy tails)

    // --- ARIMA + Forecast ---
    auto model = arima(y, ARIMAOptions{}.set_p(2).set_d(1).set_q(1));
    auto fcast = arima_forecast(y, model.value(), 10);

    // --- Diagnostics ---
    auto jb = jarque_bera(y).value();          // Normality test
    auto adf = adf_test(y).value();            // Unit root
    auto lb = ljung_box(y, 10).value();        // Autocorrelation

    // --- Risk ---
    auto var = value_at_risk(returns, 0.95, VaRMethod::CornishFisher);
    auto perf = performance_metrics(returns);

    // --- Portfolio ---
    Vector<f64> mu({0.10, 0.15, 0.12});
    Matrix<f64> cov = /* covariance matrix */;
    auto portfolio = max_sharpe_portfolio(mu, cov);

    // --- VAR + IRF ---
    Matrix<f64> Y_multi = /* T x k */;
    auto var_res = var(Y_multi, VAROptions{}.set_p(2));
    auto irf = var_irf(var_res.value(), 20);    // Impulse responses
    auto fevd = var_fevd(var_res.value(), 20);   // Variance decomposition

    // --- Granger Causality ---
    auto gc = granger_causality(y1, y2, 4);

    // --- PCA ---
    auto pca_res = pca(X, 3);  // Top 3 components
}
```

### Python Usage

```python
import numpy as np
import _hfmetal as hfm

prices = np.array([100.0, 101.5, 99.8, 102.3, 103.1, 100.9, 104.2])
returns = hfm.log_returns(prices)

# --- Estimators ---
res = hfm.ols(y, X, covariance="newey_west")
g   = hfm.garch(returns)
eg  = hfm.egarch(returns)
gjr = hfm.gjr_garch(returns)
gt  = hfm.garch_t(returns)

# --- Time Series ---
model = hfm.arima(y, p=2, d=1, q=1)
gc    = hfm.granger_causality(y, x, n_lags=4)

# --- Diagnostics ---
jb  = hfm.jarque_bera(returns)
adf = hfm.adf_test(y)
lb  = hfm.ljung_box(returns, n_lags=10)
dw  = hfm.durbin_watson(residuals)
ds  = hfm.descriptive_stats(returns)

# --- Risk & Portfolio ---
var  = hfm.value_at_risk(returns, confidence=0.95, method="cornish_fisher")
dd   = hfm.drawdown_analysis(returns)
perf = hfm.performance_metrics(returns, risk_free_rate=0.02)
port = hfm.max_sharpe_portfolio(mu, cov_matrix, risk_free_rate=0.03)

# --- PCA ---
pca = hfm.pca(X, n_components=3)

# --- Forecast Evaluation ---
fe = hfm.forecast_eval(actual, predicted)
```

---

## API Reference

### Core Types

```cpp
namespace hfm {
    using f64 = double;
    using f32 = float;
    using i64 = std::int64_t;

    enum class Backend { Auto, CPU, Accelerate, Metal };
    enum class CovarianceType { Classical, White, NeweyWest,
                                 ClusteredOneWay, ClusteredTwoWay };
    enum class VaRMethod { Historical, Parametric, CornishFisher };
    enum class ADFType { None, Intercept, InterceptAndTrend };
}
```

### Result Pattern

All estimators return `Result<T>` &mdash; either a value or an error status:

```cpp
auto result = ols(y, X);
if (result) {
    auto& res = result.value();
    // res.coefficients(), res.r_squared(), res.summary()
} else {
    std::cerr << result.status().message();
}
```

### Options Pattern

All models accept an options struct with fluent builder methods:

```cpp
auto res = garch(returns, GARCHOptions{}
    .set_max_iter(1000)
    .set_tol(1e-10));

auto model = arima(y, ARIMAOptions{}
    .set_p(2).set_d(1).set_q(1));

auto perf = performance_metrics(returns, PerformanceOptions{}
    .set_risk_free_rate(0.02)
    .set_annualization_factor(252.0));
```

---

## Benchmarks

Measured on Apple M5 Max (18 cores, 128 GB RAM):

### Core Operations

| Operation | Size | Time | Throughput |
|-----------|------|------|------------|
| Log returns | 10M | 17.7 ms | **565M elem/s** |
| Simple returns | 10M | 1.55 ms | **6.4G elem/s** |
| Realized variance | 10M | 5.0 ms | **2.0G elem/s** |
| Bipower variation | 10M | 4.9 ms | **2.0G elem/s** |
| Vector dot product | 1M | 41.7 &micro;s | **24G elem/s** |
| Matrix multiply 1024&sup2; | &mdash; | 4.5 ms | **239 GFLOP/s** |
| OLS (100K obs, 2 vars) | &mdash; | 3.5 ms | &mdash; |
| OLS + Newey-West (100K) | &mdash; | 20 ms | &mdash; |
| Rolling OLS (10K, w=60) | &mdash; | 64 ms | **156K windows/s** |
| Batched OLS (1000 regs) | &mdash; | 1.9 ms | **517K regs/s** |

### Real Data

Tested on 5-minute OHLCV bars via DuckDB:

| Test | Data | Result |
|------|------|--------|
| BTC realized variance | 1.3M 5-min returns | 0.8 ms &mdash; **1.6G elem/s** |
| EURUSD rolling OLS | 20K bars, 500-bar window | 3.9 ms (390 windows) |
| BTC GARCH(1,1) | 173 daily returns | 0.1 ms |
| BTC HAR-RV | 173 days | R&sup2; = 0.279 |
| SPX OLS + HAC | 100K 5-min bars | 9.7 ms |

---

## Architecture

```
hfmetal/
├── include/hfm/
│   ├── core/            Types, Status, Result<T>, assertions
│   ├── data/            Timestamp, Series<T>
│   ├── linalg/          Vector<T>, Matrix<T>, solvers, SVD, eigen, PCA
│   ├── runtime/         ThreadPool, ExecutionPlanner
│   ├── metal/           MetalContext, pipeline cache, kernel launcher
│   ├── estimators/      OLS, rolling/batched/expanding OLS, GLS, IV/2SLS
│   ├── covariance/      Classical, White, HAC, clustered (1-way, 2-way)
│   ├── hf/              Returns, realized measures, event study
│   ├── timeseries/      AR, ARIMA, VAR, HAR-RV, LP, Granger, IRF, FEVD
│   ├── panel/           Fixed effects, clustered SE
│   ├── models/          GARCH, EGARCH, GJR-GARCH, GARCH-t, Fama-MacBeth,
│   │                    logit/probit, Bayesian GARCH/GJR
│   ├── simulation/      Bootstrap, MCMC (MH, Adaptive Metropolis)
│   ├── diagnostics/     JB, DW, LB, BP, White, ARCH-LM, ADF, KPSS
│   ├── risk/            VaR, CVaR, drawdown, Sharpe, portfolio optimization
│   └── hfm.hpp          Umbrella header
├── src/                 Implementation files (.cpp)
├── shaders/             10 Metal compute shaders (.metal)
├── tests/
│   ├── unit/            32 test files, 184 C++ unit tests (Google Test)
│   ├── integration/     34 Python tests on real DuckDB data
│   └── metal/           GPU vs CPU equivalence tests
├── benchmarks/          Google Benchmark suites
├── python/              pybind11 bindings (~900 lines)
├── examples/            C++ example programs
└── cmake/               Build modules (warnings, sanitizers, Metal, deps)
```

### Backend Dispatch

```
                    ┌──────────────────┐
                    │ ExecutionPlanner  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              v              v              v
         ┌────────┐    ┌───────────┐   ┌────────┐
         │  CPU   │    │Accelerate │   │ Metal  │
         │threads │    │BLAS/LAPACK│   │  GPU   │
         └────────┘    └───────────┘   └────────┘
              │              │              │
         Small tasks    Dense linalg   Batched/parallel
         Control flow   SVD, eigen     Rolling windows
         Branchy code   Solves, PCA    Bootstrap/MC
```

### Numerical Design

- **Precision**: `f64` (double) for all econometric inference
- **Solvers**: QR for OLS (stability), Cholesky for SPD, SVD for condition analysis
- **LAPACK**: dgesvd (SVD), dsyev (eigendecomposition), dgels (least squares), dpotrf/dpotri (Cholesky)
- **Constraints**: Transformed parameters for GARCH family (log/logit) ensure valid estimates
- **Determinism**: CPU deterministic mode available; GPU tolerance documented

---

## Build Options

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `HFM_BUILD_TESTS` | ON | Build Google Test suite (184 tests) |
| `HFM_BUILD_BENCHMARKS` | ON | Build Google Benchmark suite |
| `HFM_BUILD_PYTHON` | OFF | Build pybind11 Python module |
| `HFM_BUILD_EXAMPLES` | ON | Build C++ examples |
| `HFM_ENABLE_METAL` | ON | Enable Metal GPU backend |
| `HFM_ENABLE_SANITIZERS` | ON | ASan/UBSan in Debug builds |

## Dependencies

| Dependency | Source | Purpose |
|-----------|--------|---------|
| **Accelerate** | macOS system | BLAS, LAPACK, vDSP |
| **Metal** | macOS system | GPU compute shaders |
| **Foundation** | macOS system | Runtime support |
| **Google Test** | FetchContent | Unit testing |
| **Google Benchmark** | FetchContent | Performance benchmarks |
| **pybind11** | FetchContent | Python bindings |

No vendored or external C++ libraries &mdash; all core dependencies are Apple system frameworks.

---

## Tested Data

Integration-tested on real market data (5-minute OHLCV bars via DuckDB):

| Asset Class | Symbols | Rows |
|------------|---------|------|
| **Crypto** | 74 (BTC, ETH, ...) | 39.3M |
| **Futures** | 131 (ES, NQ, ...) | 73.2M |
| **FX** | 78 (EURUSD, ...) | 82.6M |
| **Indices** | 125 (SPX, NDX, ...) | 39.9M |
| **Total** | **408 symbols** | **235M rows** |

---

## Module Summary

| Module | Functions | Tests |
|--------|-----------|-------|
| Core & Data | Types, Status, Result, Series, Timestamp | 16 |
| Linear Algebra | Vector, Matrix, QR, Cholesky, SVD, Eigen, PCA | 25 |
| Estimators | OLS, Rolling, Batched, GLS, IV/2SLS | 20 |
| Covariance | Classical, White, HAC, Clustered (1-way, 2-way) | 6 |
| High-Frequency | Returns, RV, BPV, Semivariance, Event Study | 15 |
| Time Series | AR, ARIMA, VAR, HAR-RV, LP, Granger, IRF, FEVD | 26 |
| Panel | Fixed Effects, Clustered SE, Fama-MacBeth | 6 |
| Volatility Models | GARCH, EGARCH, GJR-GARCH, GARCH-t, Bayesian | 14 |
| Diagnostics | JB, DW, LB, BP, White, ARCH-LM, ADF, KPSS | 17 |
| Risk & Portfolio | VaR, CVaR, Drawdown, Sharpe, Markowitz | 9 |
| Simulation | Bootstrap, MCMC, Predictive | 11 |
| Logit/Probit | Binary models, marginal effects | 6 |
| Metal GPU | 10 shaders, context, equivalence | 2 |
| Runtime | ExecutionPlanner, ThreadPool | 5 |
| **Total** | | **184** |

---

## Citation

```bibtex
@software{boucher2026hfmetal,
  author       = {Boucher, Simon-Pierre},
  title        = {{HFMetal}: High-Frequency Econometrics Engine for Apple Silicon},
  year         = {2026},
  institution  = {Universit\'{e} Laval},
  url          = {https://github.com/simonpierreboucher02/hfmetal}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built at Universit&eacute; Laval &nbsp;&middot;&nbsp; Optimized for Apple Silicon &nbsp;&middot;&nbsp; Powered by Metal & Accelerate</sub>
</p>
