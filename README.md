<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++20"/>
  <img src="https://img.shields.io/badge/Apple_Silicon-Native-black?style=for-the-badge&logo=apple&logoColor=white" alt="Apple Silicon"/>
  <img src="https://img.shields.io/badge/Metal-GPU_Compute-gray?style=for-the-badge&logo=apple&logoColor=white" alt="Metal"/>
  <img src="https://img.shields.io/badge/Accelerate-BLAS%2FLAPACK-orange?style=for-the-badge&logo=apple&logoColor=white" alt="Accelerate"/>
  <img src="https://img.shields.io/badge/Python-Bindings-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Tests-155_passing-brightgreen?style=for-the-badge" alt="Tests"/>
  <img src="https://img.shields.io/badge/macOS-15%2B-000000?style=for-the-badge&logo=macos&logoColor=white" alt="macOS"/>
</p>

<h1 align="center">⚡ HFMetal</h1>

<p align="center">
  <strong>High-Frequency Econometrics Engine for Apple Silicon</strong><br/>
  <em>Production-grade C++20 framework with Metal GPU acceleration, Accelerate BLAS/LAPACK, and Python bindings</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#api-reference">API</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#python">Python</a> •
  <a href="#citation">Citation</a>
</p>

---

## Author

**Simon-Pierre Boucher**
Université Laval, Québec, Canada
📧 [spbou4@ulaval.ca](mailto:spbou4@ulaval.ca)

---

## Overview

**HFMetal** is a numerically serious, performance-first econometrics framework designed from the ground up for Apple Silicon. It provides a hybrid CPU/GPU runtime that automatically dispatches workloads to the optimal backend:

| Backend | Use Case | Technology |
|---------|----------|------------|
| **CPU** | Control flow, small problems | C++20 threads |
| **Accelerate** | Dense linear algebra, solvers | Apple vDSP, BLAS, LAPACK |
| **Metal** | Massive parallel loops, batched estimation | Metal Shading Language |

The framework targets empirical finance and economics researchers who need:
- High-frequency data processing at scale (millions of 1-min/5-min bars)
- Rolling and batched estimation across assets, windows, and events
- Correct inference with robust standard errors (HAC, clustered, bootstrap)
- GPU acceleration where it actually helps

---

## Features

### 📊 Estimators

| Estimator | Covariance Options | GPU Support |
|-----------|-------------------|-------------|
| **OLS** | Classical, White, Newey-West HAC | ✅ |
| **Rolling OLS** | All covariance types | ✅ |
| **Expanding OLS** | All covariance types | — |
| **Batched OLS** | Classical, White | ✅ |
| **GLS / FGLS** | GLS covariance | — |
| **IV / 2SLS** | Classical, White, HAC | — |
| **Logit** | MLE (IRLS) | — |
| **Probit** | MLE (IRLS) | — |

### 📈 High-Frequency Finance

| Function | Description |
|----------|-------------|
| `log_returns()` | Logarithmic returns from prices |
| `simple_returns()` | Arithmetic returns |
| `realized_variance()` | Sum of squared intraday returns |
| `realized_volatility()` | Square root of RV |
| `bipower_variation()` | Jump-robust volatility measure |
| `realized_semivariance()` | Upside/downside RV decomposition |
| `compute_realized_measures()` | All measures + jump statistic |
| `event_study()` | Abnormal returns, CAR, volatility response |

### 📉 Time Series Models

| Model | Features |
|-------|----------|
| **AR(p)** | OLS-based, AIC/BIC, HAC SE |
| **VAR(p)** | Per-equation covariance, residual cross-covariance, AIC/BIC |
| **HAR-RV** | Heterogeneous Autoregressive for realized volatility |
| **GARCH(1,1)** | MLE (Nelder-Mead), conditional variance, standardized residuals |
| **Local Projections** | Jordà LP-IRF, cumulative IRF, HAC confidence bands |

### 📋 Panel Data

| Method | Features |
|--------|----------|
| **Fixed Effects** | Entity, time, two-way demeaning |
| **Clustered SE** | One-way, two-way (Cameron-Gelbach-Miller) |

### 🏦 Empirical Finance

| Model | Description |
|-------|-------------|
| **Fama-MacBeth** | Cross-sectional regressions with NW-corrected SE |
| **Rolling Betas** | Via rolling OLS |
| **Event Studies** | Window extraction, CAR/CAAR, grouped events |

### 🔄 Simulation & Inference

| Method | Types |
|--------|-------|
| **Bootstrap** | IID, block, circular block |
| **Pairs Bootstrap** | For regression coefficients |
| **Confidence Intervals** | Percentile-based |

### 🖥️ Metal GPU Kernels

10 shader files providing GPU-accelerated:
- Returns computation (log, simple)
- Reductions (sum, sum-of-squares, partial)
- Rolling aggregates (sum, mean, variance)
- Realized measures (RV, bipower variation)
- Event window extraction & CAR
- Bootstrap index generation & gathering
- GARCH batched log-likelihood
- Batched OLS accumulation (X'X, X'y, residuals)
- Batched matrix-vector operations

---

## Quickstart

### Prerequisites

- macOS 15+ on Apple Silicon (M1/M2/M3/M4/M5)
- CMake 3.24+
- Apple Clang (Xcode Command Line Tools)
- Python 3.9+ (for Python bindings)

### Build

```bash
git clone https://github.com/simonpierreboucher02/hfmetal.git
cd hfmetal
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHFM_BUILD_PYTHON=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Run Tests

```bash
cd build
ctest --output-on-failure
```

### C++ Example

```cpp
#include "hfm/hfm.hpp"
using namespace hfm;

int main() {
    // Log returns from prices
    Series<f64> prices({100.0, 101.5, 99.8, 102.3, 103.1});
    auto returns = log_returns(prices);

    // Realized variance
    auto rv = realized_variance(returns);

    // OLS with Newey-West HAC standard errors
    Vector<f64> y = /* ... */;
    Matrix<f64> X = /* ... */;
    auto res = ols(y, X, OLSOptions{}
        .set_covariance(CovarianceType::NeweyWest)
        .set_hac_lag(8));
    std::cout << res.value().summary();

    // GARCH(1,1)
    auto garch_res = garch(returns);
    auto& g = garch_res.value();
    // g.omega, g.alpha, g.beta, g.conditional_var, g.std_residuals

    // Rolling OLS
    auto roll = rolling_ols(y, X, RollingOptions{}.set_window(250).set_step(1));

    // VAR(1)
    Matrix<f64> Y_multi = /* T x n_vars */;
    auto var_res = var(Y_multi, VAROptions{}.set_p(1));
}
```

### Python Example

```python
import numpy as np
# From build directory:
import _hfmetal as hfm

prices = np.array([100.0, 101.5, 99.8, 102.3, 103.1, 100.9, 104.2])
returns = hfm.log_returns(prices)

# Realized measures
rv = hfm.realized_variance(returns)
rm = hfm.compute_realized_measures(returns)

# OLS with HAC
X = np.column_stack([np.ones(100), np.random.randn(100)])
y = X @ [2.0, 3.0] + np.random.randn(100) * 0.1
res = hfm.ols(y, X, covariance="newey_west")
print(res.summary())

# GARCH(1,1)
g = hfm.garch(returns)
print(f"α={g.alpha:.4f}, β={g.beta:.4f}, persistence={g.persistence:.4f}")

# VAR
Y = np.random.randn(300, 2)
v = hfm.var(Y, p=1)

# Logit
y_binary = (returns[1:] > 0).astype(float)
X_logit = returns[:-1].reshape(-1, 1)
lr = hfm.logit(y_binary, X_logit)
print(f"Pseudo-R² = {lr.pseudo_r_squared:.4f}")

# Local projections
lp = hfm.local_projections(y, returns[:100], max_horizon=10)
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
    enum class CovarianceType { Classical, White, NeweyWest, ClusteredOneWay, ClusteredTwoWay };
}
```

### Result Pattern

All estimators return `Result<T>` which is either a value or an error:

```cpp
auto result = ols(y, X);
if (result) {
    auto& res = result.value();
    // use res.coefficients(), res.r_squared(), etc.
} else {
    std::cerr << result.status().message();
}
```

### Estimator Results

Every estimator result provides:
- `coefficients()` — parameter estimates
- `std_errors()` — standard errors
- `t_stats()` — t-statistics
- `p_values()` — p-values
- `covariance_matrix()` — variance-covariance matrix
- `r_squared()` — goodness of fit
- `elapsed_ms()` — computation time
- `summary()` — formatted output string

---

## Benchmarks

Measured on Apple M5 Max (18 cores, 128GB RAM):

| Operation | Size | Time | Throughput |
|-----------|------|------|------------|
| Log returns | 10M | 17.7ms | **565M elements/s** |
| Simple returns | 10M | 1.55ms | **6.4G elements/s** |
| Realized variance | 10M | 5.0ms | **2.0G elements/s** |
| Bipower variation | 10M | 4.9ms | **2.0G elements/s** |
| Vector dot product | 1M | 41.7µs | **24G elements/s** |
| Matrix multiply 1024² | — | 4.5ms | **239 GFLOP/s** |
| OLS (100K obs, 2 vars) | — | 3.5ms | — |
| OLS + Newey-West (100K) | — | 20ms | — |
| Rolling OLS (10K, w=60) | — | 64ms | **156K windows/s** |
| Batched OLS (1000 regressions) | — | 1.9ms | **517K regressions/s** |

### Real Data Performance

Tested on 5-minute OHLCV data from DuckDB (BTC, ETH, SPX, EURUSD):

| Test | Data | Result |
|------|------|--------|
| BTC realized variance | 1.3M 5-min returns | 0.8ms (**1.6G elements/s**) |
| EURUSD rolling OLS | 20K bars, 500-bar window | 3.9ms (390 windows) |
| BTC GARCH(1,1) | 173 daily returns | 0.1ms |
| BTC HAR-RV | 173 days | R²=0.279 |
| SPX OLS + HAC | 100K 5-min bars | 9.7ms |

---

## Architecture

```
hfmetal/
├── include/hfm/
│   ├── core/          # Types, status, assertions, numeric traits
│   ├── data/          # Timestamp, Series<T>
│   ├── linalg/        # Vector<T>, Matrix<T>, solvers (QR, Cholesky, SVD)
│   ├── runtime/       # ThreadPool, ExecutionPlanner
│   ├── metal/         # MetalContext, pipeline cache, kernel launcher
│   ├── estimators/    # OLS, rolling/batched/expanding OLS, GLS, IV/2SLS
│   ├── covariance/    # Classical, White, HAC, clustered (1-way, 2-way)
│   ├── hf/            # Returns, realized measures, event study
│   ├── timeseries/    # AR, VAR, HAR-RV, rolling, local projections
│   ├── panel/         # Fixed effects, clustered SE
│   ├── models/        # Fama-MacBeth, GARCH, logit/probit
│   ├── simulation/    # Bootstrap (IID, block, circular, pairs)
│   └── hfm.hpp        # Umbrella header
├── src/               # Implementation files
├── shaders/           # 10 Metal compute shaders
├── tests/
│   ├── unit/          # 121 C++ unit tests (Google Test)
│   ├── integration/   # 34 Python tests on real DuckDB data
│   └── metal/         # GPU equivalence tests
├── benchmarks/        # Google Benchmark suites
├── python/            # pybind11 bindings + hfmetal package
├── examples/          # C++ and Python examples
└── cmake/             # Build modules (warnings, sanitizers, Metal, deps)
```

### Backend Dispatch

The `ExecutionPlanner` automatically selects the optimal backend:

```
                    ┌─────────────────┐
                    │  ExecutionPlanner │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌────────┐    ┌───────────┐   ┌────────┐
         │  CPU   │    │Accelerate │   │ Metal  │
         │threads │    │BLAS/LAPACK│   │  GPU   │
         └────────┘    └───────────┘   └────────┘
              │              │              │
         Small tasks    Dense linalg   Batched/parallel
         Control flow   Solves         Rolling windows
         Branchy code   Factorizations Bootstrap/MC
```

### Numeric Design

- **Default precision**: `double` (f64) for all econometric inference
- **Solver strategy**: QR for OLS (numerical stability), Cholesky for SPD systems
- **Condition checking**: Rank deficiency detected and reported
- **Missing values**: Explicit NA policy — preprocess before kernels
- **Determinism**: CPU deterministic mode available; GPU tolerance documented

---

## Python

### Installation (from source)

```bash
cd hfmetal
pip install .
```

### Available Functions

```python
import hfmetal as hfm

# High-frequency
hfm.log_returns(prices)
hfm.simple_returns(prices)
hfm.realized_variance(returns)
hfm.realized_volatility(returns)
hfm.bipower_variation(returns)
hfm.compute_realized_measures(returns)
hfm.event_study(returns, event_indices, left=-60, right=60)

# Estimators
hfm.ols(y, X, covariance="newey_west", hac_lag=8)
hfm.rolling_ols(y, X, window=250, step=1)
hfm.gls(y, X, Omega)
hfm.fgls(y, X)
hfm.iv_2sls(y, X, Z, covariance="white")
hfm.logit(y, X)
hfm.probit(y, X)

# Time series
hfm.ar(y, p=2)
hfm.var(Y, p=1)
hfm.har_rv(daily_rv)
hfm.local_projections(y, x, max_horizon=12)

# Models
hfm.garch(returns)
hfm.fama_macbeth(y, X, time_ids)
hfm.fixed_effects(y, X, entity_ids, time_ids, cluster_entity=True)

# Simulation
hfm.bootstrap(data, statistic_fn, n_bootstrap=1000)
```

---

## Build Options

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `HFM_BUILD_TESTS` | ON | Build Google Test suite |
| `HFM_BUILD_BENCHMARKS` | ON | Build Google Benchmark suite |
| `HFM_BUILD_PYTHON` | OFF | Build pybind11 Python module |
| `HFM_BUILD_EXAMPLES` | ON | Build C++ examples |
| `HFM_ENABLE_METAL` | ON | Enable Metal GPU backend |
| `HFM_ENABLE_SANITIZERS` | ON | ASan/UBSan in Debug builds |

---

## Dependencies

| Dependency | Source | Purpose |
|-----------|--------|---------|
| **Accelerate** | macOS system | BLAS, LAPACK, vDSP |
| **Metal** | macOS system | GPU compute |
| **Foundation** | macOS system | Runtime support |
| **Google Test** | FetchContent | Unit testing |
| **Google Benchmark** | FetchContent | Performance benchmarks |
| **pybind11** | FetchContent | Python bindings |

No vendored or external C++ libraries required — all core dependencies are Apple system frameworks.

---

## Tested Data

Integration-tested on real market data (5-minute OHLCV bars via DuckDB):

| Asset Class | Database | Symbols | Rows |
|------------|----------|---------|------|
| **Crypto** | `crypto_5min.duckdb` | 74 (BTC, ETH, ...) | 39.3M |
| **Futures** | `futures_5min.duckdb` | 131 (ES, NQ, ...) | 73.2M |
| **FX** | `fx_5min.duckdb` | 78 (EURUSD, ...) | 82.6M |
| **Indices** | `index_5min.duckdb` | 125 (SPX, NDX, ...) | 39.9M |

---

## Citation

If you use HFMetal in academic research, please cite:

```bibtex
@software{boucher2026hfmetal,
  author       = {Boucher, Simon-Pierre},
  title        = {{HFMetal}: High-Frequency Econometrics Engine for Apple Silicon},
  year         = {2026},
  institution  = {Université Laval},
  url          = {https://github.com/simonpierreboucher02/hfmetal}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with ❤️ at Université Laval • Optimized for Apple Silicon • Powered by Metal & Accelerate</sub>
</p>
