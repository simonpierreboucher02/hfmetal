#pragma once

// Core
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/core/numeric_traits.hpp"
#include "hfm/core/assert.hpp"

// Data structures
#include "hfm/data/timestamp.hpp"
#include "hfm/data/series.hpp"

// Linear algebra
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/solver.hpp"

// Runtime
#include "hfm/runtime/thread_pool.hpp"
#include "hfm/runtime/execution_planner.hpp"

// Estimators
#include "hfm/estimators/ols.hpp"
#include "hfm/estimators/rolling_ols.hpp"
#include "hfm/estimators/batched_ols.hpp"
#include "hfm/estimators/gls.hpp"
#include "hfm/estimators/iv.hpp"

// Covariance
#include "hfm/covariance/covariance.hpp"

// High-frequency
#include "hfm/hf/returns.hpp"
#include "hfm/hf/realized_measures.hpp"
#include "hfm/hf/event_study.hpp"

// Time series
#include "hfm/timeseries/ar.hpp"
#include "hfm/timeseries/var.hpp"
#include "hfm/timeseries/har.hpp"
#include "hfm/timeseries/rolling.hpp"
#include "hfm/timeseries/local_projections.hpp"

// Panel
#include "hfm/panel/fixed_effects.hpp"

// Models
#include "hfm/models/fama_macbeth.hpp"
#include "hfm/models/garch.hpp"
#include "hfm/models/logit_probit.hpp"

// Simulation
#include "hfm/simulation/bootstrap.hpp"
#include "hfm/simulation/mcmc.hpp"

// Metal (conditional)
#ifdef HFM_METAL_ENABLED
#include "hfm/metal/metal_context.hpp"
#endif
