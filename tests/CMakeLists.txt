include(GoogleTest)

function(hfm_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE hfmetal GTest::gtest_main)
    gtest_discover_tests(${name})
endfunction()

# Core & data
hfm_add_test(test_core unit/test_core.cpp)
hfm_add_test(test_vector unit/test_vector.cpp)
hfm_add_test(test_matrix unit/test_matrix.cpp)
hfm_add_test(test_timestamp unit/test_timestamp.cpp)
hfm_add_test(test_series unit/test_series.cpp)

# HF primitives
hfm_add_test(test_returns unit/test_returns.cpp)
hfm_add_test(test_realized_measures unit/test_realized_measures.cpp)
hfm_add_test(test_event_study unit/test_event_study.cpp)

# Estimators
hfm_add_test(test_ols unit/test_ols.cpp)
hfm_add_test(test_rolling_ols unit/test_rolling_ols.cpp)
hfm_add_test(test_batched_ols unit/test_batched_ols.cpp)
hfm_add_test(test_covariance unit/test_covariance.cpp)

# Panel
hfm_add_test(test_panel unit/test_panel.cpp)

# Time series
hfm_add_test(test_timeseries unit/test_timeseries.cpp)
hfm_add_test(test_var unit/test_var.cpp)

# Models
hfm_add_test(test_fama_macbeth unit/test_fama_macbeth.cpp)
hfm_add_test(test_garch unit/test_garch.cpp)
hfm_add_test(test_logit_probit unit/test_logit_probit.cpp)

# Estimators (additional)
hfm_add_test(test_gls unit/test_gls.cpp)
hfm_add_test(test_iv unit/test_iv.cpp)
hfm_add_test(test_clustered_cov unit/test_clustered_cov.cpp)
hfm_add_test(test_execution_planner unit/test_execution_planner.cpp)

# Time series (additional)
hfm_add_test(test_local_projections unit/test_local_projections.cpp)

# Simulation
hfm_add_test(test_bootstrap unit/test_bootstrap.cpp)

# Metal
if(HFM_ENABLE_METAL)
    hfm_add_test(test_metal_context metal/test_metal_context.cpp)
    hfm_add_test(test_metal_returns metal/test_metal_returns.cpp)
endif()
