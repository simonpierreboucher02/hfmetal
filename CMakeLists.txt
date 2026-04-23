cmake_minimum_required(VERSION 3.24)
project(hfmetal VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HFM_BUILD_TESTS "Build tests" ON)
option(HFM_BUILD_BENCHMARKS "Build benchmarks" ON)
option(HFM_BUILD_PYTHON "Build Python bindings" OFF)
option(HFM_BUILD_EXAMPLES "Build examples" ON)
option(HFM_ENABLE_METAL "Enable Metal GPU backend" ON)
option(HFM_ENABLE_SANITIZERS "Enable sanitizers in debug" ON)

include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/MetalSupport.cmake)
include(cmake/Dependencies.cmake)

find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

if(HFM_ENABLE_METAL)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
endif()

file(GLOB_RECURSE HFM_SOURCES
    src/core/*.cpp
    src/data/*.cpp
    src/runtime/*.cpp
    src/linalg/*.cpp
    src/models/*.cpp
    src/estimators/*.cpp
    src/covariance/*.cpp
    src/timeseries/*.cpp
    src/hf/*.cpp
    src/panel/*.cpp
    src/simulation/*.cpp
    src/diagnostics/*.cpp
    src/risk/*.cpp
    src/api/*.cpp
)

if(HFM_ENABLE_METAL)
    file(GLOB_RECURSE HFM_METAL_SOURCES src/metal/*.cpp src/metal/*.mm)
    list(APPEND HFM_SOURCES ${HFM_METAL_SOURCES})
endif()

add_library(hfmetal STATIC ${HFM_SOURCES})
target_include_directories(hfmetal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(hfmetal PUBLIC ${ACCELERATE_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
target_compile_definitions(hfmetal PUBLIC ACCELERATE_NEW_LAPACK)
hfm_set_warnings(hfmetal)

if(HFM_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    hfm_enable_sanitizers(hfmetal)
endif()

if(HFM_ENABLE_METAL)
    target_link_libraries(hfmetal PUBLIC ${METAL_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
    target_compile_definitions(hfmetal PUBLIC HFM_METAL_ENABLED=1)
    hfm_compile_metal_shaders(hfmetal "${CMAKE_CURRENT_SOURCE_DIR}/shaders" "${CMAKE_CURRENT_BINARY_DIR}/shaders")
endif()

if(HFM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HFM_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(HFM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(HFM_BUILD_PYTHON)
    add_subdirectory(python)
endif()
