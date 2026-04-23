find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_hfmetal module.cpp)
target_link_libraries(_hfmetal PRIVATE hfmetal)
target_compile_definitions(_hfmetal PRIVATE PYBIND11_MODULE)
