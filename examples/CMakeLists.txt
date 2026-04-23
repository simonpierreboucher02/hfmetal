add_executable(example_basic cpp/basic_usage.cpp)
target_link_libraries(example_basic PRIVATE hfmetal)

add_executable(example_ols cpp/ols_example.cpp)
target_link_libraries(example_ols PRIVATE hfmetal)

add_executable(example_event_study cpp/event_study_example.cpp)
target_link_libraries(example_event_study PRIVATE hfmetal)

add_executable(example_comprehensive cpp/comprehensive_example.cpp)
target_link_libraries(example_comprehensive PRIVATE hfmetal)
