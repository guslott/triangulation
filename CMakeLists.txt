cmake_minimum_required(VERSION 3.16)
project(triangulation LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set optimization flags for different build types
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find Eigen package
find_package(Eigen3 REQUIRED)

if(TARGET Eigen3::Eigen)
    message(STATUS "Using target Eigen3::Eigen")
    set(EIGEN3_LIBRARIES Eigen3::Eigen)
else()
    message(STATUS "Using EIGEN3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

# Include paths: core headers at root, baseline/benchmark headers in benchmarks/
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/benchmarks)

# Benchmark and evaluation executables
add_executable(bench_speed benchmarks/bench_speed.cpp)
add_executable(bench_scaling benchmarks/bench_scaling.cpp)
add_executable(bench_correctness benchmarks/bench_correctness.cpp)
add_executable(bench_approximation benchmarks/bench_approximation.cpp)
add_executable(baseline_acceptance benchmarks/baseline_acceptance.cpp)
add_executable(theorem_regression benchmarks/theorem_regression.cpp)
add_executable(conditioning_sweep benchmarks/conditioning_sweep.cpp)
add_executable(oxford_dinosaur_real benchmarks/oxford_dinosaur_real.cpp)

# Link against Eigen
if(TARGET Eigen3::Eigen)
    target_link_libraries(bench_speed PRIVATE ${EIGEN3_LIBRARIES})
    target_link_libraries(bench_scaling PRIVATE ${EIGEN3_LIBRARIES})
    target_link_libraries(bench_correctness PRIVATE ${EIGEN3_LIBRARIES})
    target_link_libraries(bench_approximation PRIVATE ${EIGEN3_LIBRARIES})
    target_link_libraries(baseline_acceptance PRIVATE ${EIGEN3_LIBRARIES})
    target_link_libraries(theorem_regression PRIVATE ${EIGEN3_LIBRARIES})
    target_link_libraries(conditioning_sweep PRIVATE ${EIGEN3_LIBRARIES})
    target_link_libraries(oxford_dinosaur_real PRIVATE ${EIGEN3_LIBRARIES})
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME theorem_regression COMMAND theorem_regression)
    add_test(
        NAME conditioning_sweep
        COMMAND conditioning_sweep
                ${CMAKE_BINARY_DIR}/conditioning_sweep_cases.csv
                ${CMAKE_BINARY_DIR}/conditioning_sweep_bins.csv
                ${CMAKE_BINARY_DIR}/conditioning_sweep_summary.md)
endif()
