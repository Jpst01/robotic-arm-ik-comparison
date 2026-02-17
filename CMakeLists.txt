cmake_minimum_required(VERSION 3.14)
project(robotic_arm_ik LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(ik_core STATIC
    cpp/kinematics/Kinematics.cpp
    cpp/planning/TrajectoryPlanner.cpp
    cpp/planning/ObstacleChecker.cpp
    cpp/control/ControlLoop.cpp
)
target_include_directories(ik_core PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(ik_core PUBLIC Eigen3::Eigen)

add_executable(robotic_arm_demo cpp/main.cpp)
target_link_libraries(robotic_arm_demo PRIVATE ik_core)

add_executable(run_tests tests/test_kinematics.cpp)
target_link_libraries(run_tests PRIVATE ik_core)

find_package(onnxruntime QUIET)

if(NOT onnxruntime_FOUND)
    find_library(ONNXRUNTIME_LIB onnxruntime)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
              PATH_SUFFIXES onnxruntime onnxruntime/core/session)
endif()

if(onnxruntime_FOUND OR ONNXRUNTIME_LIB)
    message(STATUS "ONNX Runtime found — building ML targets")

    add_library(ik_ml STATIC
        cpp/ml_inference/MLInference.cpp
    )
    target_include_directories(ik_ml PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
    target_link_libraries(ik_ml PUBLIC ik_core)

    if(onnxruntime_FOUND)
        target_link_libraries(ik_ml PUBLIC onnxruntime::onnxruntime)
    else()
        target_include_directories(ik_ml PUBLIC ${ONNXRUNTIME_INCLUDE})
        target_link_libraries(ik_ml PUBLIC ${ONNXRUNTIME_LIB})
    endif()

    add_executable(ml_demo cpp/ml_demo.cpp)
    target_link_libraries(ml_demo PRIVATE ik_ml)

    add_executable(benchmark cpp/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE ik_ml)
else()
    message(STATUS "ONNX Runtime NOT found — ML targets will not be built")
    message(STATUS "  Install ONNX Runtime and re-run cmake to enable ML targets")
endif()
