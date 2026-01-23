# CMake integration test for TensorFlowWrap
# This tests that find_package(TensorFlowWrap) works and tf::wrapper target exists
#
# Usage:
#   cd build && cmake --install . --prefix /tmp/tf_install
#   cd tests/cmake_integration_test
#   cmake -B build -DCMAKE_PREFIX_PATH=/tmp/tf_install
#   cmake --build build
#   ./build/integration_test

cmake_minimum_required(VERSION 3.16)
project(TensorFlowWrap_IntegrationTest CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed TensorFlowWrap package
find_package(TensorFlowWrap REQUIRED)

# Test that tf::wrapper target exists
if(NOT TARGET tf::wrapper)
    message(FATAL_ERROR "tf::wrapper target not found after find_package(TensorFlowWrap)")
endif()

message(STATUS "Found tf::wrapper target - integration test passed!")

# Create a minimal test executable
add_executable(integration_test main.cpp)
target_link_libraries(integration_test PRIVATE tf::wrapper)
