cmake_minimum_required(VERSION 3.16)
project(TensorFlowWrap VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(TF_WRAPPER_USE_STUB "Use stub instead of real TensorFlow" ON)
option(TF_WRAPPER_BUILD_TESTS "Build tests" ON)

# Stub library
if(TF_WRAPPER_USE_STUB)
    add_library(tf_stub STATIC
        third_party/tf_stub/tf_c_stub.cpp
    )
    target_include_directories(tf_stub PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tf_stub
    )
    target_compile_definitions(tf_stub PUBLIC TF_WRAPPER_TF_STUB_ENABLED)
endif()

# Main header-only library
add_library(tf_wrap INTERFACE)
target_include_directories(tf_wrap INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(tf_wrap INTERFACE cxx_std_20)

if(TF_WRAPPER_USE_STUB)
    target_link_libraries(tf_wrap INTERFACE tf_stub)
    target_compile_definitions(tf_wrap INTERFACE TF_WRAPPER_TF_STUB_ENABLED)
endif()

# Tests
if(TF_WRAPPER_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_improvements
        tests/test_improvements.cpp
    )
    target_link_libraries(test_improvements PRIVATE tf_wrap)
    target_include_directories(test_improvements PRIVATE tests)
    
    add_test(NAME test_improvements COMMAND test_improvements)
endif()

# Install
include(GNUInstallDirs)
install(DIRECTORY include/tf_wrap DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
