cmake_minimum_required(VERSION 3.16)
project(TensorFlowWrap VERSION 1.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type for single-config generators (Ninja, Makefiles).
# Multi-config generators (Visual Studio, Xcode) ignore CMAKE_BUILD_TYPE.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

# Canonical option name (matches README + CI scripts)
option(TF_WRAPPER_TF_STUB "Use the TensorFlow C API stub instead of a real TF install (tests only)" ON)

# Deprecated alias (kept for back-compat with older builds)
option(TF_WRAPPER_USE_STUB "DEPRECATED: use TF_WRAPPER_TF_STUB instead" OFF)
if(TF_WRAPPER_USE_STUB)
    message(WARNING "TF_WRAPPER_USE_STUB is deprecated; use TF_WRAPPER_TF_STUB instead.")
    set(TF_WRAPPER_TF_STUB ON)
endif()

option(TF_WRAPPER_BUILD_TESTS "Build tests" ON)

# -----------------------------------------------------------------------------
# Stub library (optional)
# -----------------------------------------------------------------------------

if(TF_WRAPPER_TF_STUB)
    add_library(tf_stub STATIC
        third_party/tf_stub/tf_c_stub.cpp
    )
    add_library(tf::stub ALIAS tf_stub)
    set_target_properties(tf_stub PROPERTIES EXPORT_NAME stub)

    target_include_directories(tf_stub PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/tf_stub>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_compile_definitions(tf_stub PUBLIC TF_WRAPPER_TF_STUB_ENABLED)
endif()

# -----------------------------------------------------------------------------
# Main header-only library target
# -----------------------------------------------------------------------------

add_library(tf_wrap INTERFACE)
add_library(tf::wrapper ALIAS tf_wrap)
set_target_properties(tf_wrap PROPERTIES EXPORT_NAME wrapper)

target_include_directories(tf_wrap INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(tf_wrap INTERFACE cxx_std_20)

if(TF_WRAPPER_TF_STUB)
    target_link_libraries(tf_wrap INTERFACE tf_stub)
    target_compile_definitions(tf_wrap INTERFACE TF_WRAPPER_TF_STUB_ENABLED)
endif()

# -----------------------------------------------------------------------------
# Tests (minimal CMake test target)
# -----------------------------------------------------------------------------

if(TF_WRAPPER_BUILD_TESTS)
    enable_testing()

    add_executable(test_improvements
        tests/test_improvements.cpp
    )
    target_link_libraries(test_improvements PRIVATE tf_wrap)
    target_include_directories(test_improvements PRIVATE tests)

    add_test(NAME test_improvements COMMAND test_improvements)
endif()

# -----------------------------------------------------------------------------
# Install + package config (find_package(TensorFlowWrap))
# -----------------------------------------------------------------------------

# Wrapper headers
install(DIRECTORY include/tf_wrap DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Stub headers (only when stub is enabled)
# NOTE: These are intentionally NOT a full TF C API and are only intended for
# wrapper unit tests.
if(TF_WRAPPER_TF_STUB)
    install(DIRECTORY third_party/tf_stub/tensorflow DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# Targets
install(TARGETS tf_wrap
    EXPORT TensorFlowWrapTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TF_WRAPPER_TF_STUB)
    install(TARGETS tf_stub
        EXPORT TensorFlowWrapTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(EXPORT TensorFlowWrapTargets
    FILE TensorFlowWrap-targets.cmake
    NAMESPACE tf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorFlowWrap
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorFlowWrap-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFlowWrap-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorFlowWrap
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFlowWrap-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFlowWrap-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFlowWrap-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorFlowWrap
)
