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

    # ScopeGuard tests (no TF dependency)
    add_executable(test_scope_guard
        tests/test_scope_guard.cpp
    )
    target_include_directories(test_scope_guard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_scope_guard COMMAND test_scope_guard)

    # SmallVector tests (no TF dependency)
    add_executable(test_small_vector
        tests/test_small_vector.cpp
    )
    target_include_directories(test_small_vector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_small_vector COMMAND test_small_vector)
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

# Package config template (support both historical and conventional filenames)
set(_tfwrap_config_in "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorFlowWrap-config.cmake.in")
if(NOT EXISTS "${_tfwrap_config_in}")
    set(_tfwrap_config_in "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorFlowWrapConfig.cmake.in")
endif()
if(NOT EXISTS "${_tfwrap_config_in}")
    message(FATAL_ERROR
        "TensorFlowWrap package config template not found. Expected one of:\n"
        "  - cmake/TensorFlowWrap-config.cmake.in\n"
        "  - cmake/TensorFlowWrapConfig.cmake.in")
endif()

configure_package_config_file(
    "${_tfwrap_config_in}"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorFlowWrapConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TensorFlowWrap"
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFlowWrapConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFlowWrapConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TensorFlowWrapConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorFlowWrap
)
