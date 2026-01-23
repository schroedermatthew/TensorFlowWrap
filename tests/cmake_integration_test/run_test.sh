#!/bin/bash
# Run CMake integration test
# This verifies that find_package(TensorFlowWrap) works correctly

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
INSTALL_DIR="${1:-/tmp/tf_wrap_install}"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== CMake Integration Test ==="
echo "Project root: $PROJECT_ROOT"
echo "Install dir: $INSTALL_DIR"
echo ""

# Build and install the main project
echo "Step 1: Building and installing TensorFlowWrap..."
cd "$PROJECT_ROOT"
rm -rf build_integration
mkdir -p build_integration
cd build_integration

cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DTF_WRAPPER_TF_STUB=ON \
         -DTF_WRAPPER_BUILD_TESTS=OFF \
         -DTF_WRAPPER_BUILD_EXAMPLES=OFF

cmake --build . --parallel
cmake --install . --prefix "$INSTALL_DIR"
cd "$PROJECT_ROOT"

# Test the integration
echo ""
echo "Step 2: Building integration test with find_package..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
         -DCMAKE_BUILD_TYPE=Release

cmake --build .

echo ""
echo "Step 3: Running integration test..."
./integration_test

echo ""
echo "=== Integration test PASSED ==="

# Cleanup
rm -rf "$PROJECT_ROOT/build_integration"
rm -rf "$BUILD_DIR"
rm -rf "$INSTALL_DIR"
