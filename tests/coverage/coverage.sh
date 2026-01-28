#!/bin/bash
# coverage.sh - Generate code coverage report for TensorFlowWrap
#
# Usage: ./tests/coverage/coverage.sh [output_dir]
#
# Requirements:
#   - g++ with gcov support
#   - lcov
#   - genhtml (part of lcov)
#
# Install on Ubuntu:
#   sudo apt-get install lcov

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/coverage}"
BUILD_DIR="$REPO_ROOT/build/coverage"

echo "=== TensorFlowWrap Code Coverage ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Compiler flags for coverage
CXX="g++"
CXXFLAGS="-std=c++20 -O0 -g --coverage -fprofile-arcs -ftest-coverage"
INCLUDES="-I$REPO_ROOT/include -I$REPO_ROOT/third_party/tf_stub"
DEFINES="-DTF_WRAPPER_TF_STUB_ENABLED=1"
STUB="$REPO_ROOT/third_party/tf_stub/tf_c_stub.cpp"

# Clean previous coverage data
echo "Cleaning previous coverage data..."
find "$BUILD_DIR" -name "*.gcda" -delete 2>/dev/null || true
find "$BUILD_DIR" -name "*.gcno" -delete 2>/dev/null || true

# Build and run tests with coverage
echo "Building tests with coverage instrumentation..."

TEST_FILES=(
    "test_tensor.cpp"
    "test_session.cpp"
    "test_scope_guard.cpp"
    "test_small_vector.cpp"
    "test_facade.cpp"
    "test_graph.cpp"
    "test_status.cpp"
    "test_error.cpp"
    "test_format.cpp"
    "test_operation.cpp"
    "test_lifecycle.cpp"
)

cd "$BUILD_DIR"

for test_file in "${TEST_FILES[@]}"; do
    test_name="${test_file%.cpp}"
    echo "  Building $test_name..."
    
    $CXX $CXXFLAGS $INCLUDES $DEFINES \
        "$STUB" \
        "$REPO_ROOT/tests/$test_file" \
        -o "$test_name"
done

echo ""
echo "Running tests..."

for test_file in "${TEST_FILES[@]}"; do
    test_name="${test_file%.cpp}"
    echo "  Running $test_name..."
    "./$test_name" > /dev/null 2>&1 || echo "    Warning: $test_name had failures"
done

echo ""
echo "Generating coverage report..."

# Capture coverage data
lcov --capture \
    --directory "$BUILD_DIR" \
    --output-file "$BUILD_DIR/coverage.info" \
    --ignore-errors source \
    --quiet

# Filter to only include our headers
lcov --extract "$BUILD_DIR/coverage.info" \
    "$REPO_ROOT/include/*" \
    --output-file "$BUILD_DIR/coverage_filtered.info" \
    --quiet

# Remove test files and stub from coverage
lcov --remove "$BUILD_DIR/coverage_filtered.info" \
    "*/third_party/*" \
    "*/tests/*" \
    --output-file "$BUILD_DIR/coverage_final.info" \
    --quiet

# Generate HTML report
genhtml "$BUILD_DIR/coverage_final.info" \
    --output-directory "$OUTPUT_DIR" \
    --title "TensorFlowWrap Code Coverage" \
    --legend \
    --show-details \
    --quiet

# Generate summary
echo ""
echo "=== Coverage Summary ==="
lcov --summary "$BUILD_DIR/coverage_final.info" 2>&1 | grep -E "lines|functions"

echo ""
echo "HTML report: $OUTPUT_DIR/index.html"
echo ""
echo "=== Coverage Complete ==="
