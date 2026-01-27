#!/bin/bash
# run_compile_fail_tests.sh
# Runs compile-fail tests and verifies they actually fail to compile.
#
# Usage: ./run_compile_fail_tests.sh [compiler] [include_dir] [stub_dir]
#
# Each .cpp file in tests/compile_fail/ should FAIL to compile.
# The test passes if compilation fails, and fails if compilation succeeds.

set -e

COMPILER="${1:-g++}"
INCLUDE_DIR="${2:-include}"
STUB_DIR="${3:-third_party/tf_stub}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILE_FAIL_DIR="${SCRIPT_DIR}/compile_fail"

if [ ! -d "$COMPILE_FAIL_DIR" ]; then
    echo "Error: compile_fail directory not found at $COMPILE_FAIL_DIR"
    exit 1
fi

# Count tests
TOTAL=0
PASSED=0
FAILED=0

echo "=== Compile-Fail Tests ==="
echo "Compiler: $COMPILER"
echo ""

for test_file in "$COMPILE_FAIL_DIR"/*.cpp; do
    if [ ! -f "$test_file" ]; then
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    test_name=$(basename "$test_file" .cpp)
    
    echo -n "Testing $test_name... "
    
    # Try to compile - we expect this to FAIL
    if $COMPILER -std=c++20 \
        -I"$INCLUDE_DIR" \
        -I"$STUB_DIR" \
        -DTF_WRAPPER_TF_STUB_ENABLED=1 \
        "$STUB_DIR/tf_c_stub.cpp" \
        "$test_file" \
        -o /dev/null 2>/dev/null; then
        # Compilation succeeded - this is a FAILURE for compile-fail tests
        echo "FAILED (compiled successfully, but should have failed)"
        FAILED=$((FAILED + 1))
    else
        # Compilation failed - this is SUCCESS for compile-fail tests
        echo "PASSED (correctly failed to compile)"
        PASSED=$((PASSED + 1))
    fi
done

echo ""
echo "=== Results: $PASSED/$TOTAL passed ==="

if [ $FAILED -gt 0 ]; then
    echo "ERROR: $FAILED test(s) compiled successfully but should have failed"
    exit 1
fi

if [ $TOTAL -eq 0 ]; then
    echo "WARNING: No compile-fail tests found"
    exit 0
fi

echo "All compile-fail tests passed!"
exit 0
