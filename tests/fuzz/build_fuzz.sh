#!/bin/bash
# build_fuzz.sh - Build all fuzz targets
#
# Usage: ./tests/fuzz/build_fuzz.sh [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/build/fuzz}"

mkdir -p "$OUTPUT_DIR"

CXX="${CXX:-clang++}"
CXXFLAGS="-std=c++20 -g -O1 -fsanitize=fuzzer,address,undefined"
INCLUDES="-I $REPO_ROOT/include -I $REPO_ROOT/third_party/tf_stub"
DEFINES="-DTF_WRAPPER_TF_STUB_ENABLED=1"
STUB="$REPO_ROOT/third_party/tf_stub/tf_c_stub.cpp"

echo "=== Building Fuzz Targets ==="
echo "Compiler: $CXX"
echo "Output: $OUTPUT_DIR"
echo ""

# Fuzz targets that need stub
STUB_TARGETS=(
    "fuzz_tensor"
    "fuzz_session"
)

# Fuzz targets that don't need stub
STANDALONE_TARGETS=(
    "fuzz_small_vector"
)

# Build stub-based targets
for target in "${STUB_TARGETS[@]}"; do
    echo "Building $target..."
    $CXX $CXXFLAGS $INCLUDES $DEFINES \
        "$STUB" \
        "$SCRIPT_DIR/$target.cpp" \
        -o "$OUTPUT_DIR/$target"
    echo "  -> $OUTPUT_DIR/$target"
done

# Build standalone targets
for target in "${STANDALONE_TARGETS[@]}"; do
    echo "Building $target..."
    $CXX $CXXFLAGS -I "$REPO_ROOT/include" \
        "$SCRIPT_DIR/$target.cpp" \
        -o "$OUTPUT_DIR/$target"
    echo "  -> $OUTPUT_DIR/$target"
done

echo ""
echo "=== Build Complete ==="
echo "Run with: $OUTPUT_DIR/<target> [corpus_dir] [options]"
