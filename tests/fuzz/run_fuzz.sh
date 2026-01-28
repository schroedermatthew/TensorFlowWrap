#!/bin/bash
# run_fuzz.sh - Run all fuzz targets with default settings
#
# Usage: ./tests/fuzz/run_fuzz.sh [runs] [max_time]
#
# Arguments:
#   runs     - Number of runs per target (default: 10000)
#   max_time - Maximum time in seconds per target (default: 60)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/fuzz"
CORPUS_DIR="$REPO_ROOT/corpus"

RUNS="${1:-10000}"
MAX_TIME="${2:-60}"

# Build first
"$SCRIPT_DIR/build_fuzz.sh" "$BUILD_DIR"

echo ""
echo "=== Running Fuzz Tests ==="
echo "Runs per target: $RUNS"
echo "Max time per target: ${MAX_TIME}s"
echo ""

TARGETS=(
    "fuzz_tensor"
    "fuzz_small_vector"
    "fuzz_session"
)

FAILED=0

for target in "${TARGETS[@]}"; do
    target_corpus="$CORPUS_DIR/$target"
    mkdir -p "$target_corpus"
    
    echo "--- $target ---"
    
    if "$BUILD_DIR/$target" "$target_corpus/" \
        -runs="$RUNS" \
        -max_total_time="$MAX_TIME" \
        -max_len=1024 \
        2>&1; then
        echo "✓ $target completed"
    else
        echo "✗ $target FAILED"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

echo "=== Results ==="
if [ $FAILED -eq 0 ]; then
    echo "All fuzz targets completed successfully"
    exit 0
else
    echo "$FAILED target(s) failed"
    exit 1
fi
