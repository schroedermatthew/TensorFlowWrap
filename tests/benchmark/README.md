# TensorFlowWrap Benchmarks

Performance benchmarks for TensorFlowWrap core operations.

## Building

```bash
# With stub (for development)
g++ -std=c++20 -O3 -DNDEBUG \
    -I include -I third_party/tf_stub \
    -DTF_WRAPPER_TF_STUB_ENABLED=1 \
    third_party/tf_stub/tf_c_stub.cpp \
    tests/benchmark/benchmark.cpp \
    -o benchmark

# With real TensorFlow
g++ -std=c++20 -O3 -DNDEBUG \
    -I include -I tensorflow_c/include \
    tests/benchmark/benchmark.cpp \
    -L tensorflow_c/lib -ltensorflow \
    -Wl,-rpath,tensorflow_c/lib \
    -pthread -o benchmark
```

## Running

```bash
# Default (10000 iterations, 1000 warmup)
./benchmark

# Custom iterations
./benchmark --iterations 50000 --warmup 5000

# JSON output (for CI comparison)
./benchmark --json > results.json
```

## Benchmarks

### Tensor Operations
- `Tensor::FromScalar<T>` - Scalar tensor creation
- `Tensor::FromVector<T>[N]` - Vector tensor creation (various sizes)
- `Tensor::Zeros<T>[N]` - Zero-initialized tensor
- `Tensor::Clone` - Deep copy
- `Tensor::read<T>` - Read view acquisition
- `Tensor::ToScalar<T>` - Scalar extraction
- `Tensor::ToVector<T>` - Vector extraction
- `Tensor::FromString` - String tensor creation

### SmallVector Operations
- `push_back` (inline vs heap spill)
- `reserve` + bulk insert
- Copy and move operations
- Comparison with `std::vector`

### Session/Graph Operations
- `Graph` construction
- `Session` construction
- `SessionOptions` construction
- `Status` construction
- `Buffer` construction

## CI Integration

The benchmark job runs on every push and compares against the previous run:

```yaml
- name: Run benchmarks
  run: |
    ./benchmark --json > current.json
    # Compare with baseline (if exists)
    if [ -f baseline.json ]; then
      python3 tests/benchmark/compare.py baseline.json current.json
    fi
```

## Interpreting Results

| Metric | Description |
|--------|-------------|
| Mean | Average time per operation |
| Median | 50th percentile (less sensitive to outliers) |
| StdDev | Standard deviation (consistency measure) |
| Throughput | Operations per second |

Lower times and higher throughput are better. Watch for:
- Large stddev (inconsistent performance)
- Regression vs baseline (>10% slower)
