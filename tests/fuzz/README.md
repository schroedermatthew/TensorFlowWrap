# TensorFlowWrap Fuzz Tests

This directory contains fuzz tests for TensorFlowWrap using [libFuzzer](https://llvm.org/docs/LibFuzzer.html).

## Prerequisites

- Clang with libFuzzer support (Clang 6.0+)
- AddressSanitizer and UndefinedBehaviorSanitizer

## Building

```bash
# Build all fuzz targets
./tests/fuzz/build_fuzz.sh

# Or build individually:
clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address,undefined \
    -I include -I third_party/tf_stub \
    -DTF_WRAPPER_TF_STUB_ENABLED=1 \
    third_party/tf_stub/tf_c_stub.cpp \
    tests/fuzz/fuzz_tensor.cpp \
    -o fuzz_tensor
```

## Running

```bash
# Run with default settings (runs indefinitely)
./fuzz_tensor

# Run with corpus directory
mkdir -p corpus/tensor
./fuzz_tensor corpus/tensor/

# Run with limits
./fuzz_tensor corpus/tensor/ -max_len=1024 -runs=100000 -max_total_time=60

# Run all fuzz targets
./tests/fuzz/run_fuzz.sh
```

## Fuzz Targets

| Target | Component | What It Tests |
|--------|-----------|---------------|
| `fuzz_tensor` | Tensor | FromScalar, FromVector, FromString, reshape, move semantics |
| `fuzz_small_vector` | SmallVector | push/pop, insert/erase, resize, copy/move |
| `fuzz_session` | Session/Graph | Construction, options, buffer, operations |

## Corpus

The `corpus/` directory stores interesting inputs discovered by the fuzzer. These inputs
are reused across runs to improve coverage. Commit the corpus to preserve discoveries.

## Finding Bugs

When a bug is found, libFuzzer will:
1. Print the crash/error details
2. Save the input to `crash-<hash>` or `timeout-<hash>`

To reproduce:
```bash
./fuzz_tensor crash-abc123
```

## CI Integration

Fuzz tests run in CI with limited iterations to catch regressions:
```yaml
- name: Fuzz tests (quick)
  run: |
    ./fuzz_tensor corpus/tensor/ -runs=10000 -max_total_time=30
    ./fuzz_small_vector corpus/small_vector/ -runs=10000 -max_total_time=30
```

For thorough fuzzing, run locally for extended periods.

## Coverage

To see what code paths are covered:
```bash
# Build with coverage
clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address \
    -fprofile-instr-generate -fcoverage-mapping \
    -I include -I third_party/tf_stub \
    -DTF_WRAPPER_TF_STUB_ENABLED=1 \
    third_party/tf_stub/tf_c_stub.cpp \
    tests/fuzz/fuzz_tensor.cpp \
    -o fuzz_tensor_cov

# Run
./fuzz_tensor_cov corpus/tensor/ -runs=10000

# Generate report
llvm-profdata merge -sparse default.profraw -o default.profdata
llvm-cov show ./fuzz_tensor_cov -instr-profile=default.profdata
```
