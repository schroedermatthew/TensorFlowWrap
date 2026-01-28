# TensorFlowWrap CI Workflow Style Guide

**Status:** Active  
**Applies to:** `.github/workflows/ci.yml`  
**Authority:** Subordinate to the *TensorFlowWrap Development Guidelines*  
**Version:** 2.0 (January 2026)

---

## 1. Purpose

This guide standardizes GitHub Actions CI workflows for TensorFlowWrap. Consistent workflows ensure:

- Uniform quality gates across all platforms
- Predictable CI behavior for contributors
- Validation with both stub and real TensorFlow
- **C++20 requirement enforcement**

---

## 2. Directory Structure

```
TensorFlowWrap/
├── include/tf_wrap/           # Public headers (.hpp)
├── tests/                     # Test files
│   ├── test_*.cpp             # Stub tests (doctest)
│   ├── test_*_tf.cpp          # Real TF tests
│   ├── compile_fail/          # Expected-fail tests
│   ├── fuzz/                  # Fuzz testing targets
│   ├── benchmark/             # Performance benchmarks
│   └── coverage/              # Coverage scripts
├── third_party/tf_stub/       # TensorFlow C API stub
├── guidelines/                # Style guides
└── .github/workflows/ci.yml   # CI workflow
```

**Critical:** Always use these paths in workflows. Use environment variables for consistency.

---

## 3. CI Goals

| Goal | Implementation |
|------|----------------|
| Multi-platform build | GCC, Clang, MSVC, Apple Clang |
| Multi-compiler version | GCC 13/14, Clang 17/18 |
| Stub testing | All platforms with TF stub |
| Real TF testing | Linux with TF 2.13-2.18 |
| Memory safety | ASan + UBSan + soak tests |
| Thread safety | Concurrent inference tests |
| Fuzzing | libFuzzer targets |
| Performance | Benchmark suite |
| Coverage | lcov reporting |
| Header hygiene | Standalone compile check |

---

## 4. Required Jobs

Every CI run MUST include these jobs:

| Job | Purpose | Required |
|-----|---------|----------|
| `header-check` | Verify headers compile standalone | ✅ |
| `cmake-test` | CMake build, install, find_package test | ✅ |
| `linux-gcc` | GCC 13/14 (C++20) build + all stub tests | ✅ |
| `linux-clang` | Clang 17/18 (C++20) build + all stub tests | ✅ |
| `windows-msvc` | MSVC (C++20) build + all stub tests | ✅ |
| `macos` | Apple Clang (C++20) build + all stub tests | ✅ |
| `sanitizers` | ASan + UBSan | ✅ |
| `soak-tests` | Long-running stress tests with ASan | ✅ |
| `thread-safety` | Concurrent inference tests | ✅ |
| `real-tensorflow` | Real TF 2.13-2.18 integration tests | ✅ |
| `fuzz-tests` | Fuzz testing with libFuzzer | ✅ |
| `benchmarks` | Performance benchmarks | ✅ |
| `coverage` | Code coverage reporting | ✅ |
| `ci-success` | Gate job aggregating all results | ✅ |

---

## 5. Test Categories

### 5.1 Stub Tests (All Platforms)

Stub tests use doctest and run with the TF stub on all platforms.

**Files:** `tests/test_*.cpp` (excluding `*_tf.cpp`)

```yaml
- name: Build and run Tensor tests (stub)
  run: |
    g++-${{ matrix.version }} -std=c++20 -O2 \
      -Wall -Wextra -Wpedantic -Werror \
      -I${{ env.INCLUDE_DIR }} \
      -I${{ env.STUB_DIR }} \
      -DTF_WRAPPER_TF_STUB_ENABLED=1 \
      ${{ env.STUB_DIR }}/tf_c_stub.cpp \
      ${{ env.TEST_DIR }}/test_tensor.cpp \
      -o test_tensor
    ./test_tensor
```

### 5.2 Real TensorFlow Tests (Linux Only)

Real TF tests use a custom framework and run only on Linux with real TensorFlow.

**Files:** `tests/test_*_tf.cpp`

```yaml
- name: Build and run tests
  run: |
    export LD_LIBRARY_PATH="$(pwd)/tensorflow_c/lib:$LD_LIBRARY_PATH"
    
    g++-12 -std=c++20 -O2 \
      -Wall -Wextra -Wpedantic \
      -I${{ env.INCLUDE_DIR }} \
      -Itensorflow_c/include \
      ${{ env.TEST_DIR }}/test_tensor_tf.cpp \
      -Ltensorflow_c/lib \
      -ltensorflow \
      -o test_tensor_tf
    
    ./test_tensor_tf
```

### 5.3 Header Self-Containment

Every public header must compile standalone.

```yaml
- name: Check each header compiles standalone
  run: |
    for header in include/tf_wrap/*.hpp; do
      echo "Checking $header..."
      echo "#include \"$header\"" > /tmp/check.cpp
      echo "int main() { return 0; }" >> /tmp/check.cpp
      g++-13 -std=c++20 -fsyntax-only \
        -I include -I third_party/tf_stub \
        -DTF_WRAPPER_TF_STUB_ENABLED=1 \
        /tmp/check.cpp
    done
```

### 5.4 Compile-Fail Tests

Tests that are expected to fail compilation, validating type safety.

```yaml
- name: Compile-fail tests
  run: |
    for f in tests/compile_fail/*.cpp; do
      echo "Testing $f (should fail)..."
      if g++ -std=c++20 -I include -c "$f" -o /dev/null 2>/dev/null; then
        echo "ERROR: $f compiled but should have failed!"
        exit 1
      fi
      echo "  ✓ Correctly failed to compile"
    done
```

---

## 6. Compiler Matrix

### 6.1 Linux GCC

| Version | Runner | Notes |
|---------|--------|-------|
| GCC 13 | ubuntu-24.04 | Primary |
| GCC 14 | ubuntu-24.04 | Latest |

### 6.2 Linux Clang

| Version | Runner | Notes |
|---------|--------|-------|
| Clang 17 | ubuntu-22.04 | |
| Clang 18 | ubuntu-22.04 | Primary |

### 6.3 Windows MSVC

| Standard | Runner | Notes |
|----------|--------|-------|
| C++20 | windows-latest | Uses `ilammy/msvc-dev-cmd@v1` |

### 6.4 macOS

| Compiler | Runner | Notes |
|----------|--------|-------|
| Apple Clang | macos-14 | ARM64 |

### 6.5 Real TensorFlow

| TF Version | Runner | Notes |
|------------|--------|-------|
| 2.13.0 | ubuntu-22.04 | Minimum supported |
| 2.14.0 | ubuntu-22.04 | |
| 2.15.0 | ubuntu-22.04 | |
| 2.16.1 | ubuntu-22.04 | |
| 2.17.1 | ubuntu-22.04 | |
| 2.18.1 | ubuntu-22.04 | Latest |

**Note:** TF 2.17.0 is excluded due to missing TSL headers in the C library.

---

## 7. Environment Variables

Define these at the workflow level:

```yaml
env:
  INCLUDE_DIR: include
  TEST_DIR: tests
  STUB_DIR: third_party/tf_stub
```

---

## 8. Warning Flags

### 8.1 GCC/Clang

```bash
-Wall -Wextra -Wpedantic -Werror
```

### 8.2 Clang Additional

```bash
-Wno-gnu-zero-variadic-macro-arguments  # Suppress doctest warning
```

### 8.3 MSVC

```
/W4 /WX /EHsc
```

---

## 9. Gate Job Pattern

The `ci-success` job aggregates all results:

```yaml
ci-success:
  name: CI Success
  runs-on: ubuntu-latest
  needs: [header-check, linux-gcc, linux-clang, windows-msvc, macos, sanitizers, real-tensorflow]
  if: always()
  steps:
    - name: Check all jobs passed
      run: |
        results=(
          "${{ needs.header-check.result }}"
          "${{ needs.linux-gcc.result }}"
          "${{ needs.linux-clang.result }}"
          "${{ needs.windows-msvc.result }}"
          "${{ needs.macos.result }}"
          "${{ needs.sanitizers.result }}"
          "${{ needs.soak-tests.result }}"
          "${{ needs.thread-safety.result }}"
          "${{ needs.real-tensorflow.result }}"
          "${{ needs.fuzz-tests.result }}"
          "${{ needs.benchmarks.result }}"
          "${{ needs.coverage.result }}"
        )
        
        for result in "${results[@]}"; do
          if [[ "$result" != "success" ]]; then
            echo "❌ One or more jobs failed"
            exit 1
          fi
        done
        
        echo "✓ All CI jobs passed"
```

---

## 10. Sanitizer Configuration

### 10.1 AddressSanitizer

```yaml
- name: Build and run with AddressSanitizer
  env:
    ASAN_OPTIONS: detect_leaks=1:abort_on_error=1
  run: |
    g++-13 -std=c++20 -g \
      -fsanitize=address -fno-omit-frame-pointer \
      -I${{ env.INCLUDE_DIR }} \
      -I${{ env.STUB_DIR }} \
      -DTF_WRAPPER_TF_STUB_ENABLED=1 \
      ${{ env.STUB_DIR }}/tf_c_stub.cpp \
      ${{ env.TEST_DIR }}/test_tensor.cpp \
      -o test_asan
    ./test_asan
```

### 10.2 UndefinedBehaviorSanitizer

```yaml
- name: Build and run with UndefinedBehaviorSanitizer
  env:
    UBSAN_OPTIONS: print_stacktrace=1:abort_on_error=1
  run: |
    g++-13 -std=c++20 -g \
      -fsanitize=undefined -fno-omit-frame-pointer \
      ...
```

---

## 11. TensorFlow C Library Download

```yaml
- name: Download TensorFlow C library
  run: |
    TF_VERSION=${{ matrix.tf_version }}
    curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-${TF_VERSION}.tar.gz" | tar xz -C tensorflow_c
```

---

## 12. Checklist for CI Changes

Before modifying the CI workflow:

- [ ] All environment variables defined in `env:` block
- [ ] CMake build, install, and find_package test included
- [ ] Stub tests run on all platforms (GCC, Clang, MSVC, macOS)
- [ ] Real TF tests run on Linux only
- [ ] Header standalone check included
- [ ] Sanitizers (ASan + UBSan) included
- [ ] Soak tests with ASan included
- [ ] Thread safety tests included
- [ ] Fuzz tests included
- [ ] Benchmarks included
- [ ] Coverage reporting included
- [ ] All TF versions 2.13-2.18 tested
- [ ] `ci-success` gate job checks all 13 required jobs
- [ ] Warning flags include `-Werror` / `/WX`
- [ ] New tests added to all platforms (Corner Cases, Comprehensive, Adversarial)
- [ ] Tested locally before pushing

---

## Changelog

### v2.0 (January 2026)
- Added cmake-test job for CMake build/install/find_package validation
- Added soak-tests, thread-safety, fuzz-tests, benchmarks, coverage jobs
- Added corner cases, comprehensive, adversarial test files
- Updated test file references (removed deleted test_real_tf_minimal.cpp)
- Compile-fail tests now implemented
- Total CI jobs: 13

### v1.0 (January 2026)
- Initial CI Workflow Style Guide for TensorFlowWrap

---

*TensorFlowWrap CI Workflow Style Guide v2.0 -- January 2026*
