# TensorFlowWrap Test Suite Style Guide

## Purpose

This guide ensures consistent, thorough test suites across all TensorFlowWrap components. Tests are the **executable specification** of the library -- they document behavior, catch regressions, and prove correctness.

## Test Framework

**We use [doctest](https://github.com/doctest/doctest)** -- a fast, header-only C++ testing framework. It provides:
- `TEST_CASE("name")` for defining tests
- `CHECK()`, `CHECK_FALSE()`, `REQUIRE()` for assertions
- `SUBCASE()` for test organization
- Automatic test registration (no manual test lists)

The only exception is `*_tf.cpp` files (real TensorFlow tests), which use a minimal custom framework to avoid `[[nodiscard]]` conflicts with `-Werror`.

## Dual Testing Strategy

TensorFlowWrap has a unique testing challenge: the library wraps TensorFlow's C API, which means tests must work both with a **stub implementation** (for fast CI on all platforms) and with **real TensorFlow** (for integration validation).

| Test Type | Framework | Runs With | Purpose |
|-----------|-----------|-----------|---------|
| `test_*.cpp` | doctest | Stub | API surface, error handling, edge cases |
| `test_*_tf.cpp` | Custom | Real TF | Actual TF execution, result verification |

---

## Test File Organization

### Directory Structure

```
tests/
├── doctest.h                    # Test framework (header-only)
├── test_small_vector.cpp        # Utility: SmallVector (stub, doctest)
├── test_scope_guard.cpp         # Utility: ScopeGuard (stub, doctest)
├── test_tensor.cpp              # Tensor API (stub, doctest)
├── test_session.cpp             # Session API (stub, doctest)
├── test_facade.cpp              # Facade API (stub, doctest)
├── test_graph.cpp               # Graph API (stub, doctest)
├── test_operation.cpp           # Operation API (stub, doctest)
├── test_status.cpp              # Status API (stub, doctest)
├── test_error.cpp               # Error API (stub, doctest)
├── test_format.cpp              # Format utilities (stub, doctest)
├── test_lifecycle.cpp           # RAII lifecycle tracking (stub, doctest)
├── test_corner_cases.cpp        # Boundary/edge case tests (stub, doctest)
├── test_comprehensive.cpp       # API coverage tests (stub, doctest)
├── test_adversarial.cpp         # Stress/attack tests (stub, doctest)
├── test_tensor_tf.cpp           # Tensor API (real TF)
├── test_session_tf.cpp          # Session API (real TF)
├── test_facade_tf.cpp           # Facade API (real TF)
├── test_graph_tf.cpp            # Graph API (real TF)
├── test_operation_tf.cpp        # Operation API (real TF)
├── test_soak_tf.cpp             # Soak/stress tests (real TF + ASan)
├── test_thread_safety_tf.cpp    # Concurrent inference (real TF)
├── test_comprehensive_tf.cpp    # BatchRun, RunContext (real TF)
├── test_adversarial_tf.cpp      # Stress/attack tests (real TF)
├── compile_fail/                # Expected-fail compilation tests
│   ├── compile_fail_tensor_invalid_dtype.cpp
│   └── ...
├── fuzz/                        # Fuzz testing
│   ├── fuzz_tensor.cpp
│   ├── fuzz_small_vector.cpp
│   ├── fuzz_session.cpp
│   └── README.md
├── benchmark/                   # Performance benchmarks
│   ├── benchmark.cpp
│   ├── compare.py
│   └── README.md
└── coverage/                    # Code coverage
    ├── coverage.sh
    └── README.md
```

### File Naming Conventions

| Pattern | Purpose | Runs With |
|---------|---------|-----------|
| `test_<component>.cpp` | Comprehensive stub tests (doctest) | Stub only |
| `test_<component>_tf.cpp` | Real TensorFlow tests | Real TF only |
| `compile_fail_<component>_<reason>.cpp` | Expected compilation failures | Neither (compile-only) |

---

## Test File Templates

### Stub Tests (doctest) - Primary Test Files

All stub tests use **doctest**. These are the primary, comprehensive tests that run on all platforms.

```cpp
// test_component.cpp
// Comprehensive tests for tf_wrap::Component
//
// Framework: doctest
// Runs with: Stub TensorFlow (all platforms)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/component.hpp"

using namespace tf_wrap;

// ============================================================================
// Construction Tests
// ============================================================================

TEST_CASE("Component - default construction") {
    Component c;
    CHECK(c.valid());
    CHECK(c.handle() != nullptr);
}

TEST_CASE("Component - move constructor") {
    Component c1;
    auto* handle = c1.handle();
    
    Component c2(std::move(c1));
    CHECK(c2.handle() == handle);
    CHECK_FALSE(c1.valid());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_CASE("Component - operation on moved-from throws") {
    Component c1;
    Component c2(std::move(c1));
    
    // Manual try-catch for [[nodiscard]] functions (see Assertion Guidelines)
    bool threw = false;
    try {
        auto result = c1.some_operation();
        (void)result;
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
}
```

### Real TensorFlow Tests (Custom Framework)

Real TF tests use a **minimal custom framework** instead of doctest. This is necessary because:
1. doctest's `CHECK_THROWS` macros conflict with `[[nodiscard]]` under `-Werror`
2. Real TF headers can trigger warnings that doctest macros expose
3. Simpler diagnostics are sufficient for integration tests

These tests only run on Linux CI with real TensorFlow installed.

```cpp
// test_component_tf.cpp
// Component tests with real TensorFlow C library
//
// Framework: Custom (see below)
// Runs with: Real TensorFlow (Linux CI only)

#include "tf_wrap/component.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace tf_wrap;

// ============================================================================
// Test Framework
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            std::cout << "Testing " #name "... " << std::flush; \
            tests_run++; \
            try { \
                test_##name(); \
                std::cout << "PASSED\n"; \
                tests_passed++; \
            } catch (const std::exception& e) { \
                std::cout << "FAILED: " << e.what() << "\n"; \
            } catch (...) { \
                std::cout << "FAILED: unknown exception\n"; \
            } \
        } \
    } test_runner_##name; \
    void test_##name()

#define REQUIRE(cond) \
    do { if (!(cond)) throw std::runtime_error("REQUIRE failed: " #cond); } while (0)

#define REQUIRE_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error("REQUIRE_CLOSE failed: " #a " vs " #b); \
        } \
    } while (0)

#define REQUIRE_THROWS(expr) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (...) { threw = true; } \
        if (!threw) throw std::runtime_error("REQUIRE_THROWS failed: " #expr); \
    } while (0)

// ============================================================================
// Tests
// ============================================================================

TEST(basic_construction) {
    Component c;
    REQUIRE(c.valid());
}

TEST(move_semantics) {
    Component c1;
    Component c2(std::move(c1));
    REQUIRE(c2.valid());
    REQUIRE(!c1.valid());
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TensorFlowWrap Component Tests (Real TF) ===\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    return (tests_passed == tests_run) ? 0 : 1;
}
```

### Compile-Fail Tests

These verify that invalid code fails to compile (type safety enforcement).

```cpp
// tests/compile_fail/compile_fail_tensor_invalid_dtype.cpp
// Expected-fail: std::string is not a valid TensorScalar type
//
// This file must FAIL to compile. If it compiles, the TensorScalar
// concept is not correctly rejecting invalid types.

#include "tf_wrap/tensor.hpp"

int main() {
    // std::string does not satisfy TensorScalar concept
    // This should trigger a compile error
    auto t = tf_wrap::Tensor::FromScalar<std::string>("hello");
    (void)t;
    return 0;
}
```

---

## Test Categories

### Required Coverage

Every component test suite should cover these areas:

| Category | What to Test | Priority |
|----------|--------------|----------|
| **Construction** | Default, parameterized, factory methods | P0 |
| **Destruction** | RAII cleanup, no leaks | P0 |
| **Move semantics** | Move ctor, move assign, moved-from state | P0 |
| **Core operations** | All public methods | P0 |
| **Error handling** | Invalid inputs, null handles, moved-from | P0 |
| **Type safety** | dtype mismatches, concept enforcement | P1 |
| **Edge cases** | Empty, single element, large sizes | P1 |
| **Exception safety** | Throwing during operations | P2 |
| **Lifecycle tracking** | Constructor/destructor balance | P2 |
| **Stress testing** | Repeated operations, memory stability | P2 |

### Component-Specific Coverage

#### Tensor

```cpp
// Factory methods
TEST(from_scalar_all_dtypes)      // float, double, int8..int64, uint8..uint64, bool, complex
TEST(from_vector_shapes)          // 1D, 2D, 3D, high-rank
TEST(from_vector_shape_mismatch)  // Should throw
TEST(zeros_and_allocate)
TEST(from_string)
TEST(clone_independence)
TEST(adopt_malloc)

// Data access
TEST(read_view)
TEST(write_view)
TEST(to_scalar)
TEST(to_vector)
TEST(view_keeps_tensor_alive)     // Critical lifetime test

// Shape operations
TEST(reshape_valid)
TEST(reshape_element_mismatch)    // Should throw
TEST(matches_shape)

// Type safety
TEST(dtype_mismatch_throws)       // read<int> on float tensor
```

#### Session

```cpp
// Construction
TEST(session_from_graph)
TEST(session_with_options)
TEST(load_savedmodel)
TEST(load_savedmodel_not_found)   // Should throw

// Operations
TEST(resolve_operation)
TEST(resolve_not_found)           // Should throw
TEST(run_inference)               // Requires real TF or graph with ops
TEST(list_devices)

// Resource management
TEST(session_move)
TEST(operations_on_moved_from)    // Should throw
```

#### Graph

```cpp
TEST(graph_construction)
TEST(get_operation)
TEST(get_operation_not_found)
TEST(get_all_operations)
TEST(graph_move)
```

---

## Assertion Guidelines

### doctest Assertions (Primary)

doctest provides these assertion macros. Use `CHECK` for non-fatal assertions (test continues on failure) and `REQUIRE` for fatal assertions (test stops on failure).

| Assertion | Use For |
|-----------|---------|
| `CHECK(cond)` | Boolean conditions (non-fatal) |
| `REQUIRE(cond)` | Boolean conditions (fatal) |
| `CHECK(a == b)` | Value equality |
| `CHECK_FALSE(cond)` | Expected false |
| `CHECK(a == doctest::Approx(b))` | Floating-point comparison |
| `CHECK_THROWS(expr)` | Expected exception (any type) |
| `CHECK_THROWS_AS(expr, type)` | Expected specific exception |
| `SUBCASE("name")` | Test sub-sections |

### The `[[nodiscard]]` Problem

Many TensorFlowWrap functions are marked `[[nodiscard]]`. When combined with `-Werror`, doctest's `CHECK_THROWS` macros cause compilation failures because they internally cast to `void`.

**Problem:**
```cpp
// FAILS TO COMPILE with -Werror=unused-result
CHECK_THROWS_AS(tensor.read<int>(), std::runtime_error);
```

**Solution:** Use manual try-catch for `[[nodiscard]]` functions:

```cpp
TEST_CASE("dtype mismatch throws") {
    auto t = Tensor::FromScalar<float>(1.0f);
    
    bool threw = false;
    try {
        auto v = t.read<int>();  // Wrong dtype - [[nodiscard]]
        (void)v;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}
```

This pattern is verbose but necessary for correct compilation under strict warnings.

### Real TF Test Assertions (Custom Framework)

The `*_tf.cpp` files use these custom macros instead of doctest:

```cpp
#define REQUIRE(cond) \
    do { if (!(cond)) throw std::runtime_error("REQUIRE failed: " #cond); } while (0)

#define REQUIRE_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error("REQUIRE_CLOSE failed"); \
        } \
    } while (0)

#define REQUIRE_THROWS(expr) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (...) { threw = true; } \
        if (!threw) throw std::runtime_error("REQUIRE_THROWS failed: " #expr); \
    } while (0)
```

Note the `(void)(expr)` in `REQUIRE_THROWS` -- this silences `[[nodiscard]]` warnings.

---

## Helper Types

### Lifecycle Tracker

Verify RAII correctness by counting constructor/destructor calls:

```cpp
class LifecycleTracker {
public:
    static inline int construct_count = 0;
    static inline int destruct_count = 0;
    
    int value;
    
    explicit LifecycleTracker(int v = 0) : value(v) { ++construct_count; }
    LifecycleTracker(const LifecycleTracker& o) : value(o.value) { ++construct_count; }
    LifecycleTracker(LifecycleTracker&& o) noexcept : value(o.value) { ++construct_count; }
    ~LifecycleTracker() { ++destruct_count; }
    
    static void reset() { construct_count = destruct_count = 0; }
    static bool balanced() { return construct_count == destruct_count; }
};

TEST_CASE("SmallVector - no leaks") {
    LifecycleTracker::reset();
    {
        SmallVector<LifecycleTracker, 4> v;
        v.push_back(LifecycleTracker(1));
        v.push_back(LifecycleTracker(2));
    }
    CHECK(LifecycleTracker::balanced());
}
```

### Exception Injector

Test exception safety:

```cpp
struct ThrowOnCopy {
    int value;
    static inline int throw_after = -1;
    static inline int copy_count = 0;
    
    ThrowOnCopy(int v) : value(v) {}
    ThrowOnCopy(const ThrowOnCopy& o) : value(o.value) {
        if (++copy_count >= throw_after && throw_after > 0) {
            throw std::runtime_error("ThrowOnCopy: copy threw");
        }
    }
    
    static void reset() { copy_count = 0; throw_after = -1; }
    static void throw_on_copy(int n) { throw_after = n; copy_count = 0; }
};
```

---

## Compile-Fail Tests

### Purpose

Compile-fail tests verify that invalid code is **rejected at compile time**. They prove that:
- Concepts reject invalid types
- `static_assert` fires on invalid configurations
- Type safety is enforced

### Structure

```cpp
// tests/compile_fail/compile_fail_<component>_<reason>.cpp

// File header explaining what should fail and why
// ...

#include "tf_wrap/component.hpp"

int main() {
    // This line should fail to compile
    // because <reason>
    auto x = /* invalid usage */;
    (void)x;
    return 0;
}
```

### CI Integration

Compile-fail tests are built with a script that **expects failure**:

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

### Required Compile-Fail Tests

| File | What It Tests |
|------|---------------|
| `compile_fail_tensor_invalid_dtype.cpp` | `TensorScalar` concept rejects `std::string` |
| `compile_fail_tensor_void_dtype.cpp` | `TensorScalar` concept rejects `void` |

---

## CI Integration

### Test Matrix

| Job | Tests Run | Platform |
|-----|-----------|----------|
| `linux-gcc` | All stub tests (doctest) | Ubuntu, GCC 13/14 |
| `linux-clang` | All stub tests (doctest) | Ubuntu, Clang 17/18 |
| `windows-msvc` | All stub tests (doctest) | Windows, MSVC |
| `macos` | All stub tests (doctest) | macOS, Apple Clang |
| `sanitizers` | Stub tests with ASan + UBSan | Ubuntu, GCC |
| `real-tensorflow` | All `*_tf.cpp` tests | Ubuntu, TF 2.13-2.18 |
| `soak-tests` | `test_soak_tf.cpp` with ASan | Ubuntu, TF 2.18 |
| `thread-safety` | `test_thread_safety_tf.cpp` | Ubuntu, TF 2.18 |
| `fuzz-tests` | Fuzz targets (30s each) | Ubuntu, Clang + libFuzzer |
| `benchmarks` | Performance benchmarks | Ubuntu |
| `coverage` | Code coverage report | Ubuntu |
| `header-check` | Header standalone compile | Ubuntu |

### Stub vs Real TF Tests

**Stub tests** (`test_*.cpp` without `_tf` suffix):
- Use doctest framework
- Run on all platforms (Linux, Windows, macOS)
- Test API surface, error handling, edge cases
- Cannot test actual TF execution

**Real TF tests** (`test_*_tf.cpp`):
- Use minimal custom framework
- Run only on Linux with real TensorFlow C library
- Test actual tensor operations, session execution, SavedModel loading
- Verify results match expected values

### Build Commands

```bash
# Stub tests (all platforms)
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
    -I include -I third_party/tf_stub \
    -DTF_WRAPPER_TF_STUB_ENABLED=1 \
    third_party/tf_stub/tf_c_stub.cpp \
    tests/test_tensor.cpp \
    -o test_tensor

# Real TF tests (Linux with TF installed)
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic \
    -I include -I /path/to/tensorflow/include \
    tests/test_tensor_tf.cpp \
    -L /path/to/tensorflow/lib -ltensorflow \
    -o test_tensor_tf
```

---

## Checklist Before Submitting

### File Structure
- [ ] Correct file naming (`test_<component>.cpp` or `test_<component>_tf.cpp`)
- [ ] Appropriate framework (doctest for stub, custom for real TF)
- [ ] File header comment explaining what's tested

### Coverage
- [ ] Construction (default, factory methods)
- [ ] Move semantics (move ctor, move assign, moved-from state)
- [ ] All public methods tested
- [ ] Error cases (invalid inputs throw appropriate exceptions)
- [ ] Edge cases (empty, single element, boundaries)
- [ ] Type safety (dtype mismatches rejected)

### Assertions
- [ ] `[[nodiscard]]` functions use manual try-catch, not `CHECK_THROWS`
- [ ] Floating-point uses `doctest::Approx()` or `REQUIRE_CLOSE`
- [ ] Exception tests verify the correct exception type

### Real TF Tests
- [ ] Tests pass with TF 2.13 through 2.18
- [ ] No stub-specific behavior tested
- [ ] SavedModel tests use CI-generated test model

### Compile-Fail Tests
- [ ] File in `tests/compile_fail/` directory
- [ ] Clear comment explaining expected failure
- [ ] Single failure point per file
- [ ] Added to CI expected-fail job

---

## Anti-Patterns to Avoid

### 1. Testing Stub Behavior in Real TF Tests

```cpp
// BAD: Empty Run() only works with stub
TEST(empty_run) {
    Graph g;
    Session s(g);
    auto results = s.Run({}, {});  // Real TF requires fetches
    REQUIRE(results.empty());
}
```

### 2. Ignoring `[[nodiscard]]` Warnings

```cpp
// BAD: Triggers -Werror=unused-result
CHECK_THROWS(tensor.dtype());

// GOOD: Manual try-catch
bool threw = false;
try { auto d = tensor.dtype(); (void)d; } catch (...) { threw = true; }
CHECK(threw);
```

### 3. Hardcoded Paths

```cpp
// BAD: Won't work in CI
auto model = Model::Load("/home/user/models/test");

// GOOD: Use relative path from test working directory
auto model = Model::Load("test_savedmodel");
```

### 4. Platform-Specific Assumptions

```cpp
// BAD: int64_t size differs across platforms
CHECK(sizeof(shape[0]) == 8);

// GOOD: Use portable types
CHECK(t.shape()[0] == std::int64_t(3));
```

---

## Completed Improvements

All planned test infrastructure improvements have been implemented:

| Feature | Status | Location |
|---------|--------|----------|
| Benchmark infrastructure | ✅ Done | `tests/benchmark/` |
| Memory leak detection (ASan) | ✅ Done | `sanitizers` + `soak-tests` jobs |
| Thread safety tests | ✅ Done | `test_thread_safety_tf.cpp` |
| Property-based / fuzz testing | ✅ Done | `tests/fuzz/` |
| Code coverage reporting | ✅ Done | `tests/coverage/` |
| Corner case tests | ✅ Done | `test_corner_cases.cpp` |
| Adversarial tests | ✅ Done | `test_adversarial.cpp`, `test_adversarial_tf.cpp` |
| Comprehensive API tests | ✅ Done | `test_comprehensive.cpp`, `test_comprehensive_tf.cpp` |

### Deferred
- Exception safety tests (complex, requires careful design)
- Allocator tracking (not currently exposed by API)

---

*TensorFlowWrap Test Suite Style Guide v2.0 -- January 2026*
