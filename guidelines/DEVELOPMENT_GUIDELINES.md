# TensorFlowWrap Development Guidelines

## Document Governance

This is the **authoritative** TensorFlowWrap guideline document.

| Document | Role | Authority |
|----------|------|-----------|
| **Development Guidelines** (this) | Normative rules, code standards, AI behavior | HIGHEST |
| **Design Document** | Architecture, rationale, design decisions | PRIMARY for design |
| **Test Suite Style Guide** | Test structure, coverage, frameworks | PRIMARY for tests |
| **CI Workflow Style Guide** | GitHub Actions, job matrix, gating | PRIMARY for CI |
| **AI Demerit Tracker** | Violation record | Record only |

**Precedence rules:**
- Development Guidelines override all other documents
- Each document should be standalone
- No document assumes another has been read

---

## 1. Library Design Principles

### 1.1 Core Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| **C++ Standard** | C++20 required (no fallback) |
| **Architecture** | Header-only |
| **Dependencies** | TensorFlow C API only; no other third-party libraries |
| **Target Domain** | Production inference |
| **Scope** | SavedModel loading and inference only |

### 1.2 What We Are

TensorFlowWrap is a **production inference wrapper**:
- RAII wrappers for TensorFlow C API resources
- Type-safe tensor access
- Zero-overhead abstraction
- Exception-based error handling

### 1.3 What We Are Not

| Excluded | Reason |
|----------|--------|
| Training | Use Python |
| Graph building | Use Python |
| Gradient computation | Not needed for inference |
| Custom ops | Out of scope |
| TensorFlow Lite | Different ecosystem |

### 1.4 Versioning & Compatibility

- No version number above 1
- Library is pre-release -- no backwards compatibility concerns
- **Never add backwards compatibility aliases**
- If something is renamed, rename it completely

---

## 2. Component Architecture

### 2.1 Header Organization

```
include/tf_wrap/
├── core.hpp              # Main include (umbrella header)
├── tensor.hpp            # Tensor creation, access, lifetime
├── session.hpp           # Session, SessionOptions, Run
├── graph.hpp             # Graph wrapper
├── model.hpp             # High-level Model + Runner API
├── status.hpp            # Status wrapper
├── error.hpp             # Error exception class
├── format.hpp            # String formatting utilities
├── scope_guard.hpp       # RAII scope guard utility
├── small_vector.hpp      # Stack-allocating vector
└── detail/               # Internal implementation details
    └── raw_tensor_ptr.hpp
```

### 2.2 Dependency Rules

| Component | May Depend On |
|-----------|---------------|
| `error.hpp` | `<exception>`, `<string>` |
| `status.hpp` | `error.hpp`, TF C API |
| `tensor.hpp` | `status.hpp`, `error.hpp` |
| `graph.hpp` | `status.hpp` |
| `session.hpp` | `tensor.hpp`, `graph.hpp`, `status.hpp` |
| `model.hpp` | `session.hpp` |
| `core.hpp` | All of the above |

**Rule:** Lower components must not include higher components.

---

## 3. Naming Conventions

### 3.1 Files

| Type | Convention | Example |
|------|------------|---------|
| Public header | `snake_case.hpp` | `tensor.hpp` |
| Detail header | `snake_case.hpp` in `detail/` | `detail/raw_tensor_ptr.hpp` |
| Test (stub) | `test_<component>.cpp` | `test_tensor.cpp` |
| Test (real TF) | `test_<component>_tf.cpp` | `test_tensor_tf.cpp` |

### 3.2 Code

| Element | Convention | Example |
|---------|------------|---------|
| Namespace | `snake_case` | `tf_wrap`, `tf_wrap::detail` |
| Class | `PascalCase` | `Tensor`, `SessionOptions` |
| Function | `PascalCase` | `FromScalar`, `ToVector` |
| Method | `PascalCase` or `snake_case` for STL compat | `read()`, `num_elements()` |
| Variable | `snake_case` | `input_tensor` |
| Member variable | `snake_case_` | `session_`, `graph_state_` |
| Constant | `kPascalCase` | `kDefaultCapacity` |
| Macro | `SCREAMING_SNAKE` | `TF_WRAPPER_TF_STUB_ENABLED` |
| Template parameter | `PascalCase` | `T`, `Dtype` |

### 3.3 STL Compatibility Exception

Methods that provide STL-compatible interfaces use `snake_case`:
- `size()`, `empty()`, `begin()`, `end()`
- `push_back()`, `pop_back()`
- `data()`, `at()`

---

## 4. Code Style

### 4.1 Formatting

| Rule | Specification |
|------|---------------|
| Line width | 100 columns target, 120 max |
| Indentation | 4 spaces (no tabs) |
| Braces | Same line (K&R style) |
| Includes | Grouped: std, TF C API, tf_wrap |

### 4.2 Include Order

```cpp
// 1. Corresponding header (for .cpp files)
#include "tf_wrap/tensor.hpp"

// 2. C++ standard library
#include <memory>
#include <span>
#include <vector>

// 3. TensorFlow C API
extern "C" {
#include <tensorflow/c/c_api.h>
}

// 4. Other tf_wrap headers
#include "tf_wrap/error.hpp"
#include "tf_wrap/status.hpp"
```

### 4.3 `[[nodiscard]]` Usage

Use `[[nodiscard]]` for:
- Factory functions (`FromScalar`, `FromVector`, `Clone`)
- Accessors that return important values (`dtype()`, `shape()`, `handle()`)
- Methods where ignoring the return value is likely a bug

```cpp
[[nodiscard]] static Tensor FromScalar(T value);
[[nodiscard]] TF_DataType dtype() const;
[[nodiscard]] std::vector<std::int64_t> shape() const;
```

### 4.4 Error Handling

Use exceptions, not error codes:

```cpp
// Good: Exception on failure
Tensor Tensor::FromVector(std::initializer_list<std::int64_t> shape, 
                          std::span<const T> data) {
    if (data.size() != expected_size) {
        throw Error::Wrapper(TF_INVALID_ARGUMENT, "FromVector",
            "data size mismatch", "", -1);
    }
    // ...
}

// Bad: Error codes
std::optional<Tensor> Tensor::FromVector(...);  // Don't do this
```

### 4.5 RAII Everywhere

Every TensorFlow resource must be wrapped:

```cpp
class Tensor {
public:
    ~Tensor() { if (tensor_) TF_DeleteTensor(tensor_); }
    
    Tensor(Tensor&& other) noexcept : tensor_(other.tensor_) {
        other.tensor_ = nullptr;
    }
    
    // Delete copy operations
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
private:
    TF_Tensor* tensor_{nullptr};
};
```

---

## 5. Documentation Standards

### 5.1 File Headers

Every header file must have a documentation comment:

```cpp
// tf_wrap/tensor.hpp
// RAII wrapper for TF_Tensor with type-safe access
//
// Provides:
// - Factory methods: FromScalar, FromVector, Zeros, Allocate
// - Type-safe read/write views
// - Automatic lifetime management
```

### 5.2 Function Documentation

Document public functions with:
- Brief description
- Parameters (if non-obvious)
- Return value
- Exceptions thrown

```cpp
/// Create a tensor from a scalar value.
/// @tparam T Scalar type (must satisfy TensorScalar concept)
/// @param value The scalar value
/// @return New tensor containing the value
/// @throws Error if allocation fails
template<TensorScalar T>
[[nodiscard]] static Tensor FromScalar(T value);
```

---

## 6. Testing Requirements

See [TEST_STYLE_GUIDE.md](TEST_STYLE_GUIDE.md) for complete details.

### 6.1 Test Coverage Requirements

Every component must have:

| Category | Required |
|----------|----------|
| Construction | ✅ |
| Move semantics | ✅ |
| All public methods | ✅ |
| Error cases | ✅ |
| Edge cases | ✅ |
| Type safety | ✅ |

### 6.2 Dual Testing Strategy

| Test File | Framework | Runs With |
|-----------|-----------|-----------|
| `test_<component>.cpp` | doctest | Stub (all platforms) |
| `test_<component>_tf.cpp` | Custom | Real TF (Linux) |

---

## 7. AI Assistant Rules

### 7.1 General Principles

| Rule | Detail |
|------|--------|
| **No code unless requested** | Do not generate code unless explicitly asked |
| **Complete files only** | **NEVER provide truncated files** |
| **Always compile** | Compile code before delivering when possible |
| **No AI comments** | Never include `// NEW`, `// FIXED`, etc. |
| **Provide downloads** | Always provide download links for modified files |

### 7.2 Compile Before Deliver

**Critical:** Always compile and test code locally before providing it.

```bash
# Minimum verification before delivering code:
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
    -Iinclude -Ithird_party/tf_stub \
    -DTF_WRAPPER_TF_STUB_ENABLED=1 \
    third_party/tf_stub/tf_c_stub.cpp \
    tests/test_<component>.cpp \
    -o test_<component>

./test_<component>
```

If compilation fails, fix the code before delivering.

### 7.3 Verification Claims

- Never say "compiled/ran" unless actually done in the session
- Include exact commands and output when claiming verification
- If compilation cannot be done, explicitly state "not compiled"

### 7.4 Common Mistakes to Avoid

| Mistake | Consequence |
|---------|-------------|
| Not compiling before delivery | CI failures, wasted user time |
| Using `CHECK_THROWS` with `[[nodiscard]]` | `-Werror` failures |
| Mixing stub-only tests with real TF | Test failures on real TF |
| Forgetting to update CI for new tests | Tests not run |
| Ignoring existing patterns | Inconsistent codebase |

### 7.5 Before Providing Code

Checklist:
- [ ] Compiled locally with stub
- [ ] All tests pass
- [ ] No warnings with `-Wall -Wextra -Wpedantic -Werror`
- [ ] Follows existing code patterns
- [ ] CI updated if adding new test files

---

## 8. CI Requirements

See [CI_STYLE_GUIDE.md](CI_STYLE_GUIDE.md) for complete details.

### 8.1 Required Platforms

| Platform | Compilers |
|----------|-----------|
| Linux | GCC 13/14, Clang 17/18 |
| Windows | MSVC |
| macOS | Apple Clang |

### 8.2 Required Tests

| Test Type | When |
|-----------|------|
| Stub tests | All platforms |
| Real TF tests | Linux only, TF 2.13-2.18 |
| Sanitizers | ASan + UBSan |
| Header check | Standalone compilation |

---

## 9. Checklist for Changes

### 9.1 New Header File

- [ ] Placed in `include/tf_wrap/`
- [ ] Uses `.hpp` extension
- [ ] Has file header documentation
- [ ] Includes only necessary headers
- [ ] Added to `core.hpp` if public
- [ ] Header standalone check passes

### 9.2 New Test File

- [ ] Correct naming (`test_<component>.cpp` or `test_<component>_tf.cpp`)
- [ ] Compiles locally
- [ ] All tests pass
- [ ] CI workflow updated
- [ ] Sanitizer tests updated

### 9.3 API Change

- [ ] All usages updated
- [ ] Tests updated
- [ ] Documentation updated
- [ ] No backwards compatibility shims

---

## Changelog

### v1.0 (January 2026)
- Initial Development Guidelines for TensorFlowWrap

---

*TensorFlowWrap Development Guidelines v1.0 -- January 2026*
