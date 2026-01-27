# TensorFlow C++ Wrapper Comparison

## What TensorFlowWrap Is

TensorFlowWrap provides RAII wrappers for TensorFlow's C API. It handles resource cleanup automatically, adds type-checked tensor access, and converts C-style error codes to C++ exceptions. The library focuses exclusively on inference—loading SavedModels and running predictions.

## What TensorFlowWrap Is Not

TensorFlowWrap does not wrap TensorFlow operations. There are no `MatMul()`, `Conv2D()`, or `Softmax()` functions. Graph building uses the raw TensorFlow C API through `TF_NewOperation`. If you need high-level operation wrappers, consider cppflow or the official TensorFlow C++ API.

---

## Libraries Compared

| Library | C++ Standard | Primary Use Case | Maintained |
|---------|:------------:|------------------|:----------:|
| **TensorFlowWrap** | C++20 | RAII inference wrappers | Yes |
| **cppflow** | C++17 | Eager execution, ops | 2022 |
| **TensorFlow C++ API** | C++17 | Full TensorFlow access | Yes |
| **TensorFlow Lite C++** | C++11 | Mobile/embedded | Yes |

---

## Feature Comparison

### Resource Management

| Feature | TensorFlowWrap | cppflow | TF C++ API | TF C API |
|---------|:--------------:|:-------:|:----------:|:--------:|
| RAII for Tensor | ✓ | ✓ | ✓ | ✗ |
| RAII for Session | ✓ | ✓ | ✓ | ✗ |
| RAII for Graph | ✓ | Partial | ✓ | ✗ |
| RAII for Status | ✓ | N/A | ✓ | ✗ |
| View lifetime safety | ✓ | ✗ | ✗ | ✗ |

TensorFlowWrap's `TensorView` holds a `shared_ptr` that keeps the underlying tensor alive. This prevents use-after-free when views outlive the tensor object.

### Type Safety

| Feature | TensorFlowWrap | cppflow | TF C++ API |
|---------|:--------------:|:-------:|:----------:|
| Compile-time dtype concepts | ✓ | ✗ | ✗ |
| Runtime dtype checking | ✓ | ✓ | ✓ |
| Explicit tensor scalar types | ✓ | ✗ | ✗ |

TensorFlowWrap uses C++20 concepts to restrict tensor element types at compile time:

```cpp
template<TensorScalar T>
static Tensor FromScalar(T value);

// Valid: float is a TensorScalar
auto t1 = Tensor::FromScalar<float>(3.14f);

// Compile error: std::string is not a TensorScalar
auto t2 = Tensor::FromScalar<std::string>("x");
```

### Operation Coverage

| Library | Wrapped Operations | Approach |
|---------|:------------------:|----------|
| **TensorFlowWrap** | 0 | Use C API's `TF_NewOperation` |
| **cppflow** | ~1500 | Auto-generated wrappers |
| **TF C++ API** | ~1500+ | Native implementation |

TensorFlowWrap deliberately excludes operation wrappers. The scope is resource management and inference execution, not graph construction.

### Error Handling

| Library | Error Model |
|---------|-------------|
| TensorFlowWrap | Exceptions with `TF_Code`, source location, context |
| cppflow | Exceptions (basic) |
| TF C++ API | `Status` return values |
| TF C API | `TF_Status*` output parameter |

TensorFlowWrap exceptions include the TensorFlow error code, the operation that failed, the source file and line, and contextual details:

```cpp
try {
    auto view = tensor.read<int32_t>();  // tensor is float
} catch (const tf_wrap::Error& e) {
    // e.code() == TF_INVALID_ARGUMENT
    // e.what() includes "dtype mismatch: expected FLOAT, got INT32"
    // e.location() points to this line
}
```

---

## Code Comparison

### Loading a SavedModel

**TensorFlowWrap:**
```cpp
#include "tf_wrap/core.hpp"

auto model = tf_wrap::Model::Load("/path/to/saved_model");
```

**cppflow:**
```cpp
#include <cppflow/cppflow.h>

cppflow::model model("/path/to/saved_model");
```

**TensorFlow C API:**
```cpp
TF_Status* status = TF_NewStatus();
TF_Graph* graph = TF_NewGraph();
TF_SessionOptions* opts = TF_NewSessionOptions();
const char* tags[] = {"serve"};

TF_Session* session = TF_LoadSessionFromSavedModel(
    opts, nullptr, "/path/to/saved_model", tags, 1, graph, nullptr, status);

if (TF_GetCode(status) != TF_OK) {
    // Handle error, clean up status, graph, opts...
}
// Must remember to delete status, opts
// Must close and delete session when done
// Must delete graph when done
```

### Running Inference

**TensorFlowWrap:**
```cpp
// Resolve operation names once at startup
TF_Output input_op = model.resolve("serving_default_x:0");
TF_Output output_op = model.resolve("StatefulPartitionedCall:0");

// Create input
auto input = tf_wrap::Tensor::FromVector<float>({1, 3}, {1.0f, 2.0f, 3.0f});

// Run inference
auto result = model.runner()
    .feed(input_op, input)
    .fetch(output_op)
    .run_one();

// Access results
for (float v : result.read<float>()) {
    std::cout << v << "\n";
}
```

**cppflow:**
```cpp
auto input = cppflow::tensor({1.0f, 2.0f, 3.0f});
input = cppflow::reshape(input, {1, 3});

auto output = model({{"serving_default_x:0", input}});
auto data = output[0].get_data<float>();
```

---

## When to Use Each Library

**Use TensorFlowWrap when:**
- You need RAII resource management over the TensorFlow C API
- You want type-checked tensor access with view lifetime safety
- Your models are pre-built SavedModels (no graph construction needed)
- You're embedding inference in a C++20 application

**Use cppflow when:**
- You want to build graphs using high-level operation wrappers
- You prefer eager execution style
- C++17 is your target

**Use TensorFlow C++ API when:**
- You need full TensorFlow functionality
- You can build TensorFlow from source
- ABI stability is not a concern

**Use TensorFlow C API directly when:**
- You need maximum control
- You're binding to another language
- You have existing C API expertise

---

## Summary

TensorFlowWrap is a thin layer over the TensorFlow C API that adds RAII, type safety, and exception-based errors. It does not provide operation wrappers or graph-building utilities. The library is appropriate for applications that load pre-trained SavedModels and run inference in C++20 environments where automatic resource cleanup and type safety matter.

| Aspect | TensorFlowWrap |
|--------|----------------|
| Scope | RAII wrappers + inference |
| Operations | None (use C API) |
| Graph building | None (use C API) |
| Resource safety | Automatic cleanup |
| Type safety | Concepts + runtime checks |
| Error handling | Exceptions |
| Dependencies | TensorFlow C library |
| Header-only | Yes |

---

*Comparison accurate as of January 2026*
