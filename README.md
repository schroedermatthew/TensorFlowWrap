# TensorFlowWrap

C++20 RAII wrappers for the TensorFlow C API. Header-only. Inference only.

TensorFlowWrap handles resource cleanup automatically, adds compile-time type checking via C++20 concepts, and converts C-style error codes to structured exceptions. Build your models in Python, export as SavedModel, run inference in C++.

## Quick Start

```cpp
#include "tf_wrap/core.hpp"

// Load model once at startup
auto model = tf_wrap::Model::Load("/path/to/saved_model");

// Resolve operation names once (not per-request)
auto [input, output] = model.resolve("serving_default_input:0",
                                      "StatefulPartitionedCall:0");

// Warmup to trigger JIT compilation
auto dummy = tf_wrap::Tensor::Zeros<float>({1, 224, 224, 3});
model.warmup(input, dummy, output);

// Hot path — handle-based, no string parsing
while (serving) {
    auto input_tensor = get_request_tensor();
    auto result = model.runner()
        .feed(input, input_tensor)
        .fetch(output)
        .run_one();

    for (float v : result.read<float>()) {
        send_response(v);
    }
}
```

All resources are cleaned up automatically when they go out of scope.

## What This Wraps

| Class | Wraps | Purpose |
|-------|-------|---------|
| `tf_wrap::Tensor` | `TF_Tensor` | Type-safe tensor creation, read/write views with lifetime safety |
| `tf_wrap::Graph` | `TF_Graph` | Immutable after session creation |
| `tf_wrap::Session` | `TF_Session` | Thread-safe `Run()` (TensorFlow's guarantee) |
| `tf_wrap::Runner` | — | Fluent API for session execution |
| `tf_wrap::Model` | — | High-level SavedModel facade with warmup, resolution, and zero-allocation hot path |

## What This Does Not Wrap

There are no `MatMul()`, `Conv2D()`, or `Softmax()` functions. TensorFlowWrap does not wrap TensorFlow operations, provide graph building utilities, or support training. If you need operation wrappers, see [cppflow](https://github.com/serizba/cppflow) or the official TensorFlow C++ API.

## Type Safety

TensorFlowWrap uses C++20 concepts to catch type errors at compile time:

```cpp
auto t = tf_wrap::Tensor::FromScalar<float>(3.14f);     // OK
auto t = tf_wrap::Tensor::FromScalar<std::string>("x");  // compile error
```

`TensorView` holds a `shared_ptr` to the underlying tensor, preventing use-after-free when views outlive the tensor object.

## Error Handling

Exceptions include the TensorFlow error code, the operation that failed, and contextual details:

```cpp
try {
    auto view = tensor.read<int32_t>();  // tensor is float
} catch (const tf_wrap::Error& e) {
    // e.code() == TF_INVALID_ARGUMENT
    // e.what(): "dtype mismatch: expected FLOAT, got INT32"
}
```

## Requirements

- C++20 compiler (GCC 13+, Clang 17+, MSVC 2022+, Apple Clang)
- TensorFlow C library 2.13+ (tested against 2.13.0, 2.14.0, 2.15.0, 2.16.1, 2.17.1, 2.18.1)
- CMake 3.20+

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/tensorflow_c
cmake --build .
```

### Running Tests Without TensorFlow

The repo includes a lightweight C API stub (`third_party/tf_stub/`) that implements enough of the TensorFlow C API to compile and run the full test suite without installing TensorFlow. This is what CI uses for most jobs.

```bash
mkdir build && cd build
cmake .. -DTFWRAP_USE_STUB=ON
cmake --build .
ctest
```

## CI

Single workflow, 14 jobs:

- **Compilers:** GCC 13/14, Clang 17/18, MSVC, Apple Clang (ARM)
- **Sanitizers:** AddressSanitizer + UndefinedBehaviorSanitizer
- **Real TensorFlow:** Tests against 6 TF versions (2.13–2.18)
- **Stress testing:** Soak tests and thread safety tests under ASan
- **Fuzz testing:** libFuzzer on tensor parsing paths
- **Benchmarks and coverage**

23 test files, including adversarial tests, corner cases, lifecycle tests, and compile-fail tests.

## Documentation

- [User Manual](docs/User_Manual.md) — full API reference and usage patterns
- [Migration Guide](docs/Migration_Guide.md) — migrating from raw C API or cppflow
- [Comparison Report](docs/COMPARISON_REPORT.md) — how TensorFlowWrap compares to alternatives

## Authorship

This library was written entirely by AI (primarily Claude by Anthropic). The human provided direction, requirements, and judgment but wrote none of the code. See the companion project [FAT-P](https://github.com/schroedermatthew/FatP) for the full methodology behind this development approach.

## License

MIT
