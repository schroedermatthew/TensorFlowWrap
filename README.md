# TensorFlowWrap

A modern C++20 header-only wrapper for the TensorFlow C API.

## Features

- **Header-only** — Just include and use, no build step for the library
- **RAII** — Automatic resource management for tensors, graphs, sessions
- **Thread-safe** — Policy-based locking (NoLock, Mutex, SharedMutex)
- **Type-safe** — Compile-time dtype checking with concepts
- **Modern C++20** — Concepts, ranges, `std::span`, `std::format`

## Quick Start

```cpp
#include <tf_wrap/all.hpp>

int main() {
    // Create tensors
    tf_wrap::FastTensor a = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 2.0f});
    tf_wrap::FastTensor b = tf_wrap::FastTensor::FromVector<float>({2}, {10.0f, 20.0f});
    
    // Build graph
    tf_wrap::FastGraph graph;
    
    auto op_a = graph.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto op_b = graph.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    graph.NewOperation("AddV2", "Sum")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    // Run
    tf_wrap::FastSession session(graph);
    auto results = session.Run({}, {{"Sum", 0}});
    
    // Extract result: [11.0, 22.0]
    auto output = results[0].ToVector<float>();
}
```

## Thread Safety Policies

```cpp
// No locking (single-threaded, fastest)
tf_wrap::FastTensor t1 = tf_wrap::FastTensor::FromScalar(1.0f);

// Mutex (exclusive access)
tf_wrap::SafeTensor t2 = tf_wrap::SafeTensor::FromScalar(2.0f);

// Shared mutex (multiple readers OR single writer)
tf_wrap::SharedTensor t3 = tf_wrap::SharedTensor::FromScalar(3.0f);
```

## Requirements

- C++20 compiler (GCC 12+, Clang 15+, MSVC 2022+)
- TensorFlow C library (for runtime)

## Installation

### Header-only

Copy `include/tf_wrap/` to your project.

### CMake

```cmake
add_subdirectory(TensorFlowWrap)
target_link_libraries(your_target PRIVATE TensorFlowWrap::TensorFlowWrap)
```

## Building Tests

```bash
mkdir build && cd build
cmake .. -DTF_WRAPPER_TF_STUB=ON  # Use stub for testing without TF
make
ctest
```

## License

MIT
