# TensorFlowWrap

A modern C++20 header-only wrapper for the TensorFlow C API.

## Features

- **Header-only** — Just include and use, no build step for the library
- **RAII** — Automatic resource management for tensors, graphs, sessions
- **Type-safe** — Compile-time dtype checking with concepts
- **Modern C++20** — Concepts, ranges, `std::span`, `std::format`
- **Lifetime-safe views** — Tensor views keep underlying data alive

## Quick Start

```cpp
#include <tf_wrap/all.hpp>

int main() {
    // Create tensors
    tf_wrap::Tensor a = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    tf_wrap::Tensor b = tf_wrap::Tensor::FromVector<float>({2}, {10.0f, 20.0f});
    
    // Build graph
    tf_wrap::Graph graph;
    
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
    tf_wrap::Session session(graph);
    auto results = session.Run({}, {{"Sum", 0}});
    
    // Extract result: [11.0, 22.0]
    auto output = results[0].ToVector<float>();
}
```

## Threading

TensorFlowWrap does not provide synchronization. Follow TensorFlow's threading rules:

- `Session::Run()` is thread-safe (TensorFlow's guarantee)
- Do not share mutable tensors across threads
- Create per-thread input tensors; treat outputs as thread-local
- Graph is frozen after Session creation (wrapper policy)

## Requirements

- C++20 compiler (GCC 12+, Clang 15+, MSVC 2022+)
- TensorFlow C library (for runtime)

## Installation

### Header-only

Copy `include/tf_wrap/` to your project.

### CMake

```cmake
add_subdirectory(TensorFlowWrap)
target_link_libraries(your_target PRIVATE tf::wrapper)
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
