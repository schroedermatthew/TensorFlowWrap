# TensorFlowWrap User Manual

## What This Library Does

TensorFlowWrap provides C++20 RAII wrappers for the TensorFlow C API. It handles resource cleanup automatically, adds type-checked tensor access, and converts C-style error codes to exceptions. The library focuses on inference—loading SavedModels and running predictions.

## What This Library Does Not Do

TensorFlowWrap does not provide operation wrappers (no `MatMul()`, `Conv2D()`, etc.), graph building utilities, training support, or gradient computation. Build models in Python, export as SavedModel, and use this library to run inference in C++.

## Requirements

- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- TensorFlow C library 2.13+
- A SavedModel exported from Python

---

## Quick Start

```cpp
#include "tf_wrap/core.hpp"
#include <iostream>

int main() {
    // Load SavedModel
    auto model = tf_wrap::Model::Load("path/to/saved_model");
    
    // Resolve operation names (do once at startup)
    auto input_op = model.resolve("serving_default_x:0");
    auto output_op = model.resolve("StatefulPartitionedCall:0");
    
    // Create input tensor
    auto input = tf_wrap::Tensor::FromVector<float>({1, 3}, {1.0f, 2.0f, 3.0f});
    
    // Run inference
    auto result = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    // Read output
    for (float v : result.read<float>()) {
        std::cout << v << "\n";
    }
}
```

All resources are cleaned up automatically when they go out of scope.

---

## Tensor Creation

Create tensors using factory methods on `tf_wrap::Tensor`:

### FromVector

Create a tensor from a shape and data:

```cpp
auto t = tf_wrap::Tensor::FromVector<float>(
    {2, 3},                                    // Shape: 2 rows, 3 columns
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}      // Data (6 elements)
);
```

The element count must match the shape. A shape of `{2, 3}` requires exactly 6 elements.

### FromScalar

Create a rank-0 tensor with a single value:

```cpp
auto t = tf_wrap::Tensor::FromScalar<float>(3.14f);
```

### Zeros

Create a zero-initialized tensor:

```cpp
auto t = tf_wrap::Tensor::Zeros<float>({224, 224, 3});
```

### Allocate

Create an uninitialized tensor for writing data directly:

```cpp
auto t = tf_wrap::Tensor::Allocate<float>({100, 100});
auto view = t.write<float>();
for (size_t i = 0; i < view.size(); ++i) {
    view[i] = compute_value(i);
}
```

### FromString

Create a string tensor:

```cpp
auto t = tf_wrap::Tensor::FromString("hello world");
```

### Clone

Copy an existing tensor:

```cpp
auto copy = original.Clone();
```

---

## Tensor Data Access

Access tensor data through views that keep the underlying memory alive.

### Read Access

```cpp
auto tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});

auto view = tensor.read<float>();  // TensorView<const float>

// Iteration
for (float v : view) {
    std::cout << v << "\n";
}

// Indexing
float first = view[0];
float checked = view.at(1);  // Throws on out-of-bounds

// STL algorithms
auto sum = std::accumulate(view.begin(), view.end(), 0.0f);
```

The view holds a `shared_ptr` to the tensor's internal state. Even if the original `Tensor` object is moved or destroyed, the view keeps the data alive:

```cpp
tf_wrap::TensorView<const float> view;
{
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    view = t.read<float>();
    // t goes out of scope here
}
// view is still valid
float value = view[0];  // 42.0f
```

### Write Access

```cpp
auto tensor = tf_wrap::Tensor::Allocate<float>({100});
auto view = tensor.write<float>();  // TensorView<float>

for (size_t i = 0; i < view.size(); ++i) {
    view[i] = static_cast<float>(i);
}
```

### Type Checking

The template parameter must match the tensor's dtype:

```cpp
auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);

auto good = t.read<float>();   // OK
auto bad = t.read<int32_t>();  // Throws: dtype mismatch
```

### Convenience Methods

For extracting data when the tensor is no longer needed:

```cpp
// Single value
float scalar = tensor.ToScalar<float>();

// Copy to vector
std::vector<float> data = tensor.ToVector<float>();
```

---

## Tensor Properties

Query tensor metadata:

```cpp
auto t = tf_wrap::Tensor::FromVector<float>({2, 3, 4}, data);

t.shape();        // {2, 3, 4}
t.rank();         // 3
t.dtype();        // TF_FLOAT
t.dtype_name();   // "FLOAT"
t.byte_size();    // 96 (2*3*4*4 bytes)
t.num_elements(); // 24
t.handle();       // Raw TF_Tensor* (for C API interop)
```

---

## Loading SavedModels

### Using Model (Recommended)

```cpp
auto model = tf_wrap::Model::Load("path/to/saved_model");
```

The default tag is `"serve"`. To specify different tags:

```cpp
auto model = tf_wrap::Model::Load("path/to/model", {"serve", "gpu"});
```

### Using Session Directly

For more control, use `Session::LoadSavedModel`:

```cpp
auto [session, graph] = tf_wrap::Session::LoadSavedModel("path/to/model");
```

This returns both the session and the graph. The session internally shares ownership of the graph, so you can discard the graph if you don't need to inspect it.

---

## Running Inference

### Resolving Operations

Convert operation names to `TF_Output` handles using `resolve()`:

```cpp
TF_Output input_op = model.resolve("serving_default_x:0");
TF_Output output_op = model.resolve("StatefulPartitionedCall:0");
```

The format is `"operation_name:output_index"`. The output index defaults to 0 if omitted.

Call `resolve()` once at startup and cache the results. Name lookup involves string operations and graph traversal.

### Using Runner

The `Runner` class provides a fluent interface for `TF_SessionRun`:

```cpp
auto result = model.runner()
    .feed(input_op, input_tensor)
    .fetch(output_op)
    .run_one();
```

For multiple outputs:

```cpp
auto results = model.runner()
    .feed(input_op, input)
    .fetch(output1_op)
    .fetch(output2_op)
    .run();  // Returns std::vector<Tensor>

auto first = std::move(results[0]);
auto second = std::move(results[1]);
```

For multiple inputs:

```cpp
auto result = model.runner()
    .feed(input1_op, tensor1)
    .feed(input2_op, tensor2)
    .fetch(output_op)
    .run_one();
```

### Reusing Runner

Create a runner once and reuse it:

```cpp
auto runner = model.runner();

for (const auto& request : requests) {
    runner.clear();  // Reset feeds/fetches
    
    auto input = create_input(request);
    auto result = runner
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    process(result);
}
```

---

## Device Enumeration

List available compute devices:

```cpp
auto devices = model.session().ListDevices();

for (const auto& dev : devices.all()) {
    std::cout << dev.name << " (" << dev.type << ")";
    if (dev.memory_bytes > 0) {
        std::cout << " " << dev.memory_bytes << " bytes";
    }
    std::cout << "\n";
}
```

Check for GPU availability:

```cpp
if (model.session().HasGPU()) {
    std::cout << "GPU available\n";
}
```

---

## Error Handling

All operations that can fail throw `tf_wrap::Error`:

```cpp
try {
    auto model = tf_wrap::Model::Load("nonexistent");
} catch (const tf_wrap::Error& e) {
    std::cerr << e.what() << "\n";
    // Includes: TF_Code, operation, source location, message
}
```

Error properties:

```cpp
try {
    // ...
} catch (const tf_wrap::Error& e) {
    TF_Code code = e.code();           // TF_NOT_FOUND, TF_INVALID_ARGUMENT, etc.
    const char* name = e.code_name();  // "NOT_FOUND", "INVALID_ARGUMENT"
    auto loc = e.location();           // std::source_location
    
    std::cerr << "Error at " << loc.file_name() << ":" << loc.line() << "\n";
    std::cerr << "Code: " << name << "\n";
    std::cerr << "Message: " << e.what() << "\n";
}
```

---

## Threading

`Session::Run` is thread-safe (TensorFlow's guarantee). Share the model across threads, but give each thread its own tensors:

```cpp
// Shared (thread-safe)
auto model = tf_wrap::Model::Load("model");
auto input_op = model.resolve("serving_default_x:0");
auto output_op = model.resolve("StatefulPartitionedCall:0");

// Per-thread
void worker(int thread_id) {
    auto input = tf_wrap::Tensor::FromVector<float>({1}, {float(thread_id)});
    
    auto result = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    // Process result...
}
```

Do not share `Tensor` objects between threads. Create tensors in the thread that uses them.

---

## Interop with C API

Access raw handles for C API interop:

```cpp
TF_Tensor* raw = tensor.handle();
TF_Session* raw_session = session.handle();
TF_Graph* raw_graph = graph.handle();
```

The wrappers still own the resources. Do not call `TF_Delete*` on handles obtained this way.

To take ownership of a raw tensor:

```cpp
TF_Tensor* raw = some_c_api_function();
auto tensor = tf_wrap::Tensor::FromRaw(raw);
// tensor now owns raw, will delete it
```

---

## API Reference

### Tensor Factory Methods

| Method | Creates |
|--------|---------|
| `FromVector<T>(shape, data)` | Tensor from shape and data |
| `FromScalar<T>(value)` | Rank-0 tensor |
| `FromString(str)` | String tensor |
| `Zeros<T>(shape)` | Zero-initialized tensor |
| `Allocate<T>(shape)` | Uninitialized tensor |
| `Clone()` | Deep copy |

### Tensor Access Methods

| Method | Returns |
|--------|---------|
| `read<T>()` | `TensorView<const T>` |
| `write<T>()` | `TensorView<T>` |
| `ToScalar<T>()` | `T` (single value) |
| `ToVector<T>()` | `std::vector<T>` (copy) |

### Tensor Properties

| Method | Returns |
|--------|---------|
| `shape()` | `const std::vector<int64_t>&` |
| `rank()` | `int` |
| `dtype()` | `TF_DataType` |
| `dtype_name()` | `const char*` |
| `byte_size()` | `size_t` |
| `num_elements()` | `size_t` |
| `handle()` | `TF_Tensor*` |

### Model Methods

| Method | Returns |
|--------|---------|
| `Load(path, tags)` | `Model` (static) |
| `resolve(name)` | `TF_Output` |
| `runner()` | `Runner` |
| `session()` | `const Session&` |
| `graph()` | `const Graph&` |
| `valid()` | `bool` |

### Session Methods

| Method | Returns |
|--------|---------|
| `LoadSavedModel(path, tags)` | `pair<Session, Graph>` (static) |
| `Run(feeds, fetches)` | `vector<Tensor>` |
| `resolve(name)` | `TF_Output` |
| `ListDevices()` | `DeviceList` |
| `HasGPU()` | `bool` |
| `handle()` | `TF_Session*` |

### Runner Methods

| Method | Returns |
|--------|---------|
| `feed(output, tensor)` | `Runner&` |
| `fetch(output)` | `Runner&` |
| `run()` | `vector<Tensor>` |
| `run_one()` | `Tensor` |
| `clear()` | `void` |

---

## Troubleshooting

### "operation not found in graph"

The operation name doesn't exist. Check the model's signature:

```python
# In Python, inspect the SavedModel
import tensorflow as tf
model = tf.saved_model.load("path/to/model")
print(model.signatures["serving_default"])
```

### "dtype mismatch"

The template type doesn't match the tensor's actual dtype:

```cpp
auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
auto view = t.read<int32_t>();  // Throws: expected FLOAT, got INT32
```

### "shape requires N elements, got M"

The data size doesn't match the shape:

```cpp
// Shape {2, 3} needs 6 elements
auto t = tf_wrap::Tensor::FromVector<float>({2, 3}, {1, 2, 3, 4, 5});  // Throws: needs 6, got 5
```

### Linking errors

Ensure you're linking against the TensorFlow C library:

```bash
g++ -std=c++20 your_code.cpp -ltensorflow -o your_app
```

And that `libtensorflow.so` is in your library path.

---

*TensorFlowWrap User Manual v1.0 — January 2026*
