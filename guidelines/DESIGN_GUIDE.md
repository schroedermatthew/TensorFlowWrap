# TensorFlowWrap Design Document

## Executive Summary

TensorFlowWrap is a **production inference wrapper** for the TensorFlow C API. It provides modern C++20 RAII wrappers that make it safe and ergonomic to load SavedModels and run inference in performance-critical applications.

**What it is:**
- A thin, zero-overhead abstraction over TensorFlow's C API
- Focused exclusively on inference (loading models, running predictions)
- Designed for embedding TensorFlow inference in C++ applications

**What it is not:**
- A training framework
- A graph-building library
- A replacement for TensorFlow's Python API

---

## Problem Statement

### The TensorFlow C API is Unsafe

TensorFlow's C API is the only stable, portable way to run TensorFlow models from C++. However, it has significant usability problems:

```c
// Raw TensorFlow C API - error-prone and verbose
TF_Status* status = TF_NewStatus();
TF_Graph* graph = TF_NewGraph();
TF_SessionOptions* opts = TF_NewSessionOptions();
TF_Buffer* run_opts = NULL;

const char* tags[] = {"serve"};
TF_Session* session = TF_LoadSessionFromSavedModel(
    opts, run_opts, "/path/to/model", tags, 1, graph, NULL, status);

if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "Error: %s\n", TF_Message(status));
    // Manual cleanup of everything allocated so far...
    TF_DeleteStatus(status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(opts);
    return;
}

// Create input tensor - manual memory management
int64_t dims[] = {1, 3};
size_t nbytes = 3 * sizeof(float);
TF_Tensor* input = TF_AllocateTensor(TF_FLOAT, dims, 2, nbytes);
float* data = (float*)TF_TensorData(input);
data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f;

// Run inference - complex setup
TF_Output inputs[1] = {{TF_GraphOperationByName(graph, "serving_default_x"), 0}};
TF_Output outputs[1] = {{TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0}};
TF_Tensor* input_values[1] = {input};
TF_Tensor* output_values[1] = {NULL};

TF_SessionRun(session, NULL,
    inputs, input_values, 1,
    outputs, output_values, 1,
    NULL, 0, NULL, status);

// Check status again...
// Process output_values[0]...
// Manual cleanup of EVERYTHING
TF_DeleteTensor(output_values[0]);
TF_DeleteTensor(input);
TF_CloseSession(session, status);
TF_DeleteSession(session, status);
TF_DeleteGraph(graph);
TF_DeleteSessionOptions(opts);
TF_DeleteStatus(status);
```

Problems:
1. **No RAII** - Manual `TF_Delete*` calls, easy to leak on error paths
2. **No type safety** - `void*` data pointers, manual dtype tracking
3. **Verbose error handling** - Check `TF_Status` after every call
4. **Complex tensor management** - Manual allocation, dimension tracking
5. **Opaque operation names** - String-based operation lookup

### Existing C++ Options are Inadequate

| Option | Problem |
|--------|---------|
| TensorFlow C++ API | Unstable ABI, massive dependencies, build complexity |
| TensorFlow Lite | Limited model support, different ecosystem |
| ONNX Runtime | Requires model conversion, different format |
| Direct C API | Unsafe, verbose (see above) |

---

## Solution: TensorFlowWrap

TensorFlowWrap provides a modern C++20 interface that is:

```cpp
// TensorFlowWrap - safe, ergonomic, zero-overhead
auto model = tf_wrap::Model::Load("/path/to/model");

auto input = tf_wrap::Tensor::FromVector<float>({1, 3}, {1.0f, 2.0f, 3.0f});

auto result = model.runner()
    .feed(model.input("x"), input)
    .fetch(model.output("output"))
    .run_one();

// result is a Tensor with automatic lifetime management
for (float v : result.read<float>().span()) {
    std::cout << v << "\n";
}
// All resources automatically cleaned up
```

### Design Principles

#### 1. RAII Everything

Every TensorFlow resource is wrapped in an RAII class:

| C API | Wrapper | Cleanup |
|-------|---------|---------|
| `TF_Tensor*` | `Tensor` | `TF_DeleteTensor` |
| `TF_Session*` | `Session` | `TF_CloseSession` + `TF_DeleteSession` |
| `TF_Graph*` | `Graph` | `TF_DeleteGraph` |
| `TF_Status*` | `Status` | `TF_DeleteStatus` |
| `TF_SessionOptions*` | `SessionOptions` | `TF_DeleteSessionOptions` |
| `TF_Buffer*` | `Buffer` | `TF_DeleteBuffer` |
| `TF_DeviceList*` | `DeviceList` | `TF_DeleteDeviceList` |

Resources are automatically released when wrappers go out of scope, even on exception paths.

#### 2. Type-Safe Tensor Access

Tensors track their dtype and provide type-checked access:

```cpp
auto t = Tensor::FromScalar<float>(3.14f);

auto view = t.read<float>();    // OK: matching dtype
auto bad = t.read<int>();       // THROWS: dtype mismatch

// Compile-time enforcement via concepts
auto t2 = Tensor::FromScalar<std::string>("x");  // COMPILE ERROR: not a TensorScalar
```

The `TensorScalar` concept restricts tensor element types to those TensorFlow actually supports.

#### 3. Zero-Overhead Abstraction

Wrappers add no runtime cost over direct C API usage:

- **No virtual functions** - All calls are direct
- **No hidden allocations** - Wrappers store only the raw pointer
- **Inline everything** - Header-only with `[[nodiscard]]` hints
- **Move-only semantics** - No accidental copies of heavy resources

```cpp
// Session is just a pointer + shared_ptr to graph state
static_assert(sizeof(Session) == sizeof(void*) + sizeof(std::shared_ptr<void>));
```

#### 4. Exceptions for Errors

TensorFlow errors become C++ exceptions with rich context:

```cpp
try {
    auto t = tensor.read<int>();  // dtype is actually float
} catch (const tf_wrap::Error& e) {
    // "[TF_INVALID_ARGUMENT] Tensor::read<T> at file.cpp:42: dtype mismatch:
    //  requested INT32 but tensor has FLOAT"
    std::cerr << e.what() << "\n";
    std::cerr << "Code: " << e.code() << "\n";        // TF_INVALID_ARGUMENT
    std::cerr << "Operation: " << e.operation() << "\n";  // "Tensor::read<T>"
}
```

Every error includes:
- TensorFlow error code
- Operation that failed
- Source location (`std::source_location`)
- Contextual details (tensor name, index, etc.)

#### 5. Inference-Only Scope

TensorFlowWrap deliberately excludes:

| Excluded | Reason |
|----------|--------|
| Graph building | Use Python for model development |
| Training | Use Python/JAX/PyTorch |
| Gradient computation | Not needed for inference |
| Custom ops | Register via TF C API directly if needed |
| Eager execution | SavedModel inference is graph-based |

This keeps the library small, focused, and maintainable.

---

## Architecture

### Component Hierarchy

```
tf_wrap/
├── core.hpp              # Main include (pulls in everything below)
├── tensor.hpp            # Tensor creation, access, lifetime
├── session.hpp           # Session, SessionOptions, Run
├── graph.hpp             # Graph wrapper (read-only after session creation)
├── status.hpp            # Status wrapper, exception conversion
├── error.hpp             # Error exception class
├── model.hpp             # High-level Model + Runner API
├── format.hpp            # String formatting utilities
├── scope_guard.hpp       # RAII scope guard utility
├── small_vector.hpp      # Stack-allocating vector for hot paths
└── detail/
    └── raw_tensor_ptr.hpp  # Internal: raw pointer RAII
```

### Dependency Graph

```
error.hpp ←── status.hpp ←── tensor.hpp
                   ↑              ↑
                   └──── session.hpp ←── model.hpp
                              ↑
                         graph.hpp
```

### Key Classes

#### `Tensor`

Wraps `TF_Tensor*`. Provides:
- Factory methods: `FromScalar`, `FromVector`, `Zeros`, `Allocate`, `FromString`, `Clone`
- Type-safe access: `read<T>()`, `write<T>()`, `ToScalar<T>()`, `ToVector<T>()`
- Shape operations: `reshape()`, `shape()`, `rank()`, `num_elements()`
- Lifetime management: `keepalive()` for view safety

#### `Session`

Wraps `TF_Session*`. Provides:
- Construction from `Graph` (freezes graph)
- `Run()` with feeds/fetches/targets
- `resolve()` to convert operation names to `TF_Output`
- `ListDevices()` for device enumeration
- `LoadSavedModel()` static factory

#### `Model`

High-level convenience wrapper combining `Session` + `Graph`. Provides:
- `Load()` static factory for SavedModels
- `runner()` fluent API for inference
- Signature-based input/output lookup

#### `Runner`

Fluent builder for inference calls:
```cpp
auto result = model.runner()
    .feed(input_op, input_tensor)
    .fetch(output_op)
    .run_one();
```

---

## Design Decisions

### Why Header-Only?

1. **Simpler integration** - Just add include path, no library to link (except TensorFlow itself)
2. **Better inlining** - Compiler sees all code, can optimize across boundaries
3. **No ABI concerns** - No binary compatibility issues between library versions
4. **Easier deployment** - Copy headers, done

Trade-off: Longer compile times for users. Mitigated by keeping headers minimal.

### Why C++20?

Required features:
- **`std::span`** - Non-owning view for tensor data
- **`std::source_location`** - Automatic error location capture
- **Concepts** - `TensorScalar` constraint for type safety
- **`[[nodiscard]]`** - Prevent ignoring important return values

C++20 is well-supported by GCC 11+, Clang 14+, MSVC 19.29+.

### Why Exceptions Instead of Error Codes?

1. **Cleaner API** - No `StatusOr<T>` wrappers everywhere
2. **Cannot be ignored** - Unlike error codes
3. **Rich context** - Exception carries operation, location, details
4. **Natural for RAII** - Constructors can fail meaningfully

For performance-critical paths that must avoid exceptions, users can pre-validate inputs.

### Why Not Support Graph Building?

1. **Scope creep** - Graph building is complex, would double library size
2. **Python is better** - Model development belongs in Python
3. **Maintenance burden** - Would need to track TensorFlow op changes
4. **Not our use case** - Production inference doesn't build graphs

Users who need graph building should use the C API directly or Python.

### Why Freeze Graph After Session Creation?

TensorFlow requires graphs be immutable once a session uses them. We enforce this:

```cpp
Graph graph;
// graph.NewOperation(...) would work here (if we supported it)

Session session(graph);  // Graph is now frozen
// graph.NewOperation(...) would throw after this point
```

This prevents subtle bugs where graph modifications don't affect the session.

### Why `TensorView` Instead of Raw Pointers?

`TensorView<T>` provides:
1. **Bounds checking** - `at()` throws on out-of-bounds
2. **Lifetime safety** - Keeps source tensor alive
3. **Range-for support** - `for (float v : tensor.read<float>())`
4. **Span access** - `.span()` for STL algorithm compatibility

```cpp
// View keeps tensor alive even if original is destroyed
TensorView<const float> view = []{
    auto t = Tensor::FromVector<float>({3}, {1, 2, 3});
    return t.read<float>();
}();
// view is still valid here!
```

### Why `SmallVector` for Run Buffers?

`Session::Run` needs temporary arrays for inputs/outputs. We use `SmallVector<T, N>` which:
1. **Stack allocates** up to N elements (common case)
2. **Heap allocates** if more needed (rare case)
3. **Zero overhead** for typical inference (1-4 inputs/outputs)

```cpp
SmallVector<TF_Output, 8> input_ops;   // 8 on stack, no malloc
SmallVector<TF_Tensor*, 8> input_vals;
```

---

## Performance Considerations

### Hot Path: `Session::Run`

The inference hot path is:
1. Prepare input tensors (user code)
2. Call `Session::Run` (TensorFlowWrap)
3. TensorFlow executes graph (TensorFlow)
4. Extract output tensors (TensorFlowWrap)

TensorFlowWrap overhead in steps 2 and 4:
- **SmallVector** stack allocation for feeds/fetches
- **Pointer copies** to build C API arrays
- **Exception check** on TensorFlow status

This is negligible compared to actual inference time (typically milliseconds).

### Memory Management

| Operation | Allocation |
|-----------|------------|
| `Tensor::FromScalar` | One `TF_AllocateTensor` call |
| `Tensor::FromVector` | One `TF_AllocateTensor` call |
| `Session::Run` | Output tensors allocated by TensorFlow |
| `TensorView` | No allocation (view into existing tensor) |
| `ToVector` | One `std::vector` allocation (copies data out) |

### Thread Safety

- `Session::Run` is thread-safe (TensorFlow guarantee)
- Each thread should have its own input tensors
- Output tensors are independent per `Run` call

```cpp
// Safe: concurrent inference
auto model = Model::Load("model");

std::vector<std::thread> workers;
for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back([&model, i] {
        auto input = Tensor::FromVector<float>({1}, {float(i)});
        auto result = model.runner()
            .feed(model.input("x"), input)
            .fetch(model.output("y"))
            .run_one();
        // Process result...
    });
}
```

---

## Testing Strategy

See [TEST_STYLE_GUIDE.md](TEST_STYLE_GUIDE.md) for details.

### Dual Testing Approach

| Test Type | Purpose | Framework |
|-----------|---------|-----------|
| Stub tests | API surface, error handling, edge cases | doctest |
| Real TF tests | Actual inference, result verification | Custom |

### CI Matrix

- **Compilers**: GCC 13/14, Clang 17/18, MSVC, Apple Clang
- **Platforms**: Linux, Windows, macOS
- **TensorFlow versions**: 2.13.0 through 2.18.1
- **Sanitizers**: AddressSanitizer, UndefinedBehaviorSanitizer

---

## Future Directions

### Planned

- [ ] Batch inference helpers
- [ ] Async `Run` with futures
- [ ] Memory-mapped tensor support
- [ ] Profiling integration (TensorFlow profiler)

### Considered but Deferred

- **GPU memory management** - Complex, user can use TF session config
- **Quantization helpers** - Model-specific, better in Python
- **ONNX import** - Different format, out of scope

### Explicitly Out of Scope

- Training / gradient computation
- Graph building / modification
- Custom op registration
- TensorFlow Lite integration
- Distributed inference

---

## References

- [TensorFlow C API Documentation](https://www.tensorflow.org/install/lang_c)
- [TensorFlow SavedModel Format](https://www.tensorflow.org/guide/saved_model)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

---

*TensorFlowWrap Design Document v1.0 -- January 2026*
