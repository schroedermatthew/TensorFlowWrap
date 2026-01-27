# Migration Guide: TensorFlow C API to TensorFlowWrap

## Scope

This guide covers migrating inference code from the raw TensorFlow C API to TensorFlowWrap. The wrapper eliminates manual resource cleanup, adds type-checked tensor access, and converts error codes to exceptions.

## Not Covered

- Training (inference only)
- Graph construction from scratch (use SavedModel export from Python)
- Custom op development
- TensorFlow Lite or ONNX

---

## Why Migrate

The TensorFlow C API requires manual management of every resource:

```c
TF_Status* status = TF_NewStatus();
TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, nbytes);
// ... use tensor ...
TF_DeleteTensor(tensor);   // Forget this: memory leak
TF_DeleteTensor(tensor);   // Call twice: heap corruption
TF_DeleteStatus(status);   // Forget this: memory leak
```

Common failure modes include leaked tensors, double-free crashes, use-after-free bugs, and ignored `TF_Status` errors. TensorFlowWrap eliminates these through RAII ownership—resources are released automatically when wrappers go out of scope.

---

## Type Mapping

| C API | TensorFlowWrap | Notes |
|-------|----------------|-------|
| `TF_Tensor*` | `tf_wrap::Tensor` | Move-only, RAII |
| `TF_Session*` | `tf_wrap::Session` | Closes and deletes automatically |
| `TF_Graph*` | `tf_wrap::Graph` | Shared ownership with Session |
| `TF_Status*` | `tf_wrap::Status` | Converts to exceptions |
| `TF_Output` | `TF_Output` | Unchanged |
| `TF_Operation*` | `TF_Operation*` | Unchanged (non-owning) |

---

## Migration Steps

### Step 1: Replace Tensor Creation

**Before (C API):**
```c
int64_t dims[] = {2, 3};
size_t nbytes = 6 * sizeof(float);
TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, nbytes);
if (!tensor) {
    return ERROR_OOM;
}
float* data = (float*)TF_TensorData(tensor);
for (int i = 0; i < 6; i++) {
    data[i] = (float)i;
}
// ... use tensor ...
TF_DeleteTensor(tensor);  // Must not forget
```

**After (TensorFlowWrap):**
```cpp
auto tensor = tf_wrap::Tensor::FromVector<float>(
    {2, 3},
    {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}
);
// Destructor handles cleanup
```

For uninitialized tensors that you'll write to:

```cpp
auto tensor = tf_wrap::Tensor::Allocate<float>({2, 3});
auto view = tensor.write<float>();
for (size_t i = 0; i < view.size(); ++i) {
    view[i] = static_cast<float>(i);
}
```

### Step 2: Replace Tensor Data Access

**Before (C API):**
```c
float* get_data(TF_Tensor* t) {
    return (float*)TF_TensorData(t);
}

void process() {
    TF_Tensor* t = create_tensor();
    float* data = get_data(t);
    TF_DeleteTensor(t);  // Bug: data is now dangling
    use_data(data);      // Use-after-free
}
```

**After (TensorFlowWrap):**
```cpp
void process() {
    auto tensor = create_tensor();
    auto view = tensor.read<float>();  // View keeps tensor alive
    
    auto tensor2 = std::move(tensor);  // Original is now empty
    
    use_data(view);  // Safe: view's keepalive prevents deallocation
}
```

The `read<T>()` method returns a view that holds a `shared_ptr` to the tensor's internal state. The underlying memory remains valid as long as any view exists.

### Step 3: Replace Session Creation

**Before (C API):**
```c
TF_Status* status = TF_NewStatus();
TF_Graph* graph = TF_NewGraph();
TF_SessionOptions* opts = TF_NewSessionOptions();
const char* tags[] = {"serve"};

TF_Session* session = TF_LoadSessionFromSavedModel(
    opts, NULL, "/path/to/model", tags, 1, graph, NULL, status);

if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "Error: %s\n", TF_Message(status));
    TF_DeleteStatus(status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(opts);
    return NULL;
}

TF_DeleteStatus(status);
TF_DeleteSessionOptions(opts);
// Must keep graph alive while session exists
// Must close and delete session when done
// Must delete graph after session is deleted
```

**After (TensorFlowWrap):**
```cpp
auto model = tf_wrap::Model::Load("/path/to/model");
// All resources cleaned up when model goes out of scope
```

If loading fails, `Model::Load` throws `tf_wrap::Error` with the TensorFlow error code and message.

### Step 4: Replace Session::Run

**Before (C API):**
```c
TF_Operation* input_op = TF_GraphOperationByName(graph, "serving_default_x");
TF_Operation* output_op = TF_GraphOperationByName(graph, "StatefulPartitionedCall");

TF_Output inputs[1] = {{input_op, 0}};
TF_Output outputs[1] = {{output_op, 0}};
TF_Tensor* input_values[1] = {input_tensor};
TF_Tensor* output_values[1] = {NULL};

TF_SessionRun(session, NULL,
    inputs, input_values, 1,
    outputs, output_values, 1,
    NULL, 0, NULL, status);

if (TF_GetCode(status) != TF_OK) {
    // Handle error...
}

// Process output_values[0]
// Must delete output_values[0] when done
```

**After (TensorFlowWrap):**
```cpp
// Resolve operation names once at startup
TF_Output input_op = model.resolve("serving_default_x:0");
TF_Output output_op = model.resolve("StatefulPartitionedCall:0");

// Run inference
auto result = model.runner()
    .feed(input_op, input_tensor)
    .fetch(output_op)
    .run_one();

// result is a Tensor with automatic cleanup
```

The `resolve()` method parses "operation_name:index" strings and looks up the operation in the graph. Call it once at startup and cache the `TF_Output` values for use in the inference loop.

### Step 5: Replace Status Checking

**Before (C API):**
```c
TF_Status* status = TF_NewStatus();
TF_SessionRun(session, ..., status);
if (TF_GetCode(status) != TF_OK) {
    const char* msg = TF_Message(status);
    TF_DeleteStatus(status);
    throw std::runtime_error(msg);
}
TF_DeleteStatus(status);
```

**After (TensorFlowWrap):**
```cpp
// Errors throw tf_wrap::Error automatically
auto result = model.runner().feed(...).fetch(...).run_one();
```

To handle errors explicitly:
```cpp
try {
    auto result = model.runner().feed(...).fetch(...).run_one();
} catch (const tf_wrap::Error& e) {
    std::cerr << "TensorFlow error: " << e.what() << "\n";
    std::cerr << "Code: " << e.code_name() << "\n";  // "INVALID_ARGUMENT"
    std::cerr << "Location: " << e.location().file_name() 
              << ":" << e.location().line() << "\n";
}
```

### Step 6: Remove Manual Cleanup

Delete all `TF_Delete*` calls. The wrappers handle cleanup automatically:

```cpp
// Before: manual cleanup everywhere
TF_DeleteTensor(tensor);
TF_CloseSession(session, status);
TF_DeleteSession(session, status);
TF_DeleteGraph(graph);
TF_DeleteStatus(status);

// After: nothing needed
// Destructors handle everything
```

---

## Complete Example

**Before (C API, ~60 lines):**
```c
#include <tensorflow/c/c_api.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* opts = TF_NewSessionOptions();
    const char* tags[] = {"serve"};
    
    TF_Session* session = TF_LoadSessionFromSavedModel(
        opts, NULL, "model", tags, 1, graph, NULL, status);
    
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Load failed: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(graph);
        TF_DeleteSessionOptions(opts);
        return 1;
    }
    TF_DeleteSessionOptions(opts);
    
    // Create input tensor
    int64_t dims[] = {1, 3};
    TF_Tensor* input = TF_AllocateTensor(TF_FLOAT, dims, 2, 3 * sizeof(float));
    float* data = (float*)TF_TensorData(input);
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f;
    
    // Setup run
    TF_Operation* input_op = TF_GraphOperationByName(graph, "serving_default_x");
    TF_Operation* output_op = TF_GraphOperationByName(graph, "StatefulPartitionedCall");
    TF_Output inputs[] = {{input_op, 0}};
    TF_Output outputs[] = {{output_op, 0}};
    TF_Tensor* input_vals[] = {input};
    TF_Tensor* output_vals[] = {NULL};
    
    TF_SessionRun(session, NULL,
        inputs, input_vals, 1,
        outputs, output_vals, 1,
        NULL, 0, NULL, status);
    
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Run failed: %s\n", TF_Message(status));
        TF_DeleteTensor(input);
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 1;
    }
    
    // Print output
    float* out = (float*)TF_TensorData(output_vals[0]);
    printf("Output: %f\n", out[0]);
    
    // Cleanup
    TF_DeleteTensor(output_vals[0]);
    TF_DeleteTensor(input);
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    return 0;
}
```

**After (TensorFlowWrap, ~20 lines):**
```cpp
#include "tf_wrap/core.hpp"
#include <iostream>

int main() {
    auto model = tf_wrap::Model::Load("model");
    
    // Resolve operations once
    auto input_op = model.resolve("serving_default_x:0");
    auto output_op = model.resolve("StatefulPartitionedCall:0");
    
    // Create input
    auto input = tf_wrap::Tensor::FromVector<float>({1, 3}, {1.0f, 2.0f, 3.0f});
    
    // Run inference
    auto result = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    // Print output
    std::cout << "Output: " << result.ToScalar<float>() << "\n";
    
    return 0;
}
```

---

## Threading

TensorFlowWrap preserves TensorFlow's threading model. `Session::Run` is thread-safe (TensorFlow's guarantee). Each thread should have its own input and output tensors:

```cpp
// Shared across threads
auto model = tf_wrap::Model::Load("model");
auto input_op = model.resolve("serving_default_x:0");
auto output_op = model.resolve("StatefulPartitionedCall:0");

// Per-thread
void handle_request(const float* data, size_t len) {
    auto input = tf_wrap::Tensor::FromVector<float>({1, len}, 
        std::vector<float>(data, data + len));
    
    auto result = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    // Process result...
}
```

---

## Error Handling

TensorFlowWrap converts all TensorFlow errors to `tf_wrap::Error` exceptions:

| TF_Code | Typical Cause |
|---------|---------------|
| `TF_INVALID_ARGUMENT` | Wrong dtype, shape mismatch |
| `TF_NOT_FOUND` | Operation name not in graph |
| `TF_RESOURCE_EXHAUSTED` | Out of memory |
| `TF_FAILED_PRECONDITION` | Session not initialized |

Every exception includes:
- `code()`: The `TF_Code` enum value
- `code_name()`: String like "INVALID_ARGUMENT"
- `context()`: The operation that failed
- `location()`: Source file and line via `std::source_location`
- `what()`: Full message combining all of the above

---

## Verification

After migration, verify correctness:

1. **Compare outputs**: Run the same inputs through old and new code, compare results
2. **Run sanitizers**: Build with `-fsanitize=address,undefined` to catch memory bugs
3. **Check for leaks**: The new code should have no `TF_Delete*` calls

```bash
# Build with sanitizers
g++ -std=c++20 -fsanitize=address,undefined -g \
    -I/path/to/tf_wrap/include \
    your_code.cpp \
    -ltensorflow -o your_app

# Run and check for issues
./your_app
```

---

## Rollback

TensorFlowWrap uses the same underlying TensorFlow C library. To roll back:

1. Replace `tf_wrap::Tensor` with `TF_Tensor*` and manual allocation
2. Replace `tf_wrap::Model` with `TF_Session*` and `TF_Graph*`
3. Add `TF_Delete*` calls at every cleanup point
4. Replace exception handling with `TF_Status*` checking

No data migration is needed—the TensorFlow model format is unchanged.

---

*Migration Guide v1.0 — January 2026*
