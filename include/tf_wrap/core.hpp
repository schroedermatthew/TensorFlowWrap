// tf/core.hpp
// Umbrella header for TensorFlow C++20 wrapper
//
// Include this single header to get core wrapper functionality (ops are opt-in).

#pragma once

#include "tf_wrap/scope_guard.hpp"   // RAII scope-exit cleanup (used internally)
#include "tf_wrap/small_vector.hpp"  // Stack-optimized vector (available for users)
#include "tf_wrap/error.hpp"         // Structured exception type (tf_wrap::Error)
#include "tf_wrap/status.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/operation.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/facade.hpp" // Ergonomic layer: TensorName, Runner, Model

// ============================================================================
// TensorFlow C++20 Wrapper - Quick Reference
// ============================================================================
//
// CORE CLASSES:
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::Tensor         - RAII wrapper for TF_Tensor
//   tf_wrap::Graph          - RAII wrapper for TF_Graph
//   tf_wrap::Session        - RAII wrapper for TF_Session
//   tf_wrap::Operation      - Non-owning handle to TF_Operation
//   tf_wrap::Status         - RAII wrapper for TF_Status
//   tf_wrap::SessionOptions - RAII wrapper for TF_SessionOptions
//
// UTILITY CLASSES (used internally, available for users):
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::ScopeGuard       - RAII cleanup on scope exit (like Go's defer)
//   tf_wrap::ScopeGuardOnFail - Cleanup only on exception (for rollback)
//   tf_wrap::ScopeGuardOnSuccess - Cleanup only on normal exit (for commit)
//   tf_wrap::SmallVector<T,N> - Stack-optimized vector for small collections
//
// ERGONOMIC LAYER (facade.hpp):
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::TensorName     - Parse "op:index" strings
//   tf_wrap::Runner         - Fluent API for session execution
//   tf_wrap::Model          - High-level SavedModel facade
//
// THREAD SAFETY:
// ─────────────────────────────────────────────────────────────────────────────
//   Session::Run() is thread-safe (TensorFlow's guarantee)
//   Graph is frozen after Session creation (immutable)
//   Tensors are NOT thread-safe - don't share mutably across threads
//   Runner is NOT thread-safe - do not share a Runner instance across threads
//   For multi-threaded serving, each request should have its own input tensors
//
// TENSOR ACCESS:
// ─────────────────────────────────────────────────────────────────────────────
//   auto view = tensor.read<float>();   // Read-only view (keeps tensor alive)
//   auto view = tensor.write<float>();  // Writable view (keeps tensor alive)
//   for (float x : view) { ... }
//
//   // Or callback-based:
//   tensor.with_read<float>([](std::span<const float> s) { ... });
//   tensor.with_write<float>([](std::span<float> s) { ... });
//
//   // Or direct pointer access:
//   float* p = tensor.data<float>();
//
// BASIC USAGE:
// ─────────────────────────────────────────────────────────────────────────────
//   // Build graph
//   tf_wrap::Graph graph;
//
//   auto tensor = tf_wrap::Tensor::FromVector<float>({1, 4}, {1, 2, 3, 4});
//
//   auto const_op = graph.NewOperation("Const", "my_const")
//       .SetAttrTensor("value", tensor.handle())
//       .SetAttrType("dtype", TF_FLOAT)
//       .Finish();
//
//   graph.NewOperation("Identity", "output")
//       .AddInput(const_op, 0)
//       .Finish();
//
//   // Run inference
//   tf_wrap::Session session(graph);
//   auto results = session.Run({tf_wrap::Fetch{"output", 0}});
//
//   // Access results
//   auto view = results[0].read<float>();
//   for (float x : view) {
//       std::cout << x << " ";
//   }
//
// FLUENT RUNNER API (recommended):
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::Session session(graph);
//   auto result = tf_wrap::Runner(session)
//       .feed("input:0", input_tensor)
//       .fetch("output:0")
//       .run_one();
//
// LOAD SAVEDMODEL (recommended for production):
// ─────────────────────────────────────────────────────────────────────────────
//   auto model = tf_wrap::Model::Load("/path/to/model");
//   auto result = model("input:0", input_tensor, "output:0");
//
//   // Or with runner for multiple inputs/outputs:
//   auto results = model.runner()
//       .feed("input1:0", tensor1)
//       .feed("input2:0", tensor2)
//       .fetch("output1:0")
//       .fetch("output2:0")
//       .run();
//
// OPTIONAL OP WRAPPERS:
// ─────────────────────────────────────────────────────────────────────────────
//   This core umbrella header intentionally does NOT include the generated op
//   wrappers. Include what you use:
//
//     #include "tf_wrap/ops/math.hpp"
//     #include "tf_wrap/ops/matrix.hpp"
//     #include "tf_wrap/ops/array.hpp"
//
//   Or include all wrapper categories:
//
//     #include "tf_wrap/ops/all.hpp"
//
// DTYPE-INFERRED FACADES (optional):
// ─────────────────────────────────────────────────────────────────────────────
//   The dtype-inferred graph-building helpers live in tf_wrap/facade_ops.hpp:
//
//     #include "tf_wrap/facade_ops.hpp"
//     using namespace tf_wrap::facade;
//     auto c1 = Scalar<float>(graph, "c1", 1.0f);
//     auto c2 = Scalar<float>(graph, "c2", 2.0f);
//     auto sum = Add(graph, "sum", c1.output(0), c2.output(0));
//
// DEVICE ENUMERATION:
// ─────────────────────────────────────────────────────────────────────────────
//   auto devices = session.ListDevices();
//   for (int i = 0; i < devices.count(); ++i) {
//       auto dev = devices.at(i);
//       std::cout << dev.name << " (" << dev.type << ")\n";
//   }
//   if (session.HasGPU()) { /* use GPU */ }
//
// SCALAR TYPES SUPPORTED:
// ─────────────────────────────────────────────────────────────────────────────
//   float, double
//   int8_t, int16_t, int32_t, int64_t
//   uint8_t, uint16_t, uint32_t, uint64_t
//   bool, complex<float>, complex<double>
//
