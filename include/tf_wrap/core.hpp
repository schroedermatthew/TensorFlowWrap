// tf_wrap/core.hpp
// Umbrella header for TensorFlow C++20 wrapper - Production Inference Edition
//
// Optimized for inference workloads. All APIs are handle-based for maximum
// performance. Resolve operation names to TF_Output once at startup, then
// use handles in the hot path.

#pragma once

#include "tf_wrap/scope_guard.hpp"   // RAII scope-exit cleanup (used internally)
#include "tf_wrap/small_vector.hpp"  // Stack-optimized vector (available for users)
#include "tf_wrap/error.hpp"         // Structured exception type (tf_wrap::Error)
#include "tf_wrap/status.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/operation.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/facade.hpp"        // RunnerBuilder, Runner, Model

// ============================================================================
// TensorFlow C++20 Wrapper - Production Inference Quick Reference
// ============================================================================
//
// CORE CLASSES:
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::Tensor         - RAII wrapper for TF_Tensor
//   tf_wrap::Graph          - RAII wrapper for TF_Graph (read-only after load)
//   tf_wrap::Session        - RAII wrapper for TF_Session
//   tf_wrap::SessionOptions - RAII wrapper for TF_SessionOptions
//   tf_wrap::Buffer         - RAII wrapper for TF_Buffer
//   tf_wrap::Feed           - Input specification (TF_Output + Tensor)
//   tf_wrap::Fetch          - Output specification (TF_Output)
//   tf_wrap::Target         - Target operation (TF_Operation*)
//   tf_wrap::RunContext     - Reusable buffers for zero-allocation hot path
//
// ERGONOMIC LAYER:
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::RunnerBuilder  - Fluent builder used at startup
//   tf_wrap::Runner         - Compiled inference signature (fast hot path)
//   tf_wrap::Model          - High-level SavedModel facade with production features
//
// THREAD SAFETY:
// ─────────────────────────────────────────────────────────────────────────────
//   Session::Run() is thread-safe (TensorFlow's guarantee)
//   Graph is frozen after Session creation (immutable)
//   Tensors are NOT thread-safe - don't share mutably across threads
//   Runner is thread-safe (immutable) once compiled
//   Runner::Context is NOT thread-safe - use one Context per thread for zero-allocation hot path
//   For multi-threaded serving, each request should have its own input tensors
//
// PRODUCTION USAGE (RECOMMENDED):
// ─────────────────────────────────────────────────────────────────────────────
//   // 1. Load model at startup
//   auto model = tf_wrap::Model::Load("/path/to/saved_model");
//
//   // 2. Resolve endpoints ONCE at startup (not per-request!)
//   auto [input, output] = model.resolve("serving_default_input:0", 
//                                         "StatefulPartitionedCall:0");
//
//   // 3. Warmup to trigger JIT compilation
//   auto dummy = tf_wrap::Tensor::Zeros<float>({1, 224, 224, 3});
//   model.warmup(input, dummy, output);
//
//   // 4. Hot path - use handles, no string parsing
//   // 3. Compile a fast inference signature ONCE at startup
//   auto run = model.runner()
//       .feed(input)
//       .fetch(output)
//       .compile();
//
//   // 4. Hot path - treat the signature like a function
//   while (serving) {
//       auto input_tensor = get_request_tensor();
//       auto result = run(input_tensor);   // "wow this is easy"
//       send_response(std::move(result));
//   }
//
// ZERO-ALLOCATION HOT PATH (OPTIONAL):
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::RunContext ctx;  // Create once, reuse
//   while (serving) {
//       ctx.reset();
//       ctx.add_feed(input, get_request_tensor());
//       ctx.add_fetch(output);
//       auto results = model.session().Run(ctx);
//   }
//
// BATCH INFERENCE:
// ─────────────────────────────────────────────────────────────────────────────
//   // Multiple calls (one TF call per input)
//   auto results = model.BatchRun(input, input_tensors, output);
//
//   // Single call (stack inputs, run once, split outputs)
//   auto results = model.BatchRunStacked(input, input_tensors, output);
//
// INPUT VALIDATION:
// ─────────────────────────────────────────────────────────────────────────────
//   // Check dtype matches expected
//   auto error = model.validate_input(input, tensor);
//   if (!error.empty()) { /* handle error */ }
//
//   // Or throw on mismatch
//   model.require_valid_input(input, tensor);
//
// TENSOR ACCESS:
// ─────────────────────────────────────────────────────────────────────────────
//   auto view = tensor.read<float>();   // Read-only view
//   auto view = tensor.write<float>();  // Writable view
//   for (float x : view) { ... }
//
//   // Or direct pointer access
//   float* p = tensor.data<float>();
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
// SUPPORTED DTYPES:
// ─────────────────────────────────────────────────────────────────────────────
//   float, double
//   int8_t, int16_t, int32_t, int64_t
//   uint8_t, uint16_t, uint32_t, uint64_t
//   bool, complex<float>, complex<double>
//
