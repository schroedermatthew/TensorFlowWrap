// tf/all.hpp
// Umbrella header for TensorFlow C++20 wrapper
//
// Include this single header to get all wrapper functionality.

#pragma once

#include "tf_wrap/status.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/operation.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/ops.hpp"  // 160 TensorFlow operations

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
// THREAD SAFETY:
// ─────────────────────────────────────────────────────────────────────────────
//   Session::Run() is thread-safe (TensorFlow's guarantee)
//   Graph is frozen after Session creation (immutable)
//   Tensors are NOT thread-safe - don't share mutably across threads
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
// EXAMPLE USAGE:
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
// LOAD SAVEDMODEL (recommended for production):
// ─────────────────────────────────────────────────────────────────────────────
//   auto [session, graph] = tf_wrap::Session::LoadSavedModel("/path/to/model");
//   auto result = session.Run({tf_wrap::Feed{"input", tensor}}, {tf_wrap::Fetch{"output"}});
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
