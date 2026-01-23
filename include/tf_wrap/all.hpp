// tf/all.hpp
// Umbrella header for TensorFlow C++20 wrapper
//
// Include this single header to get all wrapper functionality.

#pragma once

#include "tf_wrap/policy.hpp"
#include "tf_wrap/status.hpp"
#include "tf_wrap/guarded_span.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/operation.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/ops.hpp"  // 160 TensorFlow operations

// ============================================================================
// TensorFlow C++20 Wrapper - Quick Reference
// ============================================================================
//
// THREAD SAFETY POLICIES:
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::policy::NoLock      - Zero overhead, no synchronization (default)
//   tf_wrap::policy::Mutex       - Exclusive locking for thread-safe writes
//   tf_wrap::policy::SharedMutex - Reader-writer locks (concurrent reads OK)
//
// CORE CLASSES:
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::Tensor<Policy>      - RAII wrapper for TF_Tensor
//   tf_wrap::Graph<Policy>       - RAII wrapper for TF_Graph
//   tf_wrap::Session<Policy>     - RAII wrapper for TF_Session
//   tf_wrap::Operation           - Non-owning handle to TF_Operation
//   tf_wrap::Status              - RAII wrapper for TF_Status
//   tf_wrap::SessionOptions      - RAII wrapper for TF_SessionOptions
//   tf_wrap::GuardedSpan<T,G>    - Thread-safe view (span + lock guard)
//
// TYPE ALIASES:
// ─────────────────────────────────────────────────────────────────────────────
//   tf_wrap::FastTensor   = tf_wrap::Tensor<tf_wrap::policy::NoLock>
//   tf_wrap::SafeTensor   = tf_wrap::Tensor<tf_wrap::policy::Mutex>
//   tf_wrap::SharedTensor = tf_wrap::Tensor<tf_wrap::policy::SharedMutex>
//
//   tf_wrap::FastGraph    = tf_wrap::Graph<tf_wrap::policy::NoLock>
//   tf_wrap::SafeGraph    = tf_wrap::Graph<tf_wrap::policy::Mutex>
//   tf_wrap::SharedGraph  = tf_wrap::Graph<tf_wrap::policy::SharedMutex>
//
//   tf_wrap::FastSession  = tf_wrap::Session<tf_wrap::policy::NoLock>
//   tf_wrap::SafeSession  = tf_wrap::Session<tf_wrap::policy::Mutex>
//
// THREAD-SAFE TENSOR ACCESS:
// ─────────────────────────────────────────────────────────────────────────────
//   // View-based (lock held for view lifetime):
//   auto view = tensor.read<float>();    // Shared lock
//   auto view = tensor.write<float>();   // Exclusive lock
//   for (float x : view) { ... }
//
//   // Callback-based (hardest to misuse):
//   tensor.with_read<float>([](std::span<const float> s) { ... });
//   tensor.with_write<float>([](std::span<float> s) { ... });
//
//   // Unsafe (NO lock - caller must synchronize):
//   float* p = tensor.unsafe_data<float>();
//
// EXAMPLE USAGE:
// ─────────────────────────────────────────────────────────────────────────────
//   // Build graph
//   tf_wrap::Graph<> graph;
//   
//   auto tensor = tf_wrap::Tensor<>::FromVector<float>({1, 4}, {1, 2, 3, 4});
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
//   tf_wrap::Session<tf_wrap::policy::Mutex> session(graph);  // Thread-safe session
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
//   auto [session, graph] = tf_wrap::Session<>::LoadSavedModel("/path/to/model");
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
//   bool
//
