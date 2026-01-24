// test_comprehensive_bugs_new.cpp
// Edge case tests using doctest
//
// Run with: ./test_edge_cases
// Run stress tests: ./test_edge_cases -ts=stress

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)  // discarding return value of function with [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "tf_wrap/all.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("H1: Clone of empty tensor is safe") {
    tf_wrap::Tensor empty;
    auto clone = empty.Clone();
    CHECK(clone.empty());
}

TEST_CASE("H1: Clone preserves data correctly (single-threaded)") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto clone = tensor.Clone();
    
    auto orig_data = tensor.ToVector<float>();
    auto clone_data = clone.ToVector<float>();
    
    CHECK(orig_data.size() == clone_data.size());
    for (std::size_t i = 0; i < orig_data.size(); ++i) {
        CHECK(orig_data[i] == clone_data[i]);
    }
    
    // Verify they're independent
    {
        auto view = clone.write<float>();
        view[0] = 999.0f;
    }
    CHECK(tensor.ToVector<float>()[0] == 1.0f);  // Original unchanged
}

// NOTE: "H1-BUG: Clone race detection aggressive" test removed.
// This is a single-threaded library - users are responsible for external synchronization.

// ============================================================================
// H2: DebugString() Tests (deadlock was only possible with locking)
// ============================================================================

TEST_CASE("H2-BUG: Graph DebugString deadlock check") {
    // This test originally demonstrated a potential deadlock when locking was enabled.
    // With locking removed in v5.0, deadlock is no longer possible.
    // Kept for historical reference and to verify DebugString still works.
    
    tf_wrap::Graph g;
    
    auto tensor = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "test_const")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Use async with timeout to detect deadlock (should never happen now)
    auto future = std::async(std::launch::async, [&]() {
        return g.DebugString();
    });
    
    auto status = future.wait_for(std::chrono::seconds(2));
    
    if (status == std::future_status::timeout) {
        std::cout << "     DEADLOCK DETECTED (expected before fix)\n";
        // The deadlock is the bug - we detected it
        throw std::runtime_error("Deadlock detected in Graph::DebugString");
    } else {
        std::string result = future.get();
        CHECK(result.find("test_const") != std::string::npos);
        CHECK(result.find("1 operations") != std::string::npos);
        std::cout << "     No deadlock (fix applied)\n";
    }
}

TEST_CASE("H2: Graph DebugString works") {
    tf_wrap::Graph g;
    
    auto tensor = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "myconst")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    std::string debug = g.DebugString();
    CHECK(debug.find("myconst") != std::string::npos);
}

TEST_CASE("H2: Graph num_operations standalone") {
    tf_wrap::Graph g;
    
    CHECK(g.num_operations() == 0);
    
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "c1")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    CHECK(g.num_operations() == 1);
}

// ============================================================================
// Graph/Tensor/Session Coverage
// ============================================================================

TEST_CASE("Graph all methods work") {
    tf_wrap::Graph g;
    
    auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    
    (void)g.NewOperation("Const", "const1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)g.NewOperation("Const", "const2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    CHECK(g.num_operations() == 2);
    CHECK(g.GetOperation("const1").has_value());
    CHECK(g.GetOperation("nonexistent") == std::nullopt);
    
    // GetOperationOrThrow returns TF_Operation*
    TF_Operation* found = g.GetOperationOrThrow("const1");
    CHECK(std::string(TF_OperationName(found)) == "const1");
    
    bool threw = false;
    try {
        (void)g.GetOperationOrThrow("nonexistent");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Tensor read/write thread safety") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    
    // Writer
    {
        auto view = tensor.write<float>();
        view[0] = 100.0f;
    }
    
    // Reader
    {
        auto view = tensor.read<float>();
        CHECK(view[0] == 100.0f);
        CHECK(view[4] == 5.0f);
    }
}

// NOTE: "Tensor concurrent access is serialized" test removed.
// This test verified that mutex locking prevented torn reads during concurrent access.
// Since the policy-based locking system was removed in v5.0,
// tensors are no longer thread-safe and this test is not applicable.
// Users should not share mutable tensors across threads.

// NOTE: "Tensor allows concurrent readers" test also removed.
// This is a single-threaded library - users are responsible for external synchronization.

// ============================================================================
// View Lifetime Tests
// ============================================================================

TEST_CASE("View keeps tensor alive after Tensor destroyed") {
    // Create tensor in inner scope, but extract view
    std::optional<tf_wrap::Tensor::ReadView<float>> opt_view;
    
    {
        auto tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
        opt_view.emplace(tensor.read<float>());
    }  // tensor destroyed, but view keeps shared_ptr to state alive
    
    // View should still be valid!
    auto& view = *opt_view;
    CHECK(view.size() == 3);
    CHECK(view[0] == 1.0f);
    CHECK(view[1] == 2.0f);
    CHECK(view[2] == 3.0f);
}

TEST_CASE("Write view keeps tensor alive") {
    std::optional<tf_wrap::Tensor::WriteView<float>> opt_view;
    
    {
        auto tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
        opt_view.emplace(tensor.write<float>());
    }  // tensor destroyed, but view keeps shared_ptr to state alive
    
    auto& view = *opt_view;
    CHECK(view.size() == 3);
    view[0] = 100.0f;
    CHECK(view[0] == 100.0f);
}

// ============================================================================
// LoadSavedModel Lifetime Tests (M1)
// ============================================================================

// Note: This test would require a real SavedModel directory
// For stub testing, we just verify the API compiles correctly

TEST_CASE("M1: Session and Graph structured binding") {
    // This demonstrates correct usage
    tf_wrap::Graph graph;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Simulate what LoadSavedModel returns - both must stay alive
    // auto [session, graph] = Session::LoadSavedModel("/path");
    // session.Run(...);  // Works as long as graph is alive
}

// ============================================================================
// Error Consistency Tests (L1)
// ============================================================================

TEST_CASE("L1: Empty tensor error handling consistency") {
    tf_wrap::Tensor empty;
    
    // byte_size() returns 0 silently
    CHECK(empty.byte_size() == 0);
    
    // num_elements() returns 0 for empty tensor
    CHECK(empty.num_elements() == 0);
    
    // The inconsistency would be if num_elements threw but byte_size didn't
    // Both should handle empty gracefully
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

TEST_CASE("Multiple read views from same Tensor") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    auto view1 = tensor.read<float>();
    auto view2 = tensor.read<float>();  // Multiple read views are safe
    
    CHECK(view1[0] == view2[0]);
    CHECK(view1[1] == view2[1]);
    CHECK(view1[2] == view2[2]);
}

TEST_CASE("Tensor move leaves valid empty state") {
    auto t1 = tf_wrap::Tensor::FromScalar<float>(42.0f);
    auto t2 = std::move(t1);
    
    // t1 should be in valid empty state
    CHECK(t1.empty());
    CHECK(t1.handle() == nullptr);
    CHECK(t1.byte_size() == 0);
    
    // t2 should have the data
    CHECK(!t2.empty());
    CHECK(t2.ToScalar<float>() == 42.0f);
}

TEST_CASE("Graph move: moved-from must throw on use") {
    tf_wrap::Graph g1;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    CHECK(g1.num_operations() == 1);
    
    tf_wrap::Graph g2 = std::move(g1);
    
    // g2 should have the operation
    CHECK(g2.num_operations() == 1);
    CHECK(g2.valid());
    
    // g1 must be invalid and throw on use (not silently return empty)
    CHECK(!g1.valid());
    CHECK_THROWS_AS(g1.num_operations(), std::runtime_error);
}

