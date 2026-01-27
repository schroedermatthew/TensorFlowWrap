# TensorFlowWrap Testing Gaps & Remediation Plan

## Analysis Summary

The previous test plan focused on soak/stress testing. This revised plan addresses the **actual gaps**: untested public APIs, missing CMake packaging, and underutilized stub execution capabilities.

### Current Coverage

| Component | Stub Tests | Real TF Tests | Status |
|-----------|:----------:|:-------------:|--------|
| Tensor creation/access | ✓ | ✓ | **Good** |
| Graph introspection | ✓ | ✓ | **Good** |
| Operation metadata | ✓ | ✓ | **Good** |
| Lifecycle/RAII | ✓ | — | **Good** |
| Session::Run execution | ✗ | Minimal | **Gap** |
| Runner::run execution | ✗ | Minimal | **Gap** |
| BatchRun/BatchRunStacked | Error-only | ✗ | **Critical Gap** |
| Session::resolve parsing | ✗ | ✗ | **Critical Gap** |
| RunContext execution | ✗ | ✗ | **Gap** |
| CMake install/find_package | — | — | **Broken** |

### Key Insight

The stub (`third_party/tf_stub/tf_c_stub.cpp`) supports **actual graph execution** for Identity, Add, Mul, MatMul, and more. But tests only use it as a "doesn't crash" mock. This is the biggest missed opportunity.

---

## Phase 1: Fix CMake Packaging (P0)

**Problem:** `cmake/` directory is missing. CMakeLists.txt fails at line 132 looking for `cmake/TensorFlowWrap-config.cmake.in`.

### 1.1 Create Package Config Template

Create `cmake/TensorFlowWrapConfig.cmake.in`:

```cmake
@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/TensorFlowWrapTargets.cmake")

check_required_components(TensorFlowWrap)

# Provide tf::wrapper and tf::stub aliases
if(NOT TARGET tf::wrapper)
    add_library(tf::wrapper ALIAS TensorFlowWrap::wrapper)
endif()

if(TARGET TensorFlowWrap::stub AND NOT TARGET tf::stub)
    add_library(tf::stub ALIAS TensorFlowWrap::stub)
endif()
```

### 1.2 Add CI Job

```yaml
cmake-integration:
  name: CMake Integration Test
  runs-on: ubuntu-24.04
  
  steps:
    - uses: actions/checkout@v4
    
    - name: Configure
      run: cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$PWD/install
    
    - name: Build
      run: cmake --build build
    
    - name: Install
      run: cmake --install build
    
    - name: Test find_package
      run: |
        cd tests/cmake_integration_test
        ./run_test.sh $PWD/../../install
```

### 1.3 Deliverables

- [ ] `cmake/TensorFlowWrapConfig.cmake.in`
- [ ] CI job that builds, installs, and runs integration test
- [ ] Verify `find_package(TensorFlowWrap)` works

**Effort:** 0.5 days

---

## Phase 2: Stub Execution Tests (P0)

**Problem:** Stub supports graph execution but tests don't use it. Session/Runner execution logic is only tested on Linux real-TF jobs.

### 2.1 Graph Builder Helper

Create `tests/test_helpers.hpp`:

```cpp
#pragma once
#include "tf_wrap/core.hpp"

namespace test_helpers {

// Build a minimal executable graph: y = x (Identity)
inline std::tuple<tf_wrap::Graph, TF_Output, TF_Output> 
build_identity_graph() {
    tf_wrap::Graph graph;
    
    // Placeholder input
    auto input_desc = TF_NewOperation(graph.handle(), "Placeholder", "x");
    TF_SetAttrType(input_desc, "dtype", TF_FLOAT);
    tf_wrap::Status st;
    TF_Operation* input_op = TF_FinishOperation(input_desc, st.get());
    st.throw_if_error("build_identity_graph: Placeholder");
    
    // Identity output
    auto output_desc = TF_NewOperation(graph.handle(), "Identity", "y");
    TF_AddInput(output_desc, TF_Output{input_op, 0});
    TF_Operation* output_op = TF_FinishOperation(output_desc, st.get());
    st.throw_if_error("build_identity_graph: Identity");
    
    return {std::move(graph), TF_Output{input_op, 0}, TF_Output{output_op, 0}};
}

// Build: y = x + x (tests Add op)
inline std::tuple<tf_wrap::Graph, TF_Output, TF_Output>
build_add_graph() {
    tf_wrap::Graph graph;
    
    auto input_desc = TF_NewOperation(graph.handle(), "Placeholder", "x");
    TF_SetAttrType(input_desc, "dtype", TF_FLOAT);
    tf_wrap::Status st;
    TF_Operation* input_op = TF_FinishOperation(input_desc, st.get());
    st.throw_if_error("build_add_graph: Placeholder");
    
    auto add_desc = TF_NewOperation(graph.handle(), "AddV2", "y");
    TF_AddInput(add_desc, TF_Output{input_op, 0});
    TF_AddInput(add_desc, TF_Output{input_op, 0});
    TF_Operation* add_op = TF_FinishOperation(add_desc, st.get());
    st.throw_if_error("build_add_graph: AddV2");
    
    return {std::move(graph), TF_Output{input_op, 0}, TF_Output{add_op, 0}};
}

// Build: y = x * 2 (tests Mul with const)
inline std::tuple<tf_wrap::Graph, TF_Output, TF_Output>
build_mul_const_graph() {
    tf_wrap::Graph graph;
    tf_wrap::Status st;
    
    // Placeholder
    auto input_desc = TF_NewOperation(graph.handle(), "Placeholder", "x");
    TF_SetAttrType(input_desc, "dtype", TF_FLOAT);
    TF_Operation* input_op = TF_FinishOperation(input_desc, st.get());
    st.throw_if_error("Placeholder");
    
    // Const(2.0)
    auto const_tensor = tf_wrap::Tensor::FromScalar<float>(2.0f);
    auto const_desc = TF_NewOperation(graph.handle(), "Const", "two");
    TF_SetAttrType(const_desc, "dtype", TF_FLOAT);
    TF_SetAttrTensor(const_desc, "value", const_tensor.handle(), st.get());
    st.throw_if_error("Const attr");
    TF_Operation* const_op = TF_FinishOperation(const_desc, st.get());
    st.throw_if_error("Const");
    
    // Mul
    auto mul_desc = TF_NewOperation(graph.handle(), "Mul", "y");
    TF_AddInput(mul_desc, TF_Output{input_op, 0});
    TF_AddInput(mul_desc, TF_Output{const_op, 0});
    TF_Operation* mul_op = TF_FinishOperation(mul_desc, st.get());
    st.throw_if_error("Mul");
    
    return {std::move(graph), TF_Output{input_op, 0}, TF_Output{mul_op, 0}};
}

} // namespace test_helpers
```

### 2.2 Session Execution Tests

Add to `tests/test_session.cpp`:

```cpp
#include "test_helpers.hpp"

// ============================================================================
// Session::Run Execution Tests (stub evaluates these!)
// ============================================================================

TEST_CASE("Session::Run - Identity graph produces correct output") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(42.0f);
    auto results = session.Run(
        {{input_op, input}},
        {{output_op}}
    );
    
    REQUIRE(results.size() == 1);
    CHECK(results[0].ToScalar<float>() == 42.0f);
}

TEST_CASE("Session::Run - Add graph computes x + x") {
    auto [graph, input_op, output_op] = test_helpers::build_add_graph();
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(21.0f);
    auto results = session.Run(
        {{input_op, input}},
        {{output_op}}
    );
    
    REQUIRE(results.size() == 1);
    CHECK(results[0].ToScalar<float>() == 42.0f);  // 21 + 21
}

TEST_CASE("Session::Run - Mul const graph computes x * 2") {
    auto [graph, input_op, output_op] = test_helpers::build_mul_const_graph();
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(21.0f);
    auto results = session.Run(
        {{input_op, input}},
        {{output_op}}
    );
    
    REQUIRE(results.size() == 1);
    CHECK(results[0].ToScalar<float>() == 42.0f);  // 21 * 2
}

TEST_CASE("Session::Run - Vector input/output") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    auto input = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto results = session.Run(
        {{input_op, input}},
        {{output_op}}
    );
    
    REQUIRE(results.size() == 1);
    auto output = results[0].ToVector<float>();
    CHECK(output == std::vector<float>{1.0f, 2.0f, 3.0f});
}

TEST_CASE("Session::Run - Multiple fetches") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(5.0f);
    auto results = session.Run(
        {{input_op, input}},
        {{output_op}, {output_op}}  // Fetch same op twice
    );
    
    REQUIRE(results.size() == 2);
    CHECK(results[0].ToScalar<float>() == 5.0f);
    CHECK(results[1].ToScalar<float>() == 5.0f);
}
```

### 2.3 Runner Execution Tests

Add to `tests/test_facade.cpp`:

```cpp
#include "test_helpers.hpp"

// ============================================================================
// Runner Execution Tests
// ============================================================================

TEST_CASE("Runner::run_one - Identity graph") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(99.0f);
    auto result = Runner(session)
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    CHECK(result.ToScalar<float>() == 99.0f);
}

TEST_CASE("Runner::run - Multiple outputs") {
    auto [graph, input_op, output_op] = test_helpers::build_add_graph();
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(10.0f);
    auto results = Runner(session)
        .feed(input_op, input)
        .fetch(output_op)
        .fetch(output_op)
        .run();
    
    REQUIRE(results.size() == 2);
    CHECK(results[0].ToScalar<float>() == 20.0f);
    CHECK(results[1].ToScalar<float>() == 20.0f);
}

TEST_CASE("Runner::clear - Reuse runner") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    Runner runner(session);
    
    // First run
    auto input1 = Tensor::FromScalar<float>(1.0f);
    auto result1 = runner.feed(input_op, input1).fetch(output_op).run_one();
    CHECK(result1.ToScalar<float>() == 1.0f);
    
    // Clear and run again
    runner.clear();
    auto input2 = Tensor::FromScalar<float>(2.0f);
    auto result2 = runner.feed(input_op, input2).fetch(output_op).run_one();
    CHECK(result2.ToScalar<float>() == 2.0f);
}
```

### 2.4 Deliverables

- [ ] `tests/test_helpers.hpp` with graph builders
- [ ] 5+ Session::Run execution tests in `test_session.cpp`
- [ ] 3+ Runner execution tests in `test_facade.cpp`
- [ ] All tests pass on all platforms (stub executes them)

**Effort:** 1 day

---

## Phase 3: Batching Tests (P0)

**Problem:** `BatchRun` and `BatchRunStacked` have zero positive correctness tests. These are complex functions with different semantics that must be encoded in tests.

### Key Semantic Distinction

| API | TF_SessionRun calls | Ragged shapes | TF_STRING |
|-----|:-------------------:|:-------------:|:---------:|
| `BatchRun` | N (one per input) | ✓ Supported | ✓ Supported |
| `BatchRunStacked` | 1 (true batching) | ✗ Must be uniform | ✗ Rejected |

This distinction is critical for production: `BatchRun` is for convenience, `BatchRunStacked` is for throughput.

### 3.1 Stub Instrumentation

Add to `third_party/tf_stub/tf_c_stub.cpp`:

```cpp
// Session run counter for testing
static std::atomic<int> g_session_run_count{0};

extern "C" void TF_StubResetCounters() {
    g_session_run_count = 0;
}

extern "C" int TF_StubGetSessionRunCount() {
    return g_session_run_count.load();
}

// In TF_SessionRun:
void TF_SessionRun(...) {
    ++g_session_run_count;  // Add this line
    // ... existing implementation
}
```

Add to `include/tf_wrap/stub_control.hpp`:

```cpp
extern "C" void TF_StubResetCounters();
extern "C" int TF_StubGetSessionRunCount();
```

### 3.2 BatchRun Tests (N independent runs)

Create `tests/test_batching.cpp`:

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tf_wrap/core.hpp"
#include "tf_wrap/stub_control.hpp"
#include "test_helpers.hpp"

using namespace tf_wrap;

// ============================================================================
// Session::BatchRun - N independent TF_SessionRun calls
// ============================================================================

TEST_CASE("BatchRun - Scalar inputs produce matching outputs") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<float>(2.5f));
    inputs.push_back(Tensor::FromScalar<float>(-3.0f));
    
    auto results = session.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 3);
    CHECK(results[0].ToScalar<float>() == 1.0f);
    CHECK(results[1].ToScalar<float>() == 2.5f);
    CHECK(results[2].ToScalar<float>() == -3.0f);
}

TEST_CASE("BatchRun - Supports ragged shapes (different sizes per item)") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromVector<float>({1}, {1.0f}));
    inputs.push_back(Tensor::FromVector<float>({3}, {2.0f, 3.0f, 4.0f}));
    inputs.push_back(Tensor::FromVector<float>({2}, {5.0f, 6.0f}));
    
    auto results = session.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 3);
    CHECK(results[0].shape() == std::vector<int64_t>{1});
    CHECK(results[1].shape() == std::vector<int64_t>{3});
    CHECK(results[2].shape() == std::vector<int64_t>{2});
    CHECK(results[0].ToVector<float>() == std::vector<float>{1.0f});
    CHECK(results[1].ToVector<float>() == std::vector<float>{2.0f, 3.0f, 4.0f});
}

TEST_CASE("BatchRun - Works with int32 dtype") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph_int32();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<int32_t>(42));
    inputs.push_back(Tensor::FromScalar<int32_t>(-7));
    
    auto results = session.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 2);
    CHECK(results[0].ToScalar<int32_t>() == 42);
    CHECK(results[1].ToScalar<int32_t>() == -7);
}

TEST_CASE("BatchRun - Supports TF_STRING (variable-length dtype)") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph_string();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromString("short"));
    inputs.push_back(Tensor::FromString("a much longer string"));
    
    auto results = session.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 2);
    // String tensor verification depends on your ToString API
}

TEST_CASE("BatchRun - Empty input returns empty output") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;  // Empty
    auto results = session.BatchRun(input_op, inputs, output_op);
    
    CHECK(results.empty());
}

TEST_CASE("BatchRun - Calls TF_SessionRun N times") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<float>(2.0f));
    inputs.push_back(Tensor::FromScalar<float>(3.0f));
    
    TF_StubResetCounters();
    auto results = session.BatchRun(input_op, inputs, output_op);
    
    CHECK(TF_StubGetSessionRunCount() == 3);  // One call per input
}

TEST_CASE("BatchRun - Moved-from session throws FAILED_PRECONDITION") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session s1(graph);
    Session s2 = std::move(s1);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    
    bool threw = false;
    try {
        s1.BatchRun(input_op, inputs, output_op);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("BatchRun - Null tensor in inputs throws INVALID_ARGUMENT") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.emplace_back();  // Default/null tensor
    
    CHECK_THROWS_AS(session.BatchRun(input_op, inputs, output_op), Error);
}

TEST_CASE("BatchRun - Both overloads produce same results") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<float>(2.0f));
    
    // span overload
    auto results1 = session.BatchRun(input_op, std::span<const Tensor>(inputs), output_op);
    // vector overload
    auto results2 = session.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results1.size() == results2.size());
    for (size_t i = 0; i < results1.size(); ++i) {
        CHECK(results1[i].ToScalar<float>() == results2[i].ToScalar<float>());
    }
}

// ============================================================================
// Session::BatchRunStacked - Single TF_SessionRun with stacking
// ============================================================================

TEST_CASE("BatchRunStacked - Scalar items stack and split correctly") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<float>(2.0f));
    inputs.push_back(Tensor::FromScalar<float>(3.0f));
    
    auto results = session.BatchRunStacked(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 3);
    CHECK(results[0].ToScalar<float>() == 1.0f);
    CHECK(results[1].ToScalar<float>() == 2.0f);
    CHECK(results[2].ToScalar<float>() == 3.0f);
}

TEST_CASE("BatchRunStacked - 2D items stack and split correctly") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromVector<float>({2, 2}, {1, 2, 3, 4}));
    inputs.push_back(Tensor::FromVector<float>({2, 2}, {5, 6, 7, 8}));
    
    auto results = session.BatchRunStacked(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 2);
    CHECK(results[0].shape() == std::vector<int64_t>{2, 2});
    CHECK(results[1].shape() == std::vector<int64_t>{2, 2});
    CHECK(results[0].ToVector<float>() == std::vector<float>{1, 2, 3, 4});
    CHECK(results[1].ToVector<float>() == std::vector<float>{5, 6, 7, 8});
}

TEST_CASE("BatchRunStacked - Calls TF_SessionRun exactly once") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<float>(2.0f));
    inputs.push_back(Tensor::FromScalar<float>(3.0f));
    
    TF_StubResetCounters();
    auto results = session.BatchRunStacked(input_op, inputs, output_op);
    
    CHECK(TF_StubGetSessionRunCount() == 1);  // Single batched call
}

TEST_CASE("BatchRunStacked - Results match BatchRun for uniform inputs") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromVector<float>({3}, {1, 2, 3}));
    inputs.push_back(Tensor::FromVector<float>({3}, {4, 5, 6}));
    
    auto stacked_results = session.BatchRunStacked(input_op, inputs, output_op);
    
    // Recreate inputs for BatchRun (can't reuse moved tensors)
    std::vector<Tensor> inputs2;
    inputs2.push_back(Tensor::FromVector<float>({3}, {1, 2, 3}));
    inputs2.push_back(Tensor::FromVector<float>({3}, {4, 5, 6}));
    auto batch_results = session.BatchRun(input_op, inputs2, output_op);
    
    REQUIRE(stacked_results.size() == batch_results.size());
    for (size_t i = 0; i < stacked_results.size(); ++i) {
        CHECK(stacked_results[i].ToVector<float>() == batch_results[i].ToVector<float>());
    }
}

TEST_CASE("BatchRunStacked - Empty input returns empty output") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;  // Empty
    auto results = session.BatchRunStacked(input_op, inputs, output_op);
    
    CHECK(results.empty());
}

TEST_CASE("BatchRunStacked - Moved-from session throws FAILED_PRECONDITION") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session s1(graph);
    Session s2 = std::move(s1);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    
    bool threw = false;
    try {
        s1.BatchRunStacked(input_op, inputs, output_op);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("BatchRunStacked - First tensor null throws INVALID_ARGUMENT") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.emplace_back();  // Null tensor first
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    
    bool threw = false;
    try {
        session.BatchRunStacked(input_op, inputs, output_op);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
    }
    CHECK(threw);
}

TEST_CASE("BatchRunStacked - Later tensor null throws with correct index") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.emplace_back();  // Null tensor at index 1
    inputs.push_back(Tensor::FromScalar<float>(3.0f));
    
    bool threw = false;
    try {
        session.BatchRunStacked(input_op, inputs, output_op);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
        CHECK(e.index() == 1);
    }
    CHECK(threw);
}

TEST_CASE("BatchRunStacked - Dtype mismatch throws with correct index") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<int32_t>(2));  // Wrong dtype at index 1
    
    bool threw = false;
    try {
        session.BatchRunStacked(input_op, inputs, output_op);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
        CHECK(e.index() == 1);
    }
    CHECK(threw);
}

TEST_CASE("BatchRunStacked - Shape mismatch throws with correct index") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromVector<float>({2}, {1, 2}));
    inputs.push_back(Tensor::FromVector<float>({3}, {3, 4, 5}));  // Wrong shape at index 1
    
    bool threw = false;
    try {
        session.BatchRunStacked(input_op, inputs, output_op);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
        CHECK(e.index() == 1);
    }
    CHECK(threw);
}

TEST_CASE("BatchRunStacked - TF_STRING rejected (variable-length dtype)") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph_string();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromString("hello"));
    inputs.push_back(Tensor::FromString("world"));
    
    bool threw = false;
    try {
        session.BatchRunStacked(input_op, inputs, output_op);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
        // Message should mention "variable-length dtype"
    }
    CHECK(threw);
}

TEST_CASE("BatchRunStacked - Output batch dimension mismatch throws INTERNAL") {
    // Build graph where output is a constant (doesn't depend on batch size)
    auto [graph, input_op, const_output_op] = test_helpers::build_const_output_graph();
    Session session(graph);
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<float>(2.0f));
    inputs.push_back(Tensor::FromScalar<float>(3.0f));
    // Expecting batch dim 3, but const output has no batch dim
    
    bool threw = false;
    try {
        session.BatchRunStacked(input_op, inputs, const_output_op);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INTERNAL);
        // Message should mention "output batch dimension mismatch"
    }
    CHECK(threw);
}
```

### 3.3 Additional Graph Builders

Add to `tests/test_helpers.hpp`:

```cpp
// Identity graph for int32
inline std::tuple<tf_wrap::Graph, TF_Output, TF_Output>
build_identity_graph_int32() {
    tf_wrap::Graph graph;
    tf_wrap::Status st;
    
    auto* d_x = TF_NewOperation(graph.handle(), "Placeholder", "x");
    TF_SetAttrType(d_x, "dtype", TF_INT32);
    TF_Operation* x = TF_FinishOperation(d_x, st.get());
    st.throw_if_error("Placeholder");
    
    auto* d_y = TF_NewOperation(graph.handle(), "Identity", "y");
    TF_AddInput(d_y, TF_Output{x, 0});
    TF_Operation* y = TF_FinishOperation(d_y, st.get());
    st.throw_if_error("Identity");
    
    return {std::move(graph), TF_Output{x, 0}, TF_Output{y, 0}};
}

// Identity graph for string (if stub supports it)
inline std::tuple<tf_wrap::Graph, TF_Output, TF_Output>
build_identity_graph_string() {
    tf_wrap::Graph graph;
    tf_wrap::Status st;
    
    auto* d_x = TF_NewOperation(graph.handle(), "Placeholder", "x");
    TF_SetAttrType(d_x, "dtype", TF_STRING);
    TF_Operation* x = TF_FinishOperation(d_x, st.get());
    st.throw_if_error("Placeholder");
    
    auto* d_y = TF_NewOperation(graph.handle(), "Identity", "y");
    TF_AddInput(d_y, TF_Output{x, 0});
    TF_Operation* y = TF_FinishOperation(d_y, st.get());
    st.throw_if_error("Identity");
    
    return {std::move(graph), TF_Output{x, 0}, TF_Output{y, 0}};
}

// Graph where output is constant (for testing batch mismatch)
inline std::tuple<tf_wrap::Graph, TF_Output, TF_Output>
build_const_output_graph() {
    tf_wrap::Graph graph;
    tf_wrap::Status st;
    
    // Placeholder (ignored)
    auto* d_x = TF_NewOperation(graph.handle(), "Placeholder", "x");
    TF_SetAttrType(d_x, "dtype", TF_FLOAT);
    TF_Operation* x = TF_FinishOperation(d_x, st.get());
    st.throw_if_error("Placeholder");
    
    // Const output (scalar, no batch dimension)
    auto const_tensor = tf_wrap::Tensor::FromScalar<float>(42.0f);
    auto* d_c = TF_NewOperation(graph.handle(), "Const", "c");
    TF_SetAttrTensor(d_c, "value", const_tensor.handle(), st.get());
    st.throw_if_error("Const attr");
    TF_SetAttrType(d_c, "dtype", TF_FLOAT);
    TF_Operation* c = TF_FinishOperation(d_c, st.get());
    st.throw_if_error("Const");
    
    return {std::move(graph), TF_Output{x, 0}, TF_Output{c, 0}};
}
```

### 3.4 Real TF Integration Tests

Add to `tests/test_facade_tf.cpp`:

```cpp
// ============================================================================
// Model::BatchRun with Real TensorFlow
// ============================================================================

TEST(model_batch_run_ragged_shapes) {
    auto model = Model::Load("test_savedmodel");
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Different-sized inputs (ragged)
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromVector<float>({1}, {1.0f}));
    inputs.push_back(Tensor::FromVector<float>({3}, {2.0f, 3.0f, 4.0f}));
    
    auto results = model.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 2);
    REQUIRE(results[0].shape() == std::vector<int64_t>{1});
    REQUIRE(results[1].shape() == std::vector<int64_t>{3});
    
    // y = x * 2 + 1
    REQUIRE_CLOSE(results[0].ToScalar<float>(), 3.0f, 0.001f);  // 1*2+1
    auto v = results[1].ToVector<float>();
    REQUIRE_CLOSE(v[0], 5.0f, 0.001f);   // 2*2+1
    REQUIRE_CLOSE(v[1], 7.0f, 0.001f);   // 3*2+1
    REQUIRE_CLOSE(v[2], 9.0f, 0.001f);   // 4*2+1
}

// ============================================================================
// Model::BatchRunStacked with Real TensorFlow
// ============================================================================

TEST(model_batch_run_stacked_scalars) {
    auto model = Model::Load("test_savedmodel");
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Scalar inputs stack into [N] which matches model signature float32[None]
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(0.0f));
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<float>(2.0f));
    
    auto results = model.BatchRunStacked(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 3);
    // y = x * 2 + 1
    REQUIRE_CLOSE(results[0].ToScalar<float>(), 1.0f, 0.001f);  // 0*2+1
    REQUIRE_CLOSE(results[1].ToScalar<float>(), 3.0f, 0.001f);  // 1*2+1
    REQUIRE_CLOSE(results[2].ToScalar<float>(), 5.0f, 0.001f);  // 2*2+1
}

TEST(model_batch_run_stacked_shape_mismatch_throws) {
    auto model = Model::Load("test_savedmodel");
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Mismatched shapes should throw before TF
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromVector<float>({2}, {2.0f, 3.0f}));
    
    REQUIRE_THROWS(model.BatchRunStacked(input_op, inputs, output_op));
}

TEST(model_batch_run_stacked_empty_returns_empty) {
    auto model = Model::Load("test_savedmodel");
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    std::vector<Tensor> inputs;
    auto results = model.BatchRunStacked(input_op, inputs, output_op);
    
    REQUIRE(results.empty());
}
```

### 3.5 Deliverables

- [ ] Stub instrumentation: `TF_StubResetCounters()`, `TF_StubGetSessionRunCount()`
- [ ] `tests/test_batching.cpp` with 20+ tests
- [ ] Graph builders for int32, string, const-output scenarios
- [ ] Real TF tests in `test_facade_tf.cpp` (4+)
- [ ] CI runs new test file

### 3.6 Properties Encoded by Tests

| Test | Encodes |
|------|---------|
| BatchRun ragged shapes | "BatchRun accepts variable request shapes" |
| BatchRun TF_STRING | "BatchRun works with variable-length dtypes" |
| BatchRun call count | "BatchRun makes N independent calls" |
| BatchRunStacked call count | "BatchRunStacked makes 1 batched call" |
| BatchRunStacked equivalence | "Results match when shapes are uniform" |
| BatchRunStacked TF_STRING rejected | "Stacking requires fixed-size dtypes" |
| BatchRunStacked shape mismatch | "Stacking requires uniform shapes" |
| BatchRunStacked output mismatch | "Output must have batch dimension" |

**Effort:** 1.5 days (increased from 0.5 due to scope)

---

## Phase 4: Resolve Tests (P0)

**Problem:** `Session::resolve()` / `Model::resolve()` name parsing is untested.

### 4.1 Stub Tests

Add to `tests/test_session.cpp`:

```cpp
TEST_CASE("Session::resolve - Parses 'name:0' format") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    auto resolved = session.resolve("x:0");
    CHECK(resolved.oper != nullptr);
    CHECK(resolved.index == 0);
}

TEST_CASE("Session::resolve - Defaults to index 0") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    auto resolved = session.resolve("x");  // No :N suffix
    CHECK(resolved.oper != nullptr);
    CHECK(resolved.index == 0);
}

TEST_CASE("Session::resolve - Invalid index throws OUT_OF_RANGE") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    // "x" has only 1 output (index 0)
    CHECK_THROWS_AS(session.resolve("x:1"), Error);
    CHECK_THROWS_AS(session.resolve("x:99"), Error);
}

TEST_CASE("Session::resolve - Unknown op throws NOT_FOUND") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    CHECK_THROWS_AS(session.resolve("nonexistent"), Error);
    CHECK_THROWS_AS(session.resolve("nonexistent:0"), Error);
}

TEST_CASE("Session::resolve - Non-numeric suffix treated as op name") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    // "x:abc" should look for op named "x:abc", not op "x" index "abc"
    CHECK_THROWS_AS(session.resolve("x:abc"), Error);  // NOT_FOUND
}
```

### 4.2 Real TF Tests

Add to `tests/test_session_tf.cpp`:

```cpp
TEST(session_resolve_parses_colon_format) {
    auto [session, graph] = Session::LoadSavedModel("test_savedmodel");
    
    // Model has "serving_default_x" input
    auto input = session.resolve("serving_default_x:0");
    REQUIRE(input.oper != nullptr);
    REQUIRE(input.index == 0);
    
    // Should also work without :0
    auto input2 = session.resolve("serving_default_x");
    REQUIRE(input2.oper != nullptr);
    REQUIRE(input2.index == 0);
}

TEST(session_resolve_out_of_range_index) {
    auto [session, graph] = Session::LoadSavedModel("test_savedmodel");
    
    // serving_default_x has 1 output, index 99 is invalid
    bool threw = false;
    try {
        session.resolve("serving_default_x:99");
    } catch (const Error& e) {
        threw = true;
        REQUIRE(e.code() == TF_OUT_OF_RANGE);
    }
    REQUIRE(threw);
}
```

### 4.3 Deliverables

- [ ] 5+ resolve parsing tests in stub
- [ ] 2+ resolve tests with real TF
- [ ] Test error codes (NOT_FOUND vs OUT_OF_RANGE)

**Effort:** 0.5 days

---

## Phase 5: RunContext Execution Tests (P1)

**Problem:** RunContext is only tested as a container, not for actual execution.

### 5.1 Tests

Add to `tests/test_session.cpp`:

```cpp
TEST_CASE("Session::Run with RunContext - Executes correctly") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    RunContext ctx;
    auto input = Tensor::FromScalar<float>(123.0f);
    
    ctx.add_feed(input_op, input);
    ctx.add_fetch(output_op);
    
    auto results = session.Run(ctx);
    
    REQUIRE(results.size() == 1);
    CHECK(results[0].ToScalar<float>() == 123.0f);
}

TEST_CASE("Session::Run with RunContext - Reuse across calls") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    RunContext ctx;
    
    for (int i = 0; i < 5; ++i) {
        ctx.reset();
        auto input = Tensor::FromScalar<float>(static_cast<float>(i));
        ctx.add_feed(input_op, input);
        ctx.add_fetch(output_op);
        
        auto results = session.Run(ctx);
        REQUIRE(results.size() == 1);
        CHECK(results[0].ToScalar<float>() == static_cast<float>(i));
    }
}

TEST_CASE("Session::Run with RunContext - Keepalive works") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    RunContext ctx;
    
    // Add tensor that goes out of scope
    {
        auto input = Tensor::FromScalar<float>(999.0f);
        ctx.add_feed(input_op, input);
        ctx.add_fetch(output_op);
        // input goes out of scope here, but ctx holds keepalive
    }
    
    // Should still work because ctx has keepalive
    auto results = session.Run(ctx);
    REQUIRE(results.size() == 1);
    CHECK(results[0].ToScalar<float>() == 999.0f);
}
```

### 5.2 Deliverables

- [ ] RunContext execution test
- [ ] RunContext reuse test
- [ ] RunContext keepalive test

**Effort:** 0.5 days

---

## Phase 6: Strengthen Existing Tests (P1)

**Problem:** Many stub tests assert "doesn't crash" instead of "is correct".

### 6.1 Device List Tests

Fix in `tests/test_session.cpp`:

```cpp
// Before (weak)
TEST_CASE("Session - ListDevices") {
    // ...
    CHECK(devices.count() >= 0);  // Can't fail
}

// After (strict)
TEST_CASE("Session - ListDevices returns CPU in stub") {
    Graph graph;
    Session session(graph);
    auto devices = session.ListDevices();
    
    REQUIRE(devices.count() == 1);  // Stub returns exactly 1 device
    
    auto cpu = devices.at(0);
    CHECK(cpu.type == "CPU");
    CHECK(cpu.is_cpu() == true);
    CHECK(cpu.is_gpu() == false);
}

TEST_CASE("Session - HasGPU returns false in stub") {
    Graph graph;
    Session session(graph);
    CHECK(session.HasGPU() == false);  // Stub has no GPU
}
```

### 6.2 Error Code Verification

When testing throws, verify the error code:

```cpp
// Before (weak)
CHECK_THROWS(tensor.read<int32_t>());

// After (strict)
TEST_CASE("Tensor::read - Wrong dtype throws INVALID_ARGUMENT") {
    auto t = Tensor::FromScalar<float>(1.0f);
    
    bool threw = false;
    try {
        auto view = t.read<int32_t>();
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
    }
    CHECK(threw);
}
```

### 6.3 Deliverables

- [ ] Fix 5+ weak assertions to be strict
- [ ] Add error code verification to throw tests

**Effort:** 0.5 days

---

## Phase 7: Error Path Tests (P1)

**Problem:** Many defensive checks in Session/Runner code are never exercised.

### 7.1 Tests

```cpp
TEST_CASE("Runner::run_one - Throws when fetches != 1") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(1.0f);
    
    // Zero fetches
    CHECK_THROWS_AS(
        Runner(session).feed(input_op, input).run_one(),
        Error
    );
    
    // Two fetches
    CHECK_THROWS_AS(
        Runner(session).feed(input_op, input)
            .fetch(output_op).fetch(output_op).run_one(),
        Error
    );
}

TEST_CASE("Runner::feed - Null tensor throws") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session session(graph);
    
    Tensor empty;  // Moved-from or default
    CHECK_THROWS_AS(
        Runner(session).feed(input_op, empty),
        Error
    );
}

TEST_CASE("Session::Run - On moved-from session throws") {
    auto [graph, input_op, output_op] = test_helpers::build_identity_graph();
    Session s1(graph);
    Session s2 = std::move(s1);
    
    auto input = Tensor::FromScalar<float>(1.0f);
    CHECK_THROWS_AS(
        s1.Run({{input_op, input}}, {{output_op}}),
        Error
    );
}
```

### 7.2 Deliverables

- [ ] Runner error path tests (3+)
- [ ] Session moved-from tests (2+)
- [ ] Null handle tests

**Effort:** 0.5 days

---

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 1. CMake Packaging | P0 | 0.5 days | Unblocks users |
| 2. Stub Execution | P0 | 1 day | Validates core logic cross-platform |
| 3. BatchRun Tests | P0 | 0.5 days | Tests complex, risky code |
| 4. Resolve Tests | P0 | 0.5 days | Tests user-facing API |
| 5. RunContext Execution | P1 | 0.5 days | Tests optimization path |
| 6. Strengthen Assertions | P1 | 0.5 days | Catches regressions |
| 7. Error Paths | P1 | 0.5 days | Validates defensive code |

**Total: ~4.5 days**

---

## Files to Create/Modify

### New Files

```
cmake/TensorFlowWrapConfig.cmake.in    # Package config
tests/test_helpers.hpp                  # Graph builders for tests
```

### Modified Files

```
tests/test_session.cpp     # +20 tests (execution, batch, resolve, runcontext)
tests/test_facade.cpp      # +10 tests (runner execution, batch)
tests/test_session_tf.cpp  # +5 tests (resolve with real TF)
.github/workflows/ci.yml   # +1 job (cmake integration)
```

---

## Success Criteria

After implementing all phases:

- [ ] `cmake -S . -B build && cmake --build build` succeeds
- [ ] `find_package(TensorFlowWrap)` works in external project
- [ ] Stub tests execute actual graphs (Identity, Add, Mul)
- [ ] BatchRun/BatchRunStacked have positive correctness tests
- [ ] Session::resolve parsing has 5+ test cases
- [ ] RunContext works in actual Session::Run calls
- [ ] All tests pass on Linux, macOS, Windows

---

## Comparison: Old Plan vs New Plan

| Old Plan (Soak Focus) | New Plan (Gap Focus) |
|-----------------------|----------------------|
| 10K iteration stress tests | Stub execution tests |
| Memory leak detection | BatchRun correctness |
| Concurrency tests | Resolve parsing tests |
| Real model testing | CMake packaging |
| Performance benchmarks | RunContext execution |

The old plan assumed the core logic was correct and focused on production resilience. This new plan addresses the reality: **core execution paths lack basic correctness tests**.

The soak tests remain valuable but should come after Phase 1-4 are complete.

---

## Phase 8: Stub Infrastructure Enhancement (P0)

**Problem:** The stub declares `TF_StubSetNextError` but never implements it. There's no way to:
1. Count how many times `TF_SessionRun` was called (critical for verifying BatchRunStacked semantics)
2. Inject errors deterministically (for testing error paths)
3. Force allocation failures (for testing `TF_RESOURCE_EXHAUSTED` handling)

### 8.1 Implement Call Counters

Add to `third_party/tf_stub/tf_c_stub.cpp`:

```cpp
#include <atomic>

// Stub Instrumentation
static std::atomic<int> g_session_run_count{0};
static std::atomic<int> g_tensor_alloc_count{0};
static std::atomic<int> g_tensor_delete_count{0};

extern "C" void TF_StubResetCounters() {
    g_session_run_count = 0;
    g_tensor_alloc_count = 0;
    g_tensor_delete_count = 0;
}

extern "C" int TF_StubGetSessionRunCount() {
    return g_session_run_count.load();
}

// In TF_SessionRun, add: ++g_session_run_count;
// In TF_AllocateTensor, add: ++g_tensor_alloc_count;
// In TF_DeleteTensor, add: if (t) ++g_tensor_delete_count;
```

### 8.2 Implement Error Injection

```cpp
struct NextError {
    std::string api;
    TF_Code code{TF_OK};
    std::string message;
};
static NextError g_next_error;
static std::mutex g_next_error_mutex;

extern "C" void TF_StubSetNextError(const char* api, TF_Code code, const char* message) {
    std::lock_guard lock(g_next_error_mutex);
    g_next_error = {api ? api : "", code, message ? message : ""};
}

extern "C" void TF_StubClearNextError() {
    std::lock_guard lock(g_next_error_mutex);
    g_next_error = {};
}

static bool check_injected_error(const char* api, TF_Status* status) {
    std::lock_guard lock(g_next_error_mutex);
    if (g_next_error.code != TF_OK && g_next_error.api == api) {
        set_status(status, g_next_error.code, g_next_error.message.c_str());
        g_next_error = {};
        return true;
    }
    return false;
}
```

### 8.3 RAII Helpers in stub_control.hpp

```cpp
namespace tf_wrap::stub {

struct CounterScope {
    CounterScope() { TF_StubResetCounters(); }
    ~CounterScope() { TF_StubResetCounters(); }
    int session_runs() const { return TF_StubGetSessionRunCount(); }
};

struct ErrorScope {
    ErrorScope(const char* api, TF_Code code, const char* msg) {
        TF_StubSetNextError(api, code, msg);
    }
    ~ErrorScope() { TF_StubClearNextError(); }
};

} // namespace tf_wrap::stub
```

### 8.4 Tests Using Instrumentation

```cpp
TEST_CASE("BatchRun calls TF_SessionRun N times") {
    stub::CounterScope counters;
    // ... setup ...
    session.BatchRun(input_op, inputs, output_op);  // 5 inputs
    CHECK(counters.session_runs() == 5);
}

TEST_CASE("BatchRunStacked calls TF_SessionRun exactly once") {
    stub::CounterScope counters;
    // ... setup ...
    session.BatchRunStacked(input_op, inputs, output_op);  // 5 inputs
    CHECK(counters.session_runs() == 1);
}

TEST_CASE("Error injection propagates TF_Code correctly") {
    stub::ErrorScope error("TF_SessionRun", TF_INTERNAL, "Simulated");
    // ... Run should throw with code == TF_INTERNAL
}
```

### 8.5 Deliverables

- [ ] Implement `TF_StubSetNextError` (declared but missing)
- [ ] Implement call counters
- [ ] RAII helpers
- [ ] Tests verifying call counts

**Effort:** 1 day

---

## Phase 9: Comprehensive Error Path Testing (P1)

### 9.1 Session Error Paths

```cpp
TEST_CASE("Session::Run - Unsupported op throws UNIMPLEMENTED") {
    // Build graph with op stub doesn't support
    // Expect TF_UNIMPLEMENTED
}

TEST_CASE("Session::Run - Empty fetches returns empty") {
    auto results = session.Run({{input_op, input}}, {});
    CHECK(results.empty());
}

TEST_CASE("Session::Run - Const-only graph needs no feeds") {
    // Build const-only graph
    auto results = session.Run({}, {{const_op}});
    CHECK(results[0].ToScalar<float>() == expected);
}
```

### 9.2 Runner Error Paths

```cpp
TEST_CASE("Runner::run_one - Zero fetches throws") {
    CHECK_THROWS(Runner(session).feed(op, t).run_one());
}

TEST_CASE("Runner::run_one - Two fetches throws") {
    CHECK_THROWS(Runner(session).fetch(op).fetch(op).run_one());
}

TEST_CASE("Runner::feed - Null TF_Tensor* throws") {
    CHECK_THROWS(Runner(session).feed(op, nullptr));
}
```

### 9.3 Model Error Paths

```cpp
TEST_CASE("Model::resolve - Unloaded throws FAILED_PRECONDITION") {
    Model m;  // Default = unloaded
    CHECK_THROWS(m.resolve("x:0"));
}
```

**Effort:** 0.5 days

---

## Phase 10: Moved-From State Tests (P1)

```cpp
TEST_CASE("Tensor - Moved-from has null handle") {
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = std::move(t1);
    CHECK(t1.handle() == nullptr);
}

TEST_CASE("Session - Moved-from Run throws FAILED_PRECONDITION") {
    Session s1(graph);
    Session s2 = std::move(s1);
    CHECK_THROWS(s1.Run(...));  // code == TF_FAILED_PRECONDITION
}

TEST_CASE("Session - Moved-from BatchRun throws") { /* ... */ }
TEST_CASE("Session - Moved-from BatchRunStacked throws") { /* ... */ }
TEST_CASE("Session - Moved-from resolve throws") { /* ... */ }
TEST_CASE("Graph - Moved-from has null handle") { /* ... */ }
TEST_CASE("Status - Moved-from has null handle") { /* ... */ }
```

**Effort:** 0.5 days

---

## Updated Summary

| Phase | Priority | Effort | Tests |
|-------|----------|--------|-------|
| 1. CMake Packaging | P0 | 0.5 days | 1 CI job |
| 2. Stub Execution | P0 | 1 day | ~15 |
| 3. Batching Tests | P0 | 1.5 days | ~25 |
| 4. Resolve Tests | P0 | 0.5 days | ~7 |
| 5. RunContext Execution | P1 | 0.5 days | ~5 |
| 6. Strengthen Assertions | P1 | 0.5 days | ~10 fixes |
| 7. Error Paths | P1 | 0.5 days | ~8 |
| 8. Stub Infrastructure | P0 | 1 day | ~8 |
| 9. Comprehensive Errors | P1 | 0.5 days | ~10 |
| 10. Moved-From Tests | P1 | 0.5 days | ~10 |

**Total: ~7 days, ~100+ new tests**

---

## Key Properties Encoded

| Property | Test |
|----------|------|
| BatchRun accepts ragged shapes | BatchRun ragged test |
| BatchRunStacked requires uniform shapes | Shape mismatch test |
| BatchRun supports TF_STRING | TF_STRING test |
| BatchRunStacked rejects TF_STRING | Variable-length dtype test |
| BatchRun makes N calls | Counter test |
| BatchRunStacked makes 1 call | Counter test |
| Moved-from throws FAILED_PRECONDITION | Moved-from tests |
| TF errors propagate correctly | Error injection tests |
| Allocation failures → RESOURCE_EXHAUSTED | Error injection test |
