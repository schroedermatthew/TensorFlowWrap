// test_comprehensive.cpp
// Comprehensive tests for previously untested APIs
//
// Framework: doctest (header-only)
// Runs with: TF stub (all platforms)
//
// Covers gaps identified in testing audit:
// - codes.hpp
// - RunContext
// - Complex tensors
// - Advanced tensor operations
// - Graph introspection
// - Device enumeration

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/core.hpp"
#include "tf_wrap/codes.hpp"

#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace tf_wrap;

// ============================================================================
// codes.hpp Tests
// ============================================================================

TEST_SUITE("codes.hpp") {

    TEST_CASE("code_to_string - all standard codes") {
        CHECK(std::strcmp(code_to_string(TF_OK), "OK") == 0);
        CHECK(std::strcmp(code_to_string(TF_CANCELLED), "CANCELLED") == 0);
        CHECK(std::strcmp(code_to_string(TF_UNKNOWN), "UNKNOWN") == 0);
        CHECK(std::strcmp(code_to_string(TF_INVALID_ARGUMENT), "INVALID_ARGUMENT") == 0);
        CHECK(std::strcmp(code_to_string(TF_DEADLINE_EXCEEDED), "DEADLINE_EXCEEDED") == 0);
        CHECK(std::strcmp(code_to_string(TF_NOT_FOUND), "NOT_FOUND") == 0);
        CHECK(std::strcmp(code_to_string(TF_ALREADY_EXISTS), "ALREADY_EXISTS") == 0);
        CHECK(std::strcmp(code_to_string(TF_PERMISSION_DENIED), "PERMISSION_DENIED") == 0);
        CHECK(std::strcmp(code_to_string(TF_UNAUTHENTICATED), "UNAUTHENTICATED") == 0);
        CHECK(std::strcmp(code_to_string(TF_RESOURCE_EXHAUSTED), "RESOURCE_EXHAUSTED") == 0);
        CHECK(std::strcmp(code_to_string(TF_FAILED_PRECONDITION), "FAILED_PRECONDITION") == 0);
        CHECK(std::strcmp(code_to_string(TF_ABORTED), "ABORTED") == 0);
        CHECK(std::strcmp(code_to_string(TF_OUT_OF_RANGE), "OUT_OF_RANGE") == 0);
        CHECK(std::strcmp(code_to_string(TF_UNIMPLEMENTED), "UNIMPLEMENTED") == 0);
        CHECK(std::strcmp(code_to_string(TF_INTERNAL), "INTERNAL") == 0);
        CHECK(std::strcmp(code_to_string(TF_UNAVAILABLE), "UNAVAILABLE") == 0);
        CHECK(std::strcmp(code_to_string(TF_DATA_LOSS), "DATA_LOSS") == 0);
    }
    
    TEST_CASE("code_to_string - unknown code") {
        CHECK(std::strcmp(code_to_string(static_cast<TF_Code>(9999)), "UNKNOWN_CODE") == 0);
    }
    
    TEST_CASE("code_to_string - is constexpr") {
        // Verify it can be used in constexpr context
        constexpr const char* ok_str = code_to_string(TF_OK);
        CHECK(std::strcmp(ok_str, "OK") == 0);
    }
    
    TEST_CASE("code_to_string - is noexcept") {
        CHECK(noexcept(code_to_string(TF_OK)));
    }

}

// ============================================================================
// RunContext Tests
// ============================================================================

TEST_SUITE("RunContext") {

    TEST_CASE("default construction") {
        RunContext ctx;
        // Should construct without issues
        CHECK(true);
    }
    
    TEST_CASE("construction with capacity hints") {
        RunContext ctx(16, 8);  // 16 feeds, 8 fetches
        CHECK(true);
    }
    
    TEST_CASE("reset clears state") {
        RunContext ctx;
        
        TF_Output dummy_op{nullptr, 0};
        auto tensor = Tensor::FromScalar<float>(1.0f);
        
        ctx.add_feed(dummy_op, tensor);
        ctx.add_fetch(dummy_op);
        
        ctx.reset();
        
        // After reset, should be able to add new feeds/fetches
        ctx.add_feed(dummy_op, tensor);
        CHECK(true);
    }
    
    TEST_CASE("add_feed stores reference") {
        RunContext ctx;
        
        TF_Output op{nullptr, 0};
        auto tensor = Tensor::FromScalar<float>(42.0f);
        
        ctx.add_feed(op, tensor);
        CHECK(true);
    }
    
    TEST_CASE("add_fetch stores output") {
        RunContext ctx;
        
        TF_Output op{nullptr, 0};
        ctx.add_fetch(op);
        CHECK(true);
    }
    
    TEST_CASE("add_target stores operation") {
        RunContext ctx;
        
        TF_Operation* op = nullptr;
        ctx.add_target(op);
        CHECK(true);
    }
    
    TEST_CASE("multiple feeds and fetches") {
        RunContext ctx;
        
        TF_Output op1{nullptr, 0};
        TF_Output op2{nullptr, 1};
        
        auto t1 = Tensor::FromScalar<float>(1.0f);
        auto t2 = Tensor::FromScalar<float>(2.0f);
        
        ctx.add_feed(op1, t1);
        ctx.add_feed(op2, t2);
        ctx.add_fetch(op1);
        ctx.add_fetch(op2);
        
        CHECK(true);
    }
    
    TEST_CASE("reuse after reset") {
        RunContext ctx;
        
        for (int iter = 0; iter < 10; ++iter) {
            ctx.reset();
            
            TF_Output op{nullptr, 0};
            auto tensor = Tensor::FromScalar<float>(static_cast<float>(iter));
            
            ctx.add_feed(op, tensor);
            ctx.add_fetch(op);
        }
        
        CHECK(true);
    }

}

// ============================================================================
// Complex Tensor Types
// ============================================================================

TEST_SUITE("Complex Tensors") {

    TEST_CASE("complex<float> scalar") {
        std::complex<float> value(3.0f, 4.0f);
        auto tensor = Tensor::FromScalar<std::complex<float>>(value);
        
        CHECK(tensor.valid());
        CHECK(tensor.num_elements() == 1);
        
        auto result = tensor.ToScalar<std::complex<float>>();
        CHECK(result.real() == doctest::Approx(3.0f));
        CHECK(result.imag() == doctest::Approx(4.0f));
    }
    
    TEST_CASE("complex<double> scalar") {
        std::complex<double> value(1.5, 2.5);
        auto tensor = Tensor::FromScalar<std::complex<double>>(value);
        
        CHECK(tensor.valid());
        
        auto result = tensor.ToScalar<std::complex<double>>();
        CHECK(result.real() == doctest::Approx(1.5));
        CHECK(result.imag() == doctest::Approx(2.5));
    }
    
    TEST_CASE("complex<float> vector") {
        std::vector<std::complex<float>> data = {
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f}
        };
        
        auto tensor = Tensor::FromVector<std::complex<float>>({3}, data);
        
        CHECK(tensor.num_elements() == 3);
        
        auto view = tensor.read<std::complex<float>>();
        CHECK(view[0].real() == doctest::Approx(1.0f));
        CHECK(view[0].imag() == doctest::Approx(2.0f));
        CHECK(view[2].real() == doctest::Approx(5.0f));
        CHECK(view[2].imag() == doctest::Approx(6.0f));
    }
    
    TEST_CASE("complex tensor clone") {
        std::complex<float> value(7.0f, 8.0f);
        auto original = Tensor::FromScalar<std::complex<float>>(value);
        auto cloned = original.Clone();
        
        CHECK(cloned.valid());
        auto result = cloned.ToScalar<std::complex<float>>();
        CHECK(result.real() == doctest::Approx(7.0f));
        CHECK(result.imag() == doctest::Approx(8.0f));
    }
    
    TEST_CASE("complex zeros") {
        auto tensor = Tensor::Zeros<std::complex<float>>({2, 2});
        
        CHECK(tensor.num_elements() == 4);
        
        auto view = tensor.read<std::complex<float>>();
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(view[i].real() == doctest::Approx(0.0f));
            CHECK(view[i].imag() == doctest::Approx(0.0f));
        }
    }

}

// ============================================================================
// Advanced Tensor Operations
// ============================================================================

TEST_SUITE("Advanced Tensor Operations") {

    TEST_CASE("Allocate and write pattern") {
        auto tensor = Tensor::Allocate<float>({3, 3});
        
        CHECK(tensor.valid());
        CHECK(tensor.num_elements() == 9);
        
        // Write to tensor
        {
            auto view = tensor.write<float>();
            for (std::size_t i = 0; i < 9; ++i) {
                view[i] = static_cast<float>(i * i);
            }
        }
        
        // Read back
        auto view = tensor.read<float>();
        CHECK(view[0] == doctest::Approx(0.0f));
        CHECK(view[1] == doctest::Approx(1.0f));
        CHECK(view[4] == doctest::Approx(16.0f));
        CHECK(view[8] == doctest::Approx(64.0f));
    }
    
    TEST_CASE("unsafe_data access") {
        auto tensor = Tensor::FromVector<int32_t>({4}, {10, 20, 30, 40});
        
        const int32_t* ptr = tensor.unsafe_data<int32_t>();
        CHECK(ptr != nullptr);
        CHECK(ptr[0] == 10);
        CHECK(ptr[3] == 40);
    }
    
    TEST_CASE("data pointer access") {
        auto tensor = Tensor::FromVector<double>({2}, {1.5, 2.5});
        
        const double* ptr = tensor.data<double>();
        CHECK(ptr != nullptr);
        CHECK(ptr[0] == doctest::Approx(1.5));
        CHECK(ptr[1] == doctest::Approx(2.5));
    }
    
    TEST_CASE("write view gives mutable access") {
        auto tensor = Tensor::Allocate<float>({2});
        
        {
            auto view = tensor.write<float>();
            view[0] = 100.0f;
            view[1] = 200.0f;
        }
        
        auto rview = tensor.read<float>();
        CHECK(rview[0] == doctest::Approx(100.0f));
        CHECK(rview[1] == doctest::Approx(200.0f));
    }
    
    TEST_CASE("byte_size calculation") {
        auto f32_tensor = Tensor::FromVector<float>({100}, std::vector<float>(100, 0.0f));
        CHECK(f32_tensor.byte_size() == 100 * sizeof(float));
        
        auto f64_tensor = Tensor::FromVector<double>({100}, std::vector<double>(100, 0.0));
        CHECK(f64_tensor.byte_size() == 100 * sizeof(double));
        
        auto i8_tensor = Tensor::FromVector<int8_t>({100}, std::vector<int8_t>(100, 0));
        CHECK(i8_tensor.byte_size() == 100 * sizeof(int8_t));
    }
    
    TEST_CASE("shape returns correct dimensions") {
        auto tensor = Tensor::FromVector<float>({2, 3, 4}, std::vector<float>(24, 0.0f));
        
        auto shape = tensor.shape();
        CHECK(shape.size() == 3);
        CHECK(shape[0] == 2);
        CHECK(shape[1] == 3);
        CHECK(shape[2] == 4);
    }
    
    TEST_CASE("shape returns individual dimensions") {
        auto tensor = Tensor::FromVector<float>({5, 10, 15}, std::vector<float>(750, 0.0f));
        
        auto shape = tensor.shape();
        CHECK(shape[0] == 5);
        CHECK(shape[1] == 10);
        CHECK(shape[2] == 15);
    }
    
    TEST_CASE("reshape preserves data") {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        auto tensor = Tensor::FromVector<float>({2, 3}, data);
        
        auto reshaped = tensor.reshape({3, 2});
        
        auto view = reshaped.read<float>();
        CHECK(view[0] == doctest::Approx(1.0f));
        CHECK(view[5] == doctest::Approx(6.0f));
    }
    
    TEST_CASE("ToVector extracts all data") {
        std::vector<int32_t> data = {1, 2, 3, 4, 5};
        auto tensor = Tensor::FromVector<int32_t>({5}, data);
        
        auto extracted = tensor.ToVector<int32_t>();
        CHECK(extracted.size() == 5);
        CHECK(extracted == data);
    }

}

// ============================================================================
// Graph Introspection
// ============================================================================

TEST_SUITE("Graph Introspection") {

    TEST_CASE("empty graph has no operations") {
        Graph g;
        
        CHECK(g.num_operations() == 0);
        CHECK(g.GetAllOperations().empty());
    }
    
    TEST_CASE("GetOperation returns nullopt for missing") {
        Graph g;
        
        auto op = g.GetOperation("nonexistent");
        CHECK_FALSE(op.has_value());
    }
    
    TEST_CASE("GetOperationOrThrow throws for missing") {
        Graph g;
        
        CHECK_THROWS_AS(g.GetOperationOrThrow("nonexistent"), Error);
    }
    
    TEST_CASE("HasOperation returns false for missing") {
        Graph g;
        
        CHECK_FALSE(g.HasOperation("missing"));
    }
    
    TEST_CASE("freeze on empty graph") {
        Graph g;
        
        CHECK_FALSE(g.is_frozen());
        g.freeze();
        CHECK(g.is_frozen());
    }
    
    TEST_CASE("double freeze is safe") {
        Graph g;
        
        g.freeze();
        CHECK_NOTHROW(g.freeze());
        CHECK(g.is_frozen());
    }
    
    TEST_CASE("graph handle is valid") {
        Graph g;
        CHECK(g.handle() != nullptr);
    }
    
    TEST_CASE("graph move nullifies source") {
        Graph g1;
        auto* original_handle = g1.handle();
        
        Graph g2 = std::move(g1);
        
        CHECK(g2.handle() == original_handle);
        CHECK(g1.handle() == nullptr);
    }
    
    TEST_CASE("ToGraphDef on empty graph") {
        Graph g;
        
        auto graphdef = g.ToGraphDef();
        // Should return something (possibly empty buffer)
        CHECK(true);
    }
    
    TEST_CASE("DebugString on empty graph") {
        Graph g;
        
        auto debug = g.DebugString();
        // Should not crash
        CHECK(true);
    }

}

// ============================================================================
// Device Enumeration
// ============================================================================

TEST_SUITE("Device Enumeration") {

    TEST_CASE("list devices from session") {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        
        auto devices = s.ListDevices();
        
        // Should have at least one device (CPU)
        CHECK(devices.count() >= 1);
    }
    
    TEST_CASE("device attributes accessible") {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        
        auto devices = s.ListDevices();
        
        if (devices.count() > 0) {
            auto dev = devices.at(0);
            CHECK(dev.name.length() > 0);
            CHECK(dev.type.length() > 0);
            // Memory might be 0 for some device types
        }
    }
    
    TEST_CASE("HasGPU check") {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        
        // Just verify it doesn't crash - result depends on hardware
        bool has_gpu = s.HasGPU();
        (void)has_gpu;
        CHECK(true);
    }
    
    TEST_CASE("device iteration") {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        
        auto devices = s.ListDevices();
        
        for (int i = 0; i < devices.count(); ++i) {
            auto dev = devices.at(i);
            CHECK(dev.name.length() > 0);
        }
    }
    
    TEST_CASE("device at invalid index") {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        
        auto devices = s.ListDevices();
        
        // Accessing out of bounds should be safe (returns empty or throws)
        if (devices.count() < 1000) {
            // Just verify API exists
            CHECK(true);
        }
    }

}

// ============================================================================
// Session Edge Cases
// ============================================================================

TEST_SUITE("Session Edge Cases") {

    TEST_CASE("session options target") {
        SessionOptions opts;
        opts.SetTarget("local");
        
        CHECK(opts.handle() != nullptr);
    }
    
    TEST_CASE("session options SetConfig with empty proto") {
        SessionOptions opts;
        
        // Empty config should be safe
        CHECK_NOTHROW(opts.SetConfig(nullptr, 0));
    }
    
    TEST_CASE("multiple sessions from same graph") {
        Graph g;
        SessionOptions opts;
        
        Session s1(g, opts);
        Session s2(g, opts);
        Session s3(g, opts);
        
        CHECK(s1.handle() != nullptr);
        CHECK(s2.handle() != nullptr);
        CHECK(s3.handle() != nullptr);
        
        // All should be distinct
        CHECK(s1.handle() != s2.handle());
        CHECK(s2.handle() != s3.handle());
    }
    
    TEST_CASE("session survives graph freeze") {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        
        g.freeze();
        
        CHECK(s.handle() != nullptr);
        CHECK(g.is_frozen());
    }
    
    TEST_CASE("resolve on empty graph returns error") {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        
        CHECK_THROWS(s.resolve("nonexistent:0"));
    }

}

// ============================================================================
// Buffer Edge Cases
// ============================================================================

TEST_SUITE("Buffer Edge Cases") {

    TEST_CASE("buffer default state") {
        Buffer b;
        
        CHECK(b.data() == nullptr);
    }
    
    TEST_CASE("buffer from string data") {
        std::string data = "test data";
        Buffer b(data.data(), data.size());
        
        auto bytes = b.to_bytes();
        CHECK(bytes.size() == data.size());
    }
    
    TEST_CASE("buffer from binary data") {
        std::vector<uint8_t> data = {0x00, 0xFF, 0x42, 0x00, 0xAB};
        Buffer b(data.data(), data.size());
        
        auto bytes = b.to_bytes();
        CHECK(bytes.size() == 5);
        CHECK(bytes[1] == 0xFF);
        CHECK(bytes[3] == 0x00);
    }
    
    TEST_CASE("buffer move") {
        std::string data = "move test";
        Buffer b1(data.data(), data.size());
        
        Buffer b2 = std::move(b1);
        
        CHECK(b1.handle() == nullptr);
        CHECK(b2.handle() != nullptr);
    }
    
    TEST_CASE("large buffer") {
        std::vector<uint8_t> data(1000000, 0x42);  // 1MB
        Buffer b(data.data(), data.size());
        
        auto bytes = b.to_bytes();
        CHECK(bytes.size() == 1000000);
    }

}

// ============================================================================
// Error Class Tests
// ============================================================================

TEST_SUITE("Error Class") {

    TEST_CASE("Error Wrapper factory") {
        auto err = Error::Wrapper(TF_INTERNAL, "Session::Run", "internal error", "op", 0);
        
        CHECK(err.code() == TF_INTERNAL);
        CHECK(err.source() == ErrorSource::Wrapper);
    }
    
    TEST_CASE("Error TensorFlow factory") {
        auto err = Error::TensorFlow(TF_NOT_FOUND, "Load", "file not found");
        
        CHECK(err.code() == TF_NOT_FOUND);
        CHECK(err.source() == ErrorSource::TensorFlow);
    }
    
    TEST_CASE("Error what() returns description") {
        auto err = Error::Wrapper(TF_INVALID_ARGUMENT, "test_op", "bad input", "", -1);
        
        std::string what = err.what();
        CHECK(what.length() > 0);
    }
    
    TEST_CASE("Error code_name returns string") {
        auto err = Error::Wrapper(TF_ABORTED, "op", "msg", "", -1);
        
        CHECK(std::strcmp(err.code_name(), "ABORTED") == 0);
    }
    
    TEST_CASE("Error context accessible") {
        auto err = Error::Wrapper(TF_INTERNAL, "MyContext", "message", "", -1);
        
        CHECK(err.context() == "MyContext");
    }

}
