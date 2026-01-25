// tests/test_easy_helpers.cpp
// Tests for ergonomic helper functions (TensorName parsing, validation, etc.)
//
// Purpose: Verify helper layer catches errors early and works correctly

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/all.hpp"

#include <cctype>
#include <string>
#include <random>
#include <algorithm>

// ============================================================================
// TensorName Parsing (inline implementation for testing)
// ============================================================================

namespace {

struct TensorName {
    std::string op;
    int index{0};
    
    static TensorName parse(std::string_view s) {
        if (auto pos = s.rfind(':'); pos != std::string_view::npos && pos < s.size() - 1) {
            auto suffix = s.substr(pos + 1);
            // CRITICAL: Cast to unsigned char to avoid UB (ChatGPT fix)
            bool all_digits = std::all_of(suffix.begin(), suffix.end(), 
                [](unsigned char c) { return std::isdigit(c); });
            if (all_digits) {
                return {std::string(s.substr(0, pos)), std::stoi(std::string(suffix))};
            }
        }
        return {std::string(s), 0};
    }
    
    static bool looks_like_tensor_name(std::string_view s) {
        auto pos = s.rfind(':');
        if (pos == std::string_view::npos || pos >= s.size() - 1) return false;
        auto suffix = s.substr(pos + 1);
        return std::all_of(suffix.begin(), suffix.end(),
            [](unsigned char c) { return std::isdigit(c); });
    }
};

// Dtype validation helper
void validate_binary_op_dtypes(TF_DataType a, TF_DataType b, const char* op_name) {
    if (a != b) {
        throw std::runtime_error(
            std::string("Dtype mismatch in ") + op_name + 
            ": inputs have different types"
        );
    }
}

// Auto-naming generator
class NameGenerator {
public:
    std::string generate(std::string_view prefix) {
        return std::string(prefix) + "_" + std::to_string(counter_++);
    }
    void reset() { counter_ = 0; }
private:
    int counter_ = 0;
};

} // anonymous namespace

// ============================================================================
// TensorName::parse Tests
// ============================================================================

TEST_CASE("TensorName::parse basic") {
    auto n1 = TensorName::parse("foo:0");
    CHECK(n1.op == "foo");
    CHECK(n1.index == 0);
    
    auto n2 = TensorName::parse("bar:42");
    CHECK(n2.op == "bar");
    CHECK(n2.index == 42);
    
    auto n3 = TensorName::parse("baz");
    CHECK(n3.op == "baz");
    CHECK(n3.index == 0);
}

TEST_CASE("TensorName::parse edge cases") {
    SUBCASE("Colon not followed by digits") {
        auto n = TensorName::parse("foo:bar");
        CHECK(n.op == "foo:bar");
        CHECK(n.index == 0);
    }
    
    SUBCASE("Multiple colons") {
        auto n = TensorName::parse("a:b:3");
        CHECK(n.op == "a:b");
        CHECK(n.index == 3);
    }
    
    SUBCASE("Empty string") {
        auto n = TensorName::parse("");
        CHECK(n.op == "");
        CHECK(n.index == 0);
    }
    
    SUBCASE("Just a number") {
        auto n = TensorName::parse("123");
        CHECK(n.op == "123");
        CHECK(n.index == 0);
    }
    
    SUBCASE("Trailing colon") {
        auto n = TensorName::parse("foo:");
        CHECK(n.op == "foo:");
        CHECK(n.index == 0);
    }
    
    SUBCASE("Negative-looking index") {
        auto n = TensorName::parse("foo:-1");
        CHECK(n.op == "foo:-1");  // -1 contains non-digit '-'
        CHECK(n.index == 0);
    }
    
    SUBCASE("Large index") {
        auto n = TensorName::parse("op:99999");
        CHECK(n.op == "op");
        CHECK(n.index == 99999);
    }
}

TEST_CASE("TensorName::parse with high-bit chars (UB prevention)") {
    // Characters > 127 would cause UB with plain ::isdigit on signed char
    std::string name = "op\x80\x81:5";
    auto n = TensorName::parse(name);
    CHECK(n.index == 5);
}

TEST_CASE("TensorName::looks_like_tensor_name") {
    CHECK(TensorName::looks_like_tensor_name("foo:0") == true);
    CHECK(TensorName::looks_like_tensor_name("foo:123") == true);
    CHECK(TensorName::looks_like_tensor_name("foo") == false);
    CHECK(TensorName::looks_like_tensor_name("foo:") == false);
    CHECK(TensorName::looks_like_tensor_name("foo:bar") == false);
    CHECK(TensorName::looks_like_tensor_name("") == false);
}

// ============================================================================
// Dtype Validation Tests
// ============================================================================

TEST_CASE("validate_binary_op_dtypes accepts matching types") {
    CHECK_NOTHROW(validate_binary_op_dtypes(TF_FLOAT, TF_FLOAT, "Add"));
    CHECK_NOTHROW(validate_binary_op_dtypes(TF_INT32, TF_INT32, "Mul"));
    CHECK_NOTHROW(validate_binary_op_dtypes(TF_DOUBLE, TF_DOUBLE, "Sub"));
    CHECK_NOTHROW(validate_binary_op_dtypes(TF_BOOL, TF_BOOL, "LogicalAnd"));
}

TEST_CASE("validate_binary_op_dtypes rejects mismatched types") {
    CHECK_THROWS(validate_binary_op_dtypes(TF_FLOAT, TF_INT32, "Add"));
    CHECK_THROWS(validate_binary_op_dtypes(TF_DOUBLE, TF_FLOAT, "Mul"));
    CHECK_THROWS(validate_binary_op_dtypes(TF_INT64, TF_INT32, "Sub"));
    CHECK_THROWS(validate_binary_op_dtypes(TF_BOOL, TF_INT8, "LogicalAnd"));
}

// ============================================================================
// Auto-Naming Tests
// ============================================================================

TEST_CASE("NameGenerator produces unique names") {
    NameGenerator gen;
    
    CHECK(gen.generate("Add") == "Add_0");
    CHECK(gen.generate("Add") == "Add_1");
    CHECK(gen.generate("Mul") == "Mul_2");
    CHECK(gen.generate("Add") == "Add_3");
}

TEST_CASE("NameGenerator reset") {
    NameGenerator gen;
    gen.generate("Test");
    gen.generate("Test");
    gen.reset();
    CHECK(gen.generate("Test") == "Test_0");
}

// ============================================================================
// Graph Helper Integration Tests
// ============================================================================

TEST_CASE("Binary op with matching dtypes succeeds") {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto b = tf_wrap::Tensor::FromScalar<float>(2.0f);
    
    auto* op_a = g.NewOperation("Const", "a")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_b = g.NewOperation("Const", "b")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    TF_DataType dt_a = TF_OperationOutputType({op_a, 0});
    TF_DataType dt_b = TF_OperationOutputType({op_b, 0});
    
    CHECK_NOTHROW(validate_binary_op_dtypes(dt_a, dt_b, "AddV2"));
    
    auto* add_op = g.NewOperation("AddV2", "sum")
        .AddInput({op_a, 0})
        .AddInput({op_b, 0})
        .SetAttrType("T", TF_FLOAT)
        .Finish();
    
    CHECK(TF_OperationOutputType({add_op, 0}) == TF_FLOAT);
}

TEST_CASE("Binary op with mismatched dtypes should be caught") {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto b = tf_wrap::Tensor::FromScalar<int32_t>(2);
    
    auto* op_a = g.NewOperation("Const", "a")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_b = g.NewOperation("Const", "b")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    TF_DataType dt_a = TF_OperationOutputType({op_a, 0});
    TF_DataType dt_b = TF_OperationOutputType({op_b, 0});
    
    // Validation should catch the mismatch BEFORE creating the op
    CHECK_THROWS(validate_binary_op_dtypes(dt_a, dt_b, "AddV2"));
}

// ============================================================================
// Identity Helper Validation
// ============================================================================

TEST_CASE("Identity operation tracks dtype from T attribute") {
    tf_wrap::Graph g;
    
    auto input = tf_wrap::Tensor::FromScalar<int32_t>(42);
    auto* const_op = g.NewOperation("Const", "input")
        .SetAttrTensor("value", input.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* id_op = g.NewOperation("Identity", "output")
        .AddInput({const_op, 0})
        .SetAttrType("T", TF_INT32)
        .Finish();
    
    CHECK(TF_OperationOutputType({id_op, 0}) == TF_INT32);
}

// ============================================================================
// Scalar/Const Helper Behavior
// ============================================================================

TEST_CASE("Scalar const preserves dtype") {
    tf_wrap::Graph g;
    
    SUBCASE("Float") {
        auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
        auto* op = g.NewOperation("Const", "float_scalar")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();
        CHECK(TF_OperationOutputType({op, 0}) == TF_FLOAT);
    }
    
    SUBCASE("Int32") {
        auto t = tf_wrap::Tensor::FromScalar<int32_t>(1);
        auto* op = g.NewOperation("Const", "int_scalar")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_INT32)
            .Finish();
        CHECK(TF_OperationOutputType({op, 0}) == TF_INT32);
    }
    
    SUBCASE("Double") {
        auto t = tf_wrap::Tensor::FromScalar<double>(1.0);
        auto* op = g.NewOperation("Const", "double_scalar")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_DOUBLE)
            .Finish();
        CHECK(TF_OperationOutputType({op, 0}) == TF_DOUBLE);
    }
    
    SUBCASE("Bool") {
        auto t = tf_wrap::Tensor::FromScalar<bool>(true);
        auto* op = g.NewOperation("Const", "bool_scalar")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_BOOL)
            .Finish();
        CHECK(TF_OperationOutputType({op, 0}) == TF_BOOL);
    }
}



TEST_CASE("TensorName::parse fuzz does not crash")
{
    // This is a robustness test: we don't require acceptance, only that parsing is stable.
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> len_dist(0, 64);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    for (int iter = 0; iter < 5000; ++iter)
    {
        const int len = len_dist(rng);
        std::string s;
        s.resize(static_cast<std::size_t>(len));
        for (int i = 0; i < len; ++i)
        {
            s[static_cast<std::size_t>(i)] = static_cast<char>(byte_dist(rng));
        }

        // Ensure no exceptions escape except the documented ones.
        try
        {
            (void)tf_wrap::facade::TensorName::parse(s);
        }
        catch (const std::exception&)
        {
            // parse() is allowed to reject invalid inputs.
        }
    }
}
