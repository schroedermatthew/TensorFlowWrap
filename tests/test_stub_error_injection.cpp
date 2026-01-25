// tests/test_stub_error_injection.cpp
// Stub-only tests for controlled error injection

#include "doctest.h"

#include "tf_wrap/all.hpp"

#if defined(TF_WRAPPER_USE_STUB)

#include "tf_wrap/stub_control.hpp"

TEST_CASE("stub: injected TF_SessionRun error surfaces as tf_wrap::Error") {
    tf_wrap::Graph g;
    (void)g.NewOperation("Placeholder", "input")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    auto* in = g.GetOperationOrThrow("input");
    (void)g.NewOperation("Identity", "output")
        .AddInput({in, 0})
        .Finish();

    tf_wrap::Session s(g);

    tf_wrap::Tensor x = tf_wrap::Tensor::FromScalar<float>(1.0f);

    tf_wrap::StubSetNextError("TF_SessionRun", TF_INTERNAL, "injected failure");

    try {
        (void)s.Run({tf_wrap::Feed{"input", x.handle()}}, {tf_wrap::Fetch{"output"}}, {});
        FAIL_CHECK("expected tf_wrap::Error");
    } catch (const tf_wrap::Error& e) {
        CHECK(e.code() == TF_INTERNAL);
        CHECK(std::string(e.what()).find("injected failure") != std::string::npos);
    }
}

#endif
