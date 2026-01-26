// tests/test_integration.cpp
// Integration test requiring a real TensorFlow C library.
// This target is only built when TF_WRAPPER_TF_STUB=OFF.

#include "tf_wrap/core.hpp"

#include <cstdlib>
#include <vector>

int main()
{
    try {
        tf_wrap::Graph g;

        // Const A
        {
            auto a_tensor = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
            (void)g.NewOperation("Const", "A")
                .SetAttrTensor("value", a_tensor.handle())
                .SetAttrType("dtype", TF_FLOAT)
                .Finish();
        }

        // Const B
        {
            auto b_tensor = tf_wrap::Tensor::FromVector<float>({2}, {10.0f, 20.0f});
            (void)g.NewOperation("Const", "B")
                .SetAttrTensor("value", b_tensor.handle())
                .SetAttrType("dtype", TF_FLOAT)
                .Finish();
        }

        // Add A + B
        {
            TF_Operation* op_a = g.GetOperationOrThrow("A");
            TF_Operation* op_b = g.GetOperationOrThrow("B");

            (void)g.NewOperation("AddV2", "AddAB")
                .AddInput(tf_wrap::Output(op_a, 0))
                .AddInput(tf_wrap::Output(op_b, 0))
                .SetAttrType("T", TF_FLOAT)
                .Finish();
        }

        tf_wrap::Session s(g);

        // Resolve once, then run using handles.
        std::vector<tf_wrap::Fetch> fetches{tf_wrap::Fetch{s.resolve("AddAB:0")}};
        auto results = s.Run({}, fetches, {});

        if (results.size() != 1u) {
            return 1;
        }

        auto v = results[0].ToVector<float>();
        if (v.size() != 2u) {
            return 1;
        }

        // Verify: [1, 2] + [10, 20] = [11, 22]
        return (v[0] == 11.0f && v[1] == 22.0f) ? 0 : 1;

    } catch (const std::exception&) {
        return 1;
    }
}
