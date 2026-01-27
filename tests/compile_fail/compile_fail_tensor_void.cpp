// compile_fail_tensor_void.cpp
// This file should FAIL to compile.
// Tests that void* is rejected by the TensorScalar concept.
//
// Expected error: constraint not satisfied / no matching function

#include "tf_wrap/tensor.hpp"

int main() {
    // void* is not a TensorScalar type - this should fail to compile
    void* ptr = nullptr;
    auto tensor = tf_wrap::Tensor::FromScalar(ptr);
    (void)tensor;
    return 0;
}
