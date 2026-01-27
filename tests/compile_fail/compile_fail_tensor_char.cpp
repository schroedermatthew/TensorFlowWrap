// compile_fail_tensor_char.cpp
// This file should FAIL to compile.
// Tests that char is rejected by the TensorScalar concept.
// (char is distinct from int8_t/uint8_t in C++)
//
// Expected error: constraint not satisfied / no matching function

#include "tf_wrap/tensor.hpp"

int main() {
    // char is not a TensorScalar type - this should fail to compile
    auto tensor = tf_wrap::Tensor::FromScalar('x');
    (void)tensor;
    return 0;
}
