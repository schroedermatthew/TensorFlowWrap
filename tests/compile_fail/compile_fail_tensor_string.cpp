// compile_fail_tensor_string.cpp
// This file should FAIL to compile.
// Tests that std::string is rejected by the TensorScalar concept.
//
// Expected error: constraint not satisfied / no matching function

#include "tf_wrap/tensor.hpp"
#include <string>

int main() {
    // std::string is not a TensorScalar type - this should fail to compile
    auto tensor = tf_wrap::Tensor::FromScalar(std::string("hello"));
    (void)tensor;
    return 0;
}
