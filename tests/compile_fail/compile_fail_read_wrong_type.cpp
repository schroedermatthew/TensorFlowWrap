// compile_fail_read_wrong_type.cpp
// This file should FAIL to compile.
// Tests that reading a tensor with a non-TensorScalar type is rejected.
//
// Expected error: constraint not satisfied / no matching function

#include "tf_wrap/tensor.hpp"
#include <vector>

int main() {
    // Create a float tensor
    auto tensor = tf_wrap::Tensor::FromScalar(1.0f);
    
    // Try to read it as std::vector<std::string> - this should fail
    auto data = tensor.read<std::string>();
    (void)data;
    return 0;
}
