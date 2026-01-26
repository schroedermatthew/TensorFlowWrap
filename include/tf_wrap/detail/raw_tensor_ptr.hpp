// tf_wrap/detail/raw_tensor_ptr.hpp
// Internal helpers for owning raw TF_Tensor* with unique_ptr.

#pragma once

#include <memory>

extern "C" {
#include <tensorflow/c/c_api.h>
}

namespace tf_wrap::detail {

struct TensorDeleter {
    void operator()(TF_Tensor* t) const noexcept {
        if (t) {
            TF_DeleteTensor(t);
        }
    }
};

using RawTensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

} // namespace tf_wrap::detail
