// fuzz_tensor.cpp
// Fuzz test for tf_wrap::Tensor
//
// Build with libFuzzer:
//   clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address,undefined \
//       -I include -I third_party/tf_stub \
//       -DTF_WRAPPER_TF_STUB_ENABLED=1 \
//       third_party/tf_stub/tf_c_stub.cpp \
//       tests/fuzz/fuzz_tensor.cpp \
//       -o fuzz_tensor
//
// Run:
//   ./fuzz_tensor corpus/ -max_len=1024 -runs=100000
//
// This fuzzer exercises:
// - Tensor::FromVector with arbitrary shapes and data
// - Tensor::FromScalar with various values
// - Tensor::FromString with arbitrary strings
// - Tensor read/write operations
// - Tensor reshape operations

#include "tf_wrap/tensor.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <string>

using namespace tf_wrap;

// FuzzedDataProvider-like helper to consume fuzz input
class FuzzInput {
public:
    FuzzInput(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}
    
    bool empty() const { return pos_ >= size_; }
    size_t remaining() const { return size_ - pos_; }
    
    uint8_t ConsumeByte() {
        if (pos_ >= size_) return 0;
        return data_[pos_++];
    }
    
    template<typename T>
    T Consume() {
        T value{};
        size_t bytes = std::min(sizeof(T), remaining());
        if (bytes > 0) {
            std::memcpy(&value, data_ + pos_, bytes);
            pos_ += bytes;
        }
        return value;
    }
    
    std::vector<uint8_t> ConsumeBytes(size_t count) {
        count = std::min(count, remaining());
        std::vector<uint8_t> result(data_ + pos_, data_ + pos_ + count);
        pos_ += count;
        return result;
    }
    
    std::string ConsumeString(size_t max_length) {
        size_t length = std::min(max_length, remaining());
        std::string result(reinterpret_cast<const char*>(data_ + pos_), length);
        pos_ += length;
        return result;
    }
    
    template<typename T>
    std::vector<T> ConsumeVector(size_t max_count) {
        size_t count = std::min(max_count, remaining() / sizeof(T));
        std::vector<T> result(count);
        for (size_t i = 0; i < count; ++i) {
            result[i] = Consume<T>();
        }
        return result;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

// Fuzz Tensor::FromScalar
void FuzzFromScalar(FuzzInput& input) {
    uint8_t dtype_choice = input.ConsumeByte() % 10;
    
    try {
        switch (dtype_choice) {
            case 0: {
                float v = input.Consume<float>();
                auto t = Tensor::FromScalar<float>(v);
                (void)t.ToScalar<float>();
                break;
            }
            case 1: {
                double v = input.Consume<double>();
                auto t = Tensor::FromScalar<double>(v);
                (void)t.ToScalar<double>();
                break;
            }
            case 2: {
                int32_t v = input.Consume<int32_t>();
                auto t = Tensor::FromScalar<int32_t>(v);
                (void)t.ToScalar<int32_t>();
                break;
            }
            case 3: {
                int64_t v = input.Consume<int64_t>();
                auto t = Tensor::FromScalar<int64_t>(v);
                (void)t.ToScalar<int64_t>();
                break;
            }
            case 4: {
                uint8_t v = input.Consume<uint8_t>();
                auto t = Tensor::FromScalar<uint8_t>(v);
                (void)t.ToScalar<uint8_t>();
                break;
            }
            case 5: {
                uint16_t v = input.Consume<uint16_t>();
                auto t = Tensor::FromScalar<uint16_t>(v);
                (void)t.ToScalar<uint16_t>();
                break;
            }
            case 6: {
                uint32_t v = input.Consume<uint32_t>();
                auto t = Tensor::FromScalar<uint32_t>(v);
                (void)t.ToScalar<uint32_t>();
                break;
            }
            case 7: {
                uint64_t v = input.Consume<uint64_t>();
                auto t = Tensor::FromScalar<uint64_t>(v);
                (void)t.ToScalar<uint64_t>();
                break;
            }
            case 8: {
                int8_t v = input.Consume<int8_t>();
                auto t = Tensor::FromScalar<int8_t>(v);
                (void)t.ToScalar<int8_t>();
                break;
            }
            case 9: {
                int16_t v = input.Consume<int16_t>();
                auto t = Tensor::FromScalar<int16_t>(v);
                (void)t.ToScalar<int16_t>();
                break;
            }
        }
    } catch (const std::exception&) {
        // Expected for invalid inputs
    }
}

// Fuzz Tensor::FromVector with float data
void FuzzFromVectorFloat(FuzzInput& input) {
    // Get dimensions (1-4 dims, each 1-16)
    uint8_t num_dims = (input.ConsumeByte() % 4) + 1;
    std::vector<int64_t> dims;
    int64_t total = 1;
    
    for (uint8_t i = 0; i < num_dims && total < 1024; ++i) {
        int64_t dim = (input.ConsumeByte() % 16) + 1;
        if (total * dim > 1024) dim = 1;  // Cap total size
        dims.push_back(dim);
        total *= dim;
    }
    
    // Get data
    std::vector<float> data = input.ConsumeVector<float>(total);
    
    // Pad or truncate to match shape
    data.resize(total, 0.0f);
    
    try {
        auto tensor = Tensor::FromVector<float>(dims, data);
        
        // Try various operations
        (void)tensor.valid();
        (void)tensor.dtype();
        (void)tensor.rank();
        (void)tensor.shape();
        (void)tensor.num_elements();
        (void)tensor.byte_size();
        
        // Read back
        auto view = tensor.read<float>();
        for (size_t i = 0; i < std::min(view.size(), size_t(10)); ++i) {
            (void)view[i];
        }
        
        // Try reshape if we have enough data
        if (total > 1 && dims.size() > 1) {
            std::vector<int64_t> new_dims = {total};
            auto reshaped = tensor.reshape(new_dims);
            (void)reshaped.rank();
        }
        
        // Clone
        auto cloned = tensor.Clone();
        (void)cloned.valid();
        
    } catch (const std::exception&) {
        // Expected for invalid inputs
    }
}

// Fuzz Tensor::FromString
void FuzzFromString(FuzzInput& input) {
    std::string str = input.ConsumeString(256);
    
    try {
        auto tensor = Tensor::FromString(str);
        (void)tensor.valid();
        (void)tensor.dtype();
        
        std::string back = tensor.ToString();
        // Verify round-trip (should match)
        if (back != str) {
            // This would be a bug
            __builtin_trap();
        }
    } catch (const std::exception&) {
        // Expected for some inputs
    }
}

// Fuzz move semantics
void FuzzMoveSemantics(FuzzInput& input) {
    try {
        float v = input.Consume<float>();
        auto t1 = Tensor::FromScalar<float>(v);
        
        // Move construct
        auto t2 = std::move(t1);
        if (t1.valid()) {
            __builtin_trap();  // Bug: moved-from should be invalid
        }
        if (!t2.valid()) {
            __builtin_trap();  // Bug: move target should be valid
        }
        
        // Move assign
        auto t3 = Tensor::FromScalar<float>(0.0f);
        t3 = std::move(t2);
        if (t2.valid()) {
            __builtin_trap();  // Bug: moved-from should be invalid
        }
        if (!t3.valid()) {
            __builtin_trap();  // Bug: move target should be valid
        }
        
    } catch (const std::exception&) {
        // Expected for some inputs
    }
}

// Main fuzz target
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) return 0;
    
    FuzzInput input(data, size);
    
    // Choose which operation to fuzz
    uint8_t op = input.ConsumeByte() % 4;
    
    switch (op) {
        case 0:
            FuzzFromScalar(input);
            break;
        case 1:
            FuzzFromVectorFloat(input);
            break;
        case 2:
            FuzzFromString(input);
            break;
        case 3:
            FuzzMoveSemantics(input);
            break;
    }
    
    return 0;
}
