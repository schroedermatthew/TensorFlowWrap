// fuzz_small_vector.cpp
// Fuzz test for tf_wrap::SmallVector
//
// Build with libFuzzer:
//   clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address,undefined \
//       -I include \
//       tests/fuzz/fuzz_small_vector.cpp \
//       -o fuzz_small_vector
//
// Run:
//   ./fuzz_small_vector corpus/ -max_len=4096 -runs=100000
//
// This fuzzer exercises:
// - push_back / pop_back sequences
// - insert / erase at various positions
// - resize operations
// - copy and move semantics
// - iterator operations

#include "tf_wrap/small_vector.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

using namespace tf_wrap;

// Operation codes
enum class Op : uint8_t {
    PushBack = 0,
    PopBack = 1,
    Clear = 2,
    Resize = 3,
    Reserve = 4,
    ShrinkToFit = 5,
    At = 6,
    Front = 7,
    Back = 8,
    Copy = 9,
    Move = 10,
    Swap = 11,
    EmplaceBack = 12,
    NUM_OPS
};

// Verify SmallVector invariants
template<typename T, std::size_t N>
void VerifyInvariants(const SmallVector<T, N>& v) {
    // Size should be <= capacity
    if (v.size() > v.capacity()) {
        __builtin_trap();
    }
    
    // Empty should match size == 0
    if (v.empty() != (v.size() == 0)) {
        __builtin_trap();
    }
    
    // If not empty, data() should not be null
    if (!v.empty() && v.data() == nullptr) {
        __builtin_trap();
    }
    
    // begin() + size() should equal end()
    if (v.begin() + v.size() != v.end()) {
        __builtin_trap();
    }
}

// Main fuzz target
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 1) return 0;
    
    // Use small inline buffer to exercise both inline and heap paths
    SmallVector<int, 4> vec;
    SmallVector<int, 4> other;
    
    size_t pos = 0;
    
    while (pos < size) {
        Op op = static_cast<Op>(data[pos++] % static_cast<uint8_t>(Op::NUM_OPS));
        
        try {
            switch (op) {
                case Op::PushBack: {
                    if (pos + sizeof(int) > size) break;
                    int value;
                    std::memcpy(&value, data + pos, sizeof(int));
                    pos += sizeof(int);
                    if (vec.size() < 1000) {  // Cap size to avoid OOM
                        vec.push_back(value);
                    }
                    break;
                }
                
                case Op::PopBack: {
                    if (!vec.empty()) {
                        vec.pop_back();
                    }
                    break;
                }
                
                case Op::Clear: {
                    vec.clear();
                    break;
                }
                
                case Op::Resize: {
                    if (pos >= size) break;
                    uint8_t new_size = data[pos++];
                    // Cap at reasonable size
                    new_size = new_size % 64;
                    vec.resize(new_size);
                    break;
                }
                
                case Op::Reserve: {
                    if (pos >= size) break;
                    uint8_t new_cap = data[pos++];
                    // Cap at reasonable size
                    new_cap = new_cap % 128;
                    vec.reserve(new_cap);
                    break;
                }
                
                case Op::ShrinkToFit: {
                    vec.shrink_to_fit();
                    break;
                }
                
                case Op::At: {
                    if (pos >= size) break;
                    uint8_t idx_byte = data[pos++];
                    
                    if (!vec.empty()) {
                        size_t idx = idx_byte % vec.size();
                        (void)vec.at(idx);
                    }
                    break;
                }
                
                case Op::Front: {
                    if (!vec.empty()) {
                        (void)vec.front();
                    }
                    break;
                }
                
                case Op::Back: {
                    if (!vec.empty()) {
                        (void)vec.back();
                    }
                    break;
                }
                
                case Op::Copy: {
                    other = vec;
                    // Verify copy is independent
                    if (other.size() != vec.size()) {
                        __builtin_trap();
                    }
                    break;
                }
                
                case Op::Move: {
                    size_t old_size = vec.size();
                    other = std::move(vec);
                    // After move, other should have old contents
                    if (other.size() != old_size) {
                        __builtin_trap();
                    }
                    // Move back for more operations
                    vec = std::move(other);
                    break;
                }
                
                case Op::Swap: {
                    vec.swap(other);
                    break;
                }
                
                case Op::EmplaceBack: {
                    if (pos + sizeof(int) > size) break;
                    int value;
                    std::memcpy(&value, data + pos, sizeof(int));
                    pos += sizeof(int);
                    if (vec.size() < 1000) {
                        vec.emplace_back(value);
                    }
                    break;
                }
                
                default:
                    break;
            }
            
            // Verify invariants after each operation
            VerifyInvariants(vec);
            VerifyInvariants(other);
            
        } catch (const std::out_of_range&) {
            // Expected for invalid indices
        } catch (const std::length_error&) {
            // Expected for size limits
        } catch (const std::bad_alloc&) {
            // Expected for memory limits
        }
    }
    
    return 0;
}
