// fuzz_session.cpp
// Fuzz test for tf_wrap::Session and related types
//
// Build with libFuzzer:
//   clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address,undefined \
//       -I include -I third_party/tf_stub \
//       -DTF_WRAPPER_TF_STUB_ENABLED=1 \
//       third_party/tf_stub/tf_c_stub.cpp \
//       tests/fuzz/fuzz_session.cpp \
//       -o fuzz_session
//
// Run:
//   ./fuzz_session corpus/ -max_len=1024 -runs=100000
//
// This fuzzer exercises:
// - Graph construction and operations
// - Session creation with various options
// - Buffer operations
// - SessionOptions configuration
// - Move semantics

#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>

using namespace tf_wrap;

// Operation codes
enum class Op : uint8_t {
    CreateGraph = 0,
    CreateSession = 1,
    CreateSessionWithOpts = 2,
    MoveGraph = 3,
    MoveSession = 4,
    FreezeGraph = 5,
    GetOperation = 6,
    ListDevices = 7,
    Resolve = 8,
    CreateBuffer = 9,
    BufferFromData = 10,
    MoveBuffer = 11,
    CreateSessionOptions = 12,
    SetTarget = 13,
    NUM_OPS
};

// Fuzz target
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 1) return 0;
    
    // State
    Graph graph;
    std::unique_ptr<Session> session;
    Buffer buffer;
    SessionOptions opts;
    
    size_t pos = 0;
    
    while (pos < size) {
        Op op = static_cast<Op>(data[pos++] % static_cast<uint8_t>(Op::NUM_OPS));
        
        try {
            switch (op) {
                case Op::CreateGraph: {
                    graph = Graph();
                    break;
                }
                
                case Op::CreateSession: {
                    if (graph.handle()) {
                        session = std::make_unique<Session>(graph, SessionOptions());
                    }
                    break;
                }
                
                case Op::CreateSessionWithOpts: {
                    if (graph.handle()) {
                        session = std::make_unique<Session>(graph, opts);
                    }
                    break;
                }
                
                case Op::MoveGraph: {
                    Graph other = std::move(graph);
                    // Move back
                    graph = std::move(other);
                    break;
                }
                
                case Op::MoveSession: {
                    if (session) {
                        Session other = std::move(*session);
                        *session = std::move(other);
                    }
                    break;
                }
                
                case Op::FreezeGraph: {
                    if (graph.handle()) {
                        graph.freeze();
                        (void)graph.is_frozen();
                    }
                    break;
                }
                
                case Op::GetOperation: {
                    if (pos + 1 > size) break;
                    uint8_t len = data[pos++];
                    len = len % 32;  // Cap name length
                    
                    if (pos + len > size) break;
                    std::string name(reinterpret_cast<const char*>(data + pos), len);
                    pos += len;
                    
                    if (graph.handle()) {
                        // GetOperation returns optional, won't throw for not found
                        auto op_result = graph.GetOperation(name);
                        (void)op_result;
                    }
                    break;
                }
                
                case Op::ListDevices: {
                    if (session && session->handle()) {
                        auto devices = session->ListDevices();
                        (void)devices.count();
                    }
                    break;
                }
                
                case Op::Resolve: {
                    if (pos + 1 > size) break;
                    uint8_t len = data[pos++];
                    len = len % 32;
                    
                    if (pos + len > size) break;
                    std::string name(reinterpret_cast<const char*>(data + pos), len);
                    pos += len;
                    
                    if (session && session->handle()) {
                        // resolve throws on not found, expected
                        try {
                            auto output = session->resolve(name);
                            (void)output;
                        } catch (const Error&) {
                            // Expected
                        }
                    }
                    break;
                }
                
                case Op::CreateBuffer: {
                    buffer = Buffer();
                    break;
                }
                
                case Op::BufferFromData: {
                    if (pos + 1 > size) break;
                    uint8_t len = data[pos++];
                    len = len % 64;
                    
                    if (pos + len > size) break;
                    
                    buffer = Buffer(data + pos, len);
                    pos += len;
                    
                    // Verify round-trip
                    auto bytes = buffer.to_bytes();
                    if (bytes.size() != len) {
                        __builtin_trap();
                    }
                    break;
                }
                
                case Op::MoveBuffer: {
                    Buffer other = std::move(buffer);
                    buffer = std::move(other);
                    break;
                }
                
                case Op::CreateSessionOptions: {
                    opts = SessionOptions();
                    break;
                }
                
                case Op::SetTarget: {
                    if (pos + 1 > size) break;
                    uint8_t len = data[pos++];
                    len = len % 32;
                    
                    if (pos + len > size) break;
                    std::string target(reinterpret_cast<const char*>(data + pos), len);
                    pos += len;
                    
                    opts.SetTarget(target.c_str());
                    break;
                }
                
                default:
                    break;
            }
            
        } catch (const std::exception&) {
            // Expected for some operations
        }
    }
    
    return 0;
}
