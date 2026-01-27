// tf_wrap/graph.hpp
// RAII wrapper for TF_Graph - Production Inference Edition (read-only)
//
// This is a read-only graph wrapper for inference. Graphs are loaded from
// SavedModel, not built programmatically. Build graphs in Python, export
// as SavedModel, load and run inference in C++.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include <tensorflow/c/c_api.h>
}

#include "tf_wrap/error.hpp"
#include "tf_wrap/format.hpp"
#include "tf_wrap/scope_guard.hpp"
#include "tf_wrap/status.hpp"

namespace tf_wrap {

// Forward declarations
class Graph;
struct OperationInfo;

namespace detail {

struct GraphState {
    TF_Graph* graph{nullptr};
    bool frozen{false};

    GraphState() : graph(TF_NewGraph()) {
        if (!graph) {
            throw Error::Wrapper(TF_INTERNAL, "GraphState",
                "TF_NewGraph failed", "", -1);
        }
    }

    ~GraphState() {
        if (graph) {
            TF_DeleteGraph(graph);
        }
    }

    GraphState(const GraphState&) = delete;
    GraphState& operator=(const GraphState&) = delete;
};

} // namespace detail

// ============================================================================
// OperationInfo - Information about an operation (for introspection)
// ============================================================================

struct OperationInfo {
    std::string op_name;
    std::string op_type;
    int num_inputs;
    int num_outputs;
};

// ============================================================================
// Graph - RAII wrapper for TF_Graph (read-only for inference)
// ============================================================================

class Graph {
public:
    Graph() : state_(std::make_shared<detail::GraphState>()) {}

    ~Graph() = default;

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&& other) noexcept
        : state_(std::move(other.state_)) {}

    Graph& operator=(Graph&& other) noexcept {
        if (this != &other) {
            state_ = std::move(other.state_);
        }
        return *this;
    }
    
    [[nodiscard]] bool valid() const noexcept { 
        return state_ && state_->graph != nullptr; 
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Operation lookup
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] std::optional<TF_Operation*> GetOperation(const std::string& name) const {
        ensure_valid_("GetOperation");
        TF_Operation* op = TF_GraphOperationByName(state_->graph, name.c_str());
        return op ? std::optional{op} : std::nullopt;
    }
    
    [[nodiscard]] TF_Operation* GetOperationOrThrow(const std::string& name) const {
        auto opt = GetOperation(name);
        if (!opt) {
            throw Error::Wrapper(TF_NOT_FOUND, "Graph::GetOperationOrThrow",
                "operation not found in graph", name, -1);
        }
        return *opt;
    }
    
    [[nodiscard]] bool HasOperation(const std::string& name) const {
        return GetOperation(name).has_value();
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Graph introspection
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] std::vector<TF_Operation*> GetAllOperations() const {
        ensure_valid_("GetAllOperations");
        
        std::vector<TF_Operation*> ops;
        std::size_t pos = 0;
        TF_Operation* op;
        
        while ((op = TF_GraphNextOperation(state_->graph, &pos)) != nullptr) {
            ops.push_back(op);
        }
        
        return ops;
    }
    
    [[nodiscard]] std::size_t num_operations() const {
        ensure_valid_("num_operations");
        
        std::size_t count = 0;
        std::size_t pos = 0;
        while (TF_GraphNextOperation(state_->graph, &pos) != nullptr) {
            ++count;
        }
        return count;
    }
    
    [[nodiscard]] std::vector<TF_Operation*> GetOperationsByType(const std::string& op_type) const {
        ensure_valid_("GetOperationsByType");
        
        std::vector<TF_Operation*> ops;
        std::size_t pos = 0;
        TF_Operation* op;
        
        while ((op = TF_GraphNextOperation(state_->graph, &pos)) != nullptr) {
            if (std::string_view(TF_OperationOpType(op)) == op_type) {
                ops.push_back(op);
            }
        }
        
        return ops;
    }
    
    [[nodiscard]] std::vector<OperationInfo> GetOperationInfoByType(const std::string& op_type) const {
        ensure_valid_("GetOperationInfoByType");
        
        std::vector<OperationInfo> result;
        std::size_t pos = 0;
        TF_Operation* op;
        
        while ((op = TF_GraphNextOperation(state_->graph, &pos)) != nullptr) {
            if (std::string_view(TF_OperationOpType(op)) == op_type) {
                result.push_back({
                    TF_OperationName(op),
                    TF_OperationOpType(op),
                    TF_OperationNumInputs(op),
                    TF_OperationNumOutputs(op)
                });
            }
        }
        
        return result;
    }
    
    [[nodiscard]] std::vector<OperationInfo> GetPlaceholders() const {
        return GetOperationInfoByType("Placeholder");
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Serialization (for debugging)
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] std::vector<std::uint8_t> ToGraphDef() const {
        ensure_valid_("ToGraphDef");
        
        TF_Buffer* buf = TF_NewBuffer();
        if (!buf) {
            throw Error::Wrapper(TF_INTERNAL, "Graph::ToGraphDef",
                "TF_NewBuffer failed", "", -1);
        }
        TF_SCOPE_EXIT { TF_DeleteBuffer(buf); };
        
        Status st;
        TF_GraphToGraphDef(state_->graph, buf, st.get());
        st.throw_if_error("TF_GraphToGraphDef");
        
        if (buf->length == 0) {
            return {};
        }
        if (!buf->data) {
            throw Error::Wrapper(TF_INTERNAL, "Graph::ToGraphDef",
                "TF_GraphToGraphDef returned null data", "", -1);
        }

        const auto* p = static_cast<const std::uint8_t*>(buf->data);
        return std::vector<std::uint8_t>(p, p + buf->length);
    }
    
    [[nodiscard]] std::string DebugString() const {
        ensure_valid_("DebugString");
        
        std::string result;
        result += "Graph with " + std::to_string(num_operations()) + " operations:\n";
        
        std::size_t pos = 0;
        TF_Operation* op;
        while ((op = TF_GraphNextOperation(state_->graph, &pos)) != nullptr) {
            const char* name = TF_OperationName(op);
            const char* type = TF_OperationOpType(op);
            const int num_inputs = TF_OperationNumInputs(op);
            const int num_outputs = TF_OperationNumOutputs(op);
            
            result += "  ";
            result += name ? name : "(null)";
            result += " (";
            result += type ? type : "(null)";
            result += ") inputs=";
            result += std::to_string(num_inputs);
            result += " outputs=";
            result += std::to_string(num_outputs);
            result += "\n";
        }
        
        return result;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Handle access and freeze state
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] TF_Graph* handle() const noexcept { 
        return state_ ? state_->graph : nullptr; 
    }
    
    void freeze() noexcept { 
        if (state_) state_->frozen = true; 
    }
    
    [[nodiscard]] bool is_frozen() const noexcept { 
        return state_ && state_->frozen; 
    }

    [[nodiscard]] std::shared_ptr<detail::GraphState> share_state() const noexcept { 
        return state_; 
    }

private:
    std::shared_ptr<detail::GraphState> state_{};
    
    void ensure_valid_(const char* fn) const {
        if (!state_ || !state_->graph) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, 
                detail::format("Graph::{}", fn),
                "graph is in moved-from state", "", -1);
        }
    }
};

} // namespace tf_wrap
