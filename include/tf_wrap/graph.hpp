// tf/graph.hpp
// RAII wrapper for TF_Graph
//
// v5 - Thread safety policies REMOVED (simplification):
// - Single Graph class (no policy templates)
// - No mutex/locking machinery
//
// Thread safety contract:
// - Graph mutation is NOT thread-safe
// - Don't modify graph from multiple threads
// - Graph is frozen after Session creation (this wrapper's policy choice)
// - Session::Run() is thread-safe (TensorFlow's guarantee)

#pragma once

#include <cassert>
#include <cstdio>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" {
#include <tensorflow/c/c_api.h>
}

#include "tf_wrap/format.hpp"
#include "tf_wrap/operation.hpp"
#include "tf_wrap/status.hpp"
#include "tf_wrap/tensor.hpp"

namespace tf_wrap {

// Forward declarations
class Graph;
struct OperationInfo;

// ============================================================================
// OperationBuilder - Fluent builder for graph operations
// ============================================================================

class OperationBuilder {
public:
    OperationBuilder(TF_Graph* graph,
                     const std::string& op_type,
                     const std::string& name)
        : graph_(graph)
        , desc_(TF_NewOperation(graph, op_type.c_str(), name.c_str()))
        , finished_(false)
    {
        if (!desc_) {
            throw std::runtime_error(tf_wrap::detail::format(
                "TF_NewOperation failed: type='{}', name='{}'", op_type, name));
        }
    }
    
    ~OperationBuilder() noexcept {
        if (!finished_ && desc_) {
            #ifdef TF_WRAPPER_TF_STUB_ENABLED
            TF_DeleteOperationDescription(desc_);
            #else
            #ifndef NDEBUG
            std::fprintf(stderr, 
                "[TensorFlowWrap WARNING] OperationBuilder destroyed without Finish() - "
                "operation was discarded (potential resource leak)\n");
            #endif
            #endif
            desc_ = nullptr;
        }
    }
    
    OperationBuilder(const OperationBuilder&) = delete;
    OperationBuilder& operator=(const OperationBuilder&) = delete;
    
    OperationBuilder(OperationBuilder&& other) noexcept
        : graph_(other.graph_)
        , desc_(other.desc_)
        , finished_(other.finished_)
    {
        other.desc_ = nullptr;
        other.finished_ = true;
    }
    
    OperationBuilder& operator=(OperationBuilder&& other) noexcept {
        if (this != &other) {
            graph_ = other.graph_;
            desc_ = other.desc_;
            finished_ = other.finished_;
            other.desc_ = nullptr;
            other.finished_ = true;
        }
        return *this;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Attribute setters (fluent interface)
    // ─────────────────────────────────────────────────────────────────
    
    OperationBuilder& SetAttrTensor(const char* name, TF_Tensor* tensor) & {
        Status st;
        TF_SetAttrTensor(desc_, name, tensor, st.get());
        st.throw_if_error(tf_wrap::detail::format("SetAttrTensor('{}')", name));
        return *this;
    }
    OperationBuilder&& SetAttrTensor(const char* name, TF_Tensor* tensor) && {
        return std::move(SetAttrTensor(name, tensor));
    }
    
    OperationBuilder& SetAttrType(const char* name, TF_DataType dtype) & {
        TF_SetAttrType(desc_, name, dtype);
        return *this;
    }
    OperationBuilder&& SetAttrType(const char* name, TF_DataType dtype) && {
        return std::move(SetAttrType(name, dtype));
    }
    
    OperationBuilder& SetAttrTypeList(const char* name, std::span<const TF_DataType> types) & {
        TF_SetAttrTypeList(desc_, name, types.data(), static_cast<int>(types.size()));
        return *this;
    }
    OperationBuilder&& SetAttrTypeList(const char* name, std::span<const TF_DataType> types) && {
        return std::move(SetAttrTypeList(name, types));
    }
    
    OperationBuilder& SetAttrShape(const char* name, std::span<const std::int64_t> dims) & {
        TF_SetAttrShape(desc_, name, dims.data(), static_cast<int>(dims.size()));
        return *this;
    }
    OperationBuilder&& SetAttrShape(const char* name, std::span<const std::int64_t> dims) && {
        return std::move(SetAttrShape(name, dims));
    }
    OperationBuilder& SetAttrShape(const char* name, std::initializer_list<std::int64_t> dims) & {
        TF_SetAttrShape(desc_, name, dims.begin(), static_cast<int>(dims.size()));
        return *this;
    }
    OperationBuilder&& SetAttrShape(const char* name, std::initializer_list<std::int64_t> dims) && {
        return std::move(SetAttrShape(name, dims));
    }
    
    OperationBuilder& SetAttrInt(const char* name, std::int64_t value) & {
        TF_SetAttrInt(desc_, name, value);
        return *this;
    }
    OperationBuilder&& SetAttrInt(const char* name, std::int64_t value) && {
        return std::move(SetAttrInt(name, value));
    }
    
    OperationBuilder& SetAttrIntList(const char* name, std::span<const std::int64_t> values) & {
        TF_SetAttrIntList(desc_, name, values.data(), static_cast<int>(values.size()));
        return *this;
    }
    OperationBuilder&& SetAttrIntList(const char* name, std::span<const std::int64_t> values) && {
        return std::move(SetAttrIntList(name, values));
    }
    
    OperationBuilder& SetAttrFloat(const char* name, float value) & {
        TF_SetAttrFloat(desc_, name, value);
        return *this;
    }
    OperationBuilder&& SetAttrFloat(const char* name, float value) && {
        return std::move(SetAttrFloat(name, value));
    }
    
    OperationBuilder& SetAttrFloatList(const char* name, std::span<const float> values) & {
        TF_SetAttrFloatList(desc_, name, values.data(), static_cast<int>(values.size()));
        return *this;
    }
    OperationBuilder&& SetAttrFloatList(const char* name, std::span<const float> values) && {
        return std::move(SetAttrFloatList(name, values));
    }
    
    OperationBuilder& SetAttrBool(const char* name, bool value) & {
        TF_SetAttrBool(desc_, name, value ? 1 : 0);
        return *this;
    }
    OperationBuilder&& SetAttrBool(const char* name, bool value) && {
        return std::move(SetAttrBool(name, value));
    }
    
    OperationBuilder& SetAttrString(const char* name, std::string_view value) & {
        TF_SetAttrString(desc_, name, value.data(), value.size());
        return *this;
    }
    OperationBuilder&& SetAttrString(const char* name, std::string_view value) && {
        return std::move(SetAttrString(name, value));
    }
    
    OperationBuilder& SetAttrFuncName(const char* name, std::string_view func_name) & {
        TF_SetAttrFuncName(desc_, name, func_name.data(), func_name.size());
        return *this;
    }
    OperationBuilder&& SetAttrFuncName(const char* name, std::string_view func_name) && {
        return std::move(SetAttrFuncName(name, func_name));
    }
    
    OperationBuilder& SetDevice(const char* device) & {
        TF_SetDevice(desc_, device);
        return *this;
    }
    OperationBuilder&& SetDevice(const char* device) && {
        return std::move(SetDevice(device));
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Input setters
    // ─────────────────────────────────────────────────────────────────
    
    OperationBuilder& AddInput(TF_Output input) & {
        TF_AddInput(desc_, input);
        return *this;
    }
    OperationBuilder&& AddInput(TF_Output input) && {
        return std::move(AddInput(input));
    }
    
    OperationBuilder& AddInput(TF_Operation* op, int index = 0) & {
        return AddInput(TF_Output{op, index});
    }
    OperationBuilder&& AddInput(TF_Operation* op, int index = 0) && {
        return std::move(AddInput(op, index));
    }
    
    OperationBuilder& AddInputList(std::span<const TF_Output> inputs) & {
        TF_AddInputList(desc_, inputs.data(), static_cast<int>(inputs.size()));
        return *this;
    }
    OperationBuilder&& AddInputList(std::span<const TF_Output> inputs) && {
        return std::move(AddInputList(inputs));
    }
    
    OperationBuilder& AddControlInput(TF_Operation* op) & {
        TF_AddControlInput(desc_, op);
        return *this;
    }
    OperationBuilder&& AddControlInput(TF_Operation* op) && {
        return std::move(AddControlInput(op));
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Finish building
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] TF_Operation* Finish() {
        if (finished_) {
            throw std::runtime_error("OperationBuilder::Finish() called twice");
        }
        
        Status st;
        TF_Operation* op = TF_FinishOperation(desc_, st.get());
        finished_ = true;
        desc_ = nullptr;
        
        st.throw_if_error("TF_FinishOperation");
        return op;
    }
    
    [[nodiscard]] bool valid() const noexcept { return desc_ != nullptr && !finished_; }

private:
    TF_Graph* graph_;
    TF_OperationDescription* desc_;
    bool finished_;
};

// ============================================================================
// Graph - RAII wrapper for TF_Graph
// ============================================================================

class Graph {
public:
    Graph() : graph_(TF_NewGraph()) {
        if (!graph_) {
            throw std::runtime_error("TF_NewGraph: failed to create graph");
        }
    }

    ~Graph() {
        if (graph_) {
            TF_DeleteGraph(graph_);
        }
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&& other) noexcept
        : graph_(other.graph_)
        , frozen_(other.frozen_)
    {
        other.graph_ = nullptr;
        other.frozen_ = false;
    }

    Graph& operator=(Graph&& other) noexcept {
        if (this != &other) {
            if (graph_) TF_DeleteGraph(graph_);
            graph_ = other.graph_;
            frozen_ = other.frozen_;
            other.graph_ = nullptr;
            other.frozen_ = false;
        }
        return *this;
    }
    
    [[nodiscard]] bool valid() const noexcept { return graph_ != nullptr; }
    
    // ─────────────────────────────────────────────────────────────────
    // Import GraphDef
    // ─────────────────────────────────────────────────────────────────
    
    void ImportGraphDef(const void* proto, std::size_t proto_len, const char* prefix = "") {
        ensure_valid_("ImportGraphDef");
        ensure_not_frozen_("ImportGraphDef");
        
        TF_Buffer* buf = TF_NewBufferFromString(proto, proto_len);
        if (!buf) {
            throw std::runtime_error("ImportGraphDef: TF_NewBufferFromString failed");
        }
        
        TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
        if (!opts) {
            TF_DeleteBuffer(buf);
            throw std::runtime_error("ImportGraphDef: TF_NewImportGraphDefOptions failed");
        }
        
        if (prefix && prefix[0] != '\0') {
            TF_ImportGraphDefOptionsSetPrefix(opts, prefix);
        }
        
        Status st;
        TF_GraphImportGraphDef(graph_, buf, opts, st.get());
        
        TF_DeleteImportGraphDefOptions(opts);
        TF_DeleteBuffer(buf);
        
        st.throw_if_error("TF_GraphImportGraphDef");
    }
    
    void ImportGraphDef(const TF_Buffer* buf, const char* prefix = "") {
        if (!buf || !buf->data) {
            throw std::invalid_argument("ImportGraphDef: null buffer");
        }
        ImportGraphDef(buf->data, buf->length, prefix);
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Operation lookup
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] std::optional<TF_Operation*> GetOperation(const std::string& name) const {
        ensure_valid_("GetOperation");
        TF_Operation* op = TF_GraphOperationByName(graph_, name.c_str());
        return op ? std::optional{op} : std::nullopt;
    }
    
    [[nodiscard]] TF_Operation* GetOperationOrThrow(const std::string& name) const {
        auto opt = GetOperation(name);
        if (!opt) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Operation '{}' not found in graph", name));
        }
        return *opt;
    }
    
    [[nodiscard]] bool HasOperation(const std::string& name) const {
        return GetOperation(name).has_value();
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Create new operations
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] OperationBuilder NewOperation(const std::string& op_type, const std::string& name) {
        ensure_valid_("NewOperation");
        ensure_not_frozen_("NewOperation");
        return OperationBuilder(graph_, op_type, name);
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Graph info
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] std::vector<TF_Operation*> GetAllOperations() const {
        ensure_valid_("GetAllOperations");
        
        std::vector<TF_Operation*> ops;
        std::size_t pos = 0;
        TF_Operation* op;
        
        while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
            ops.push_back(op);
        }
        
        return ops;
    }
    
    [[nodiscard]] std::size_t num_operations() const {
        ensure_valid_("num_operations");
        
        std::size_t count = 0;
        std::size_t pos = 0;
        while (TF_GraphNextOperation(graph_, &pos) != nullptr) {
            ++count;
        }
        return count;
    }
    
    [[nodiscard]] std::vector<std::uint8_t> ToGraphDef() const {
        ensure_valid_("ToGraphDef");
        
        TF_Buffer* buf = TF_NewBuffer();
        if (!buf) {
            throw std::runtime_error("ToGraphDef: TF_NewBuffer failed");
        }
        
        Status st;
        TF_GraphToGraphDef(graph_, buf, st.get());
        
        if (!st.ok()) {
            TF_DeleteBuffer(buf);
            st.throw_if_error("TF_GraphToGraphDef");
        }
        
        std::vector<std::uint8_t> result(
            static_cast<const std::uint8_t*>(buf->data),
            static_cast<const std::uint8_t*>(buf->data) + buf->length);
        
        TF_DeleteBuffer(buf);
        return result;
    }
    
    [[nodiscard]] std::vector<TF_Operation*> GetOperationsByType(const std::string& op_type) const {
        ensure_valid_("GetOperationsByType");
        
        std::vector<TF_Operation*> ops;
        std::size_t pos = 0;
        TF_Operation* op;
        
        while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
            if (std::string_view(TF_OperationOpType(op)) == op_type) {
                ops.push_back(op);
            }
        }
        
        return ops;
    }
    
    [[nodiscard]] std::string DebugString() const {
        ensure_valid_("DebugString");
        
        std::size_t count = 0;
        std::size_t pos = 0;
        while (TF_GraphNextOperation(graph_, &pos) != nullptr) {
            ++count;
        }
        pos = 0;
        
        std::string result;
        result += "Graph with " + std::to_string(count) + " operations:\n";
        
        TF_Operation* op;
        while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
            const char* name = TF_OperationName(op);
            const char* type = TF_OperationOpType(op);
            const int num_inputs = TF_OperationNumInputs(op);
            const int num_outputs = TF_OperationNumOutputs(op);
            
            result += "  ";
            result += name;
            result += " (";
            result += type;
            result += ") inputs=";
            result += std::to_string(num_inputs);
            result += " outputs=";
            result += std::to_string(num_outputs);
            result += "\n";
        }
        
        return result;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Raw handle and freeze
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] TF_Graph* handle() const noexcept { return graph_; }
    
    void freeze() noexcept { frozen_ = true; }
    [[nodiscard]] bool is_frozen() const noexcept { return frozen_; }
    
    // Methods defined after OperationInfo struct
    [[nodiscard]] std::vector<OperationInfo> GetPlaceholders() const;
    [[nodiscard]] std::vector<OperationInfo> GetOperationInfoByType(const std::string& op_type) const;

private:
    TF_Graph* graph_{nullptr};
    bool frozen_{false};
    
    void ensure_valid_(const char* fn) const {
        if (!graph_) {
            throw std::runtime_error(std::string(fn) + ": Graph is in moved-from state");
        }
    }
    
    void ensure_not_frozen_(const char* fn) const {
        if (frozen_) {
            throw std::runtime_error(std::string(fn) + 
                ": Graph is frozen (a Session has been created from it). "
                "This wrapper requires graphs to be immutable after Session creation.");
        }
    }
};

// ============================================================================
// GraphFunction - RAII wrapper for TF_Function
// ============================================================================

class GraphFunction {
public:
    GraphFunction() = default;
    
    ~GraphFunction() {
        if (func_) TF_DeleteFunction(func_);
    }
    
    GraphFunction(const GraphFunction&) = delete;
    GraphFunction& operator=(const GraphFunction&) = delete;
    
    GraphFunction(GraphFunction&& other) noexcept : func_(other.func_) {
        other.func_ = nullptr;
    }
    
    GraphFunction& operator=(GraphFunction&& other) noexcept {
        if (this != &other) {
            if (func_) TF_DeleteFunction(func_);
            func_ = other.func_;
            other.func_ = nullptr;
        }
        return *this;
    }
    
    [[nodiscard]] static GraphFunction FromGraph(
        const Graph& graph,
        const std::string& name,
        std::span<const TF_Output> inputs,
        std::span<const TF_Output> outputs,
        const std::string& description = "")
    {
        Status st;
        
        TF_Function* func = TF_GraphToFunction(
            graph.handle(),
            name.c_str(),
            0,
            -1,
            nullptr,
            static_cast<int>(inputs.size()),
            inputs.data(),
            static_cast<int>(outputs.size()),
            outputs.data(),
            nullptr,
            nullptr,
            description.empty() ? nullptr : description.c_str(),
            st.get());
        
        st.throw_if_error("TF_GraphToFunction");
        
        GraphFunction result;
        result.func_ = func;
        return result;
    }
    
    [[nodiscard]] bool valid() const noexcept { return func_ != nullptr; }
    [[nodiscard]] TF_Function* handle() const noexcept { return func_; }
    
    [[nodiscard]] const char* name() const {
        if (!func_) return "";
        return TF_FunctionName(func_);
    }
    
    void CopyTo(Graph& graph, const GraphFunction* grad = nullptr) const {
        if (!func_) {
            throw std::runtime_error("GraphFunction::CopyTo: function is null");
        }
        
        Status st;
        TF_GraphCopyFunction(
            graph.handle(),
            func_,
            grad ? grad->handle() : nullptr,
            st.get());
        
        st.throw_if_error("TF_GraphCopyFunction");
    }
    
private:
    TF_Function* func_{nullptr};
};

// ============================================================================
// OperationInfo - Information about an operation
// ============================================================================

struct OperationInfo {
    std::string op_name;
    std::string op_type;
    int num_inputs;
    int num_outputs;
};

// ============================================================================
// Graph member: GetPlaceholders
// ============================================================================

inline std::vector<OperationInfo> Graph::GetPlaceholders() const {
    return GetOperationInfoByType("Placeholder");
}

inline std::vector<OperationInfo> Graph::GetOperationInfoByType(const std::string& op_type) const {
    ensure_valid_("GetOperationInfoByType");
    
    std::vector<OperationInfo> result;
    std::size_t pos = 0;
    TF_Operation* op;
    
    while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
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

// Backward compatibility alias

} // namespace tf_wrap
