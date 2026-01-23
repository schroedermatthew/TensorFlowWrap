#!/usr/bin/env python3
"""
TensorFlow Ops C++ Wrapper Generator

This script generates C++20 type-safe wrappers for TensorFlow operations.
It can either:
1. Extract ops from TensorFlow's op registry (requires TensorFlow installed)
2. Use a built-in list of common ops

Usage:
    python generate_ops.py --from-tf      # Generate from TensorFlow
    python generate_ops.py --builtin      # Use built-in op list
    python generate_ops.py --output FILE  # Specify output file
"""

import argparse
import datetime
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict

# =============================================================================
# Op Definition Data Structures
# =============================================================================

@dataclass
class OpInput:
    name: str
    dtype: str  # "tensor", "tensor_list", "int", "float", "string", etc.
    description: str = ""

@dataclass
class OpAttr:
    name: str
    dtype: str  # "type", "int", "float", "bool", "string", "shape", "list(int)", etc.
    default: Optional[str] = None
    description: str = ""

@dataclass
class OpOutput:
    name: str
    dtype: str
    description: str = ""

@dataclass
class OpDef:
    name: str
    inputs: List[OpInput]
    outputs: List[OpOutput]
    attrs: List[OpAttr]
    description: str = ""
    category: str = "misc"

# =============================================================================
# Built-in Op Definitions (Common TensorFlow Operations)
# =============================================================================

BUILTIN_OPS = [
    # =========================================================================
    # Math Operations
    # =========================================================================
    OpDef("Add", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x + y element-wise", "math"),
    
    OpDef("AddV2", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x + y element-wise (with broadcasting)", "math"),
    
    OpDef("Sub", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x - y element-wise", "math"),
    
    OpDef("Mul", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x * y element-wise", "math"),
    
    OpDef("Div", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x / y element-wise", "math"),
    
    OpDef("RealDiv", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x / y element-wise for real types", "math"),
    
    OpDef("FloorDiv", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns floor(x / y) element-wise", "math"),
    
    OpDef("Mod", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x % y element-wise", "math"),
    
    OpDef("Pow", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x^y element-wise", "math"),
    
    OpDef("Maximum", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns max(x, y) element-wise", "math"),
    
    OpDef("Minimum", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns min(x, y) element-wise", "math"),
    
    OpDef("Neg", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns -x element-wise", "math"),
    
    OpDef("Abs", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns |x| element-wise", "math"),
    
    OpDef("Sign", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns sign of x element-wise", "math"),
    
    OpDef("Reciprocal", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns 1/x element-wise", "math"),
    
    OpDef("Square", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns x^2 element-wise", "math"),
    
    OpDef("Sqrt", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns sqrt(x) element-wise", "math"),
    
    OpDef("Rsqrt", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns 1/sqrt(x) element-wise", "math"),
    
    OpDef("Exp", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns e^x element-wise", "math"),
    
    OpDef("Expm1", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns e^x - 1 element-wise", "math"),
    
    OpDef("Log", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns ln(x) element-wise", "math"),
    
    OpDef("Log1p", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns ln(1 + x) element-wise", "math"),
    
    OpDef("Sin", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns sin(x) element-wise", "math"),
    
    OpDef("Cos", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns cos(x) element-wise", "math"),
    
    OpDef("Tan", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns tan(x) element-wise", "math"),
    
    OpDef("Asin", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns asin(x) element-wise", "math"),
    
    OpDef("Acos", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns acos(x) element-wise", "math"),
    
    OpDef("Atan", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns atan(x) element-wise", "math"),
    
    OpDef("Sinh", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns sinh(x) element-wise", "math"),
    
    OpDef("Cosh", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns cosh(x) element-wise", "math"),
    
    OpDef("Tanh", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns tanh(x) element-wise", "math"),
    
    OpDef("Ceil", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns ceil(x) element-wise", "math"),
    
    OpDef("Floor", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns floor(x) element-wise", "math"),
    
    OpDef("Round", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns round(x) element-wise", "math"),
    
    OpDef("Rint", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns round to nearest integer element-wise", "math"),
    
    OpDef("Sigmoid", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Returns 1/(1+e^(-x)) element-wise", "math"),

    # =========================================================================
    # Matrix Operations
    # =========================================================================
    OpDef("MatMul", [OpInput("a", "tensor"), OpInput("b", "tensor")],
          [OpOutput("product", "tensor")],
          [OpAttr("T", "type"), OpAttr("transpose_a", "bool", "false"),
           OpAttr("transpose_b", "bool", "false")],
          "Matrix multiplication", "matrix"),
    
    OpDef("BatchMatMul", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("adj_x", "bool", "false"),
           OpAttr("adj_y", "bool", "false")],
          "Batched matrix multiplication", "matrix"),
    
    OpDef("BatchMatMulV2", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("adj_x", "bool", "false"),
           OpAttr("adj_y", "bool", "false")],
          "Batched matrix multiplication with broadcasting", "matrix"),
    
    OpDef("Transpose", [OpInput("x", "tensor"), OpInput("perm", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type"), OpAttr("Tperm", "type")],
          "Permutes dimensions according to perm", "matrix"),
    
    OpDef("MatrixInverse", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("adjoint", "bool", "false")],
          "Matrix inverse", "matrix"),
    
    OpDef("MatrixDeterminant", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type")],
          "Matrix determinant", "matrix"),
    
    OpDef("Cholesky", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type")],
          "Cholesky decomposition", "matrix"),
    
    OpDef("Qr", [OpInput("input", "tensor")],
          [OpOutput("q", "tensor"), OpOutput("r", "tensor")],
          [OpAttr("T", "type"), OpAttr("full_matrices", "bool", "false")],
          "QR decomposition", "matrix"),
    
    OpDef("Svd", [OpInput("input", "tensor")],
          [OpOutput("s", "tensor"), OpOutput("u", "tensor"), OpOutput("v", "tensor")],
          [OpAttr("T", "type"), OpAttr("compute_uv", "bool", "true"),
           OpAttr("full_matrices", "bool", "false")],
          "SVD decomposition", "matrix"),
    
    OpDef("Einsum", [OpInput("inputs", "tensor_list")],
          [OpOutput("output", "tensor")],
          [OpAttr("equation", "string"), OpAttr("T", "type"), OpAttr("N", "int")],
          "Einstein summation", "matrix"),

    # =========================================================================
    # Reduction Operations
    # =========================================================================
    OpDef("Sum", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tidx", "type"), OpAttr("keep_dims", "bool", "false")],
          "Sum along axis", "reduce"),
    
    OpDef("Prod", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tidx", "type"), OpAttr("keep_dims", "bool", "false")],
          "Product along axis", "reduce"),
    
    OpDef("Mean", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tidx", "type"), OpAttr("keep_dims", "bool", "false")],
          "Mean along axis", "reduce"),
    
    OpDef("Max", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tidx", "type"), OpAttr("keep_dims", "bool", "false")],
          "Maximum along axis", "reduce"),
    
    OpDef("Min", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tidx", "type"), OpAttr("keep_dims", "bool", "false")],
          "Minimum along axis", "reduce"),
    
    OpDef("All", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("Tidx", "type"), OpAttr("keep_dims", "bool", "false")],
          "Logical AND along axis", "reduce"),
    
    OpDef("Any", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("Tidx", "type"), OpAttr("keep_dims", "bool", "false")],
          "Logical OR along axis", "reduce"),
    
    OpDef("ArgMax", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tidx", "type"), OpAttr("output_type", "type")],
          "Index of maximum along axis", "reduce"),
    
    OpDef("ArgMin", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tidx", "type"), OpAttr("output_type", "type")],
          "Index of minimum along axis", "reduce"),

    # =========================================================================
    # Comparison Operations
    # =========================================================================
    OpDef("Equal", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x == y element-wise", "compare"),
    
    OpDef("NotEqual", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x != y element-wise", "compare"),
    
    OpDef("Less", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x < y element-wise", "compare"),
    
    OpDef("LessEqual", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x <= y element-wise", "compare"),
    
    OpDef("Greater", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x > y element-wise", "compare"),
    
    OpDef("GreaterEqual", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [OpAttr("T", "type")],
          "Returns x >= y element-wise", "compare"),
    
    OpDef("LogicalAnd", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [],
          "Returns x AND y element-wise", "compare"),
    
    OpDef("LogicalOr", [OpInput("x", "tensor"), OpInput("y", "tensor")],
          [OpOutput("z", "tensor")], [],
          "Returns x OR y element-wise", "compare"),
    
    OpDef("LogicalNot", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [],
          "Returns NOT x element-wise", "compare"),

    # =========================================================================
    # Neural Network Operations
    # =========================================================================
    OpDef("Relu", [OpInput("features", "tensor")],
          [OpOutput("activations", "tensor")], [OpAttr("T", "type")],
          "ReLU activation: max(0, x)", "nn"),
    
    OpDef("Relu6", [OpInput("features", "tensor")],
          [OpOutput("activations", "tensor")], [OpAttr("T", "type")],
          "ReLU6 activation: min(max(0, x), 6)", "nn"),
    
    OpDef("LeakyRelu", [OpInput("features", "tensor")],
          [OpOutput("activations", "tensor")],
          [OpAttr("T", "type"), OpAttr("alpha", "float", "0.2")],
          "Leaky ReLU activation", "nn"),
    
    OpDef("Elu", [OpInput("features", "tensor")],
          [OpOutput("activations", "tensor")], [OpAttr("T", "type")],
          "ELU activation", "nn"),
    
    OpDef("Selu", [OpInput("features", "tensor")],
          [OpOutput("activations", "tensor")], [OpAttr("T", "type")],
          "SELU activation", "nn"),
    
    OpDef("Softmax", [OpInput("logits", "tensor")],
          [OpOutput("softmax", "tensor")], [OpAttr("T", "type")],
          "Softmax activation", "nn"),
    
    OpDef("LogSoftmax", [OpInput("logits", "tensor")],
          [OpOutput("logsoftmax", "tensor")], [OpAttr("T", "type")],
          "Log-softmax activation", "nn"),
    
    OpDef("Softplus", [OpInput("features", "tensor")],
          [OpOutput("activations", "tensor")], [OpAttr("T", "type")],
          "Softplus activation: ln(1 + e^x)", "nn"),
    
    OpDef("Softsign", [OpInput("features", "tensor")],
          [OpOutput("activations", "tensor")], [OpAttr("T", "type")],
          "Softsign activation: x / (|x| + 1)", "nn"),
    
    OpDef("BiasAdd", [OpInput("value", "tensor"), OpInput("bias", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("data_format", "string", "NHWC")],
          "Adds bias to value", "nn"),
    
    OpDef("Conv2D", [OpInput("input", "tensor"), OpInput("filter", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("strides", "list(int)"),
           OpAttr("padding", "string"), OpAttr("data_format", "string", "NHWC"),
           OpAttr("dilations", "list(int)", "[1,1,1,1]")],
          "2D convolution", "nn"),
    
    OpDef("Conv2DBackpropInput",
          [OpInput("input_sizes", "tensor"), OpInput("filter", "tensor"),
           OpInput("out_backprop", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("strides", "list(int)"),
           OpAttr("padding", "string"), OpAttr("data_format", "string", "NHWC"),
           OpAttr("dilations", "list(int)", "[1,1,1,1]")],
          "Conv2D input gradient (transposed convolution)", "nn"),
    
    OpDef("DepthwiseConv2dNative",
          [OpInput("input", "tensor"), OpInput("filter", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("strides", "list(int)"),
           OpAttr("padding", "string"), OpAttr("data_format", "string", "NHWC"),
           OpAttr("dilations", "list(int)", "[1,1,1,1]")],
          "Depthwise 2D convolution", "nn"),
    
    OpDef("MaxPool", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("ksize", "list(int)"),
           OpAttr("strides", "list(int)"), OpAttr("padding", "string"),
           OpAttr("data_format", "string", "NHWC")],
          "Max pooling", "nn"),
    
    OpDef("AvgPool", [OpInput("value", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("ksize", "list(int)"),
           OpAttr("strides", "list(int)"), OpAttr("padding", "string"),
           OpAttr("data_format", "string", "NHWC")],
          "Average pooling", "nn"),
    
    OpDef("MaxPool3D", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("ksize", "list(int)"),
           OpAttr("strides", "list(int)"), OpAttr("padding", "string"),
           OpAttr("data_format", "string", "NDHWC")],
          "3D max pooling", "nn"),
    
    OpDef("AvgPool3D", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("ksize", "list(int)"),
           OpAttr("strides", "list(int)"), OpAttr("padding", "string"),
           OpAttr("data_format", "string", "NDHWC")],
          "3D average pooling", "nn"),
    
    OpDef("FusedBatchNorm",
          [OpInput("x", "tensor"), OpInput("scale", "tensor"),
           OpInput("offset", "tensor"), OpInput("mean", "tensor"),
           OpInput("variance", "tensor")],
          [OpOutput("y", "tensor"), OpOutput("batch_mean", "tensor"),
           OpOutput("batch_variance", "tensor"), OpOutput("reserve_space_1", "tensor"),
           OpOutput("reserve_space_2", "tensor")],
          [OpAttr("T", "type"), OpAttr("epsilon", "float", "0.0001"),
           OpAttr("data_format", "string", "NHWC"), OpAttr("is_training", "bool", "true")],
          "Fused batch normalization", "nn"),
    
    OpDef("FusedBatchNormV3",
          [OpInput("x", "tensor"), OpInput("scale", "tensor"),
           OpInput("offset", "tensor"), OpInput("mean", "tensor"),
           OpInput("variance", "tensor")],
          [OpOutput("y", "tensor"), OpOutput("batch_mean", "tensor"),
           OpOutput("batch_variance", "tensor"), OpOutput("reserve_space_1", "tensor"),
           OpOutput("reserve_space_2", "tensor"), OpOutput("reserve_space_3", "tensor")],
          [OpAttr("T", "type"), OpAttr("U", "type"), OpAttr("epsilon", "float", "0.0001"),
           OpAttr("data_format", "string", "NHWC"), OpAttr("is_training", "bool", "true")],
          "Fused batch normalization V3", "nn"),
    
    OpDef("LRN", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("depth_radius", "int", "5"),
           OpAttr("bias", "float", "1"), OpAttr("alpha", "float", "1"),
           OpAttr("beta", "float", "0.5")],
          "Local response normalization", "nn"),
    
    OpDef("Dropout", [OpInput("x", "tensor"), OpInput("rate", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Dropout regularization", "nn"),
    
    OpDef("SoftmaxCrossEntropyWithLogits",
          [OpInput("features", "tensor"), OpInput("labels", "tensor")],
          [OpOutput("loss", "tensor"), OpOutput("backprop", "tensor")],
          [OpAttr("T", "type")],
          "Softmax cross entropy loss", "nn"),
    
    OpDef("SparseSoftmaxCrossEntropyWithLogits",
          [OpInput("features", "tensor"), OpInput("labels", "tensor")],
          [OpOutput("loss", "tensor"), OpOutput("backprop", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tlabels", "type")],
          "Sparse softmax cross entropy loss", "nn"),

    # =========================================================================
    # Array Operations
    # =========================================================================
    OpDef("Const", [],
          [OpOutput("output", "tensor")],
          [OpAttr("value", "tensor"), OpAttr("dtype", "type")],
          "Constant tensor", "array"),
    
    OpDef("Placeholder", [],
          [OpOutput("output", "tensor")],
          [OpAttr("dtype", "type"), OpAttr("shape", "shape")],
          "Placeholder for feeding data", "array"),
    
    OpDef("Identity", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type")],
          "Identity function", "array"),
    
    OpDef("IdentityN", [OpInput("input", "tensor_list")],
          [OpOutput("output", "tensor_list")], [OpAttr("T", "list(type)")],
          "Identity for multiple tensors", "array"),
    
    OpDef("Reshape", [OpInput("tensor", "tensor"), OpInput("shape", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type"), OpAttr("Tshape", "type")],
          "Reshape tensor", "array"),
    
    OpDef("Squeeze", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("squeeze_dims", "list(int)", "[]")],
          "Remove size-1 dimensions", "array"),
    
    OpDef("ExpandDims", [OpInput("input", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type"), OpAttr("Tdim", "type")],
          "Insert dimension of size 1", "array"),
    
    OpDef("Concat", [OpInput("concat_dim", "tensor"), OpInput("values", "tensor_list")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type"), OpAttr("N", "int")],
          "Concatenate tensors", "array"),
    
    OpDef("ConcatV2", [OpInput("values", "tensor_list"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("N", "int"), OpAttr("Tidx", "type")],
          "Concatenate tensors V2", "array"),
    
    OpDef("Split", [OpInput("split_dim", "tensor"), OpInput("value", "tensor")],
          [OpOutput("output", "tensor_list")],
          [OpAttr("T", "type"), OpAttr("num_split", "int")],
          "Split tensor into subtensors", "array"),
    
    OpDef("SplitV", [OpInput("value", "tensor"), OpInput("size_splits", "tensor"),
                    OpInput("split_dim", "tensor")],
          [OpOutput("output", "tensor_list")],
          [OpAttr("T", "type"), OpAttr("num_split", "int"), OpAttr("Tlen", "type")],
          "Split tensor with variable sizes", "array"),
    
    OpDef("Slice", [OpInput("input", "tensor"), OpInput("begin", "tensor"),
                   OpInput("size", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Index", "type")],
          "Slice from tensor", "array"),
    
    OpDef("StridedSlice", [OpInput("input", "tensor"), OpInput("begin", "tensor"),
                          OpInput("end", "tensor"), OpInput("strides", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Index", "type"),
           OpAttr("begin_mask", "int", "0"), OpAttr("end_mask", "int", "0"),
           OpAttr("ellipsis_mask", "int", "0"), OpAttr("new_axis_mask", "int", "0"),
           OpAttr("shrink_axis_mask", "int", "0")],
          "Strided slice from tensor", "array"),
    
    OpDef("Gather", [OpInput("params", "tensor"), OpInput("indices", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("Tparams", "type"), OpAttr("Tindices", "type")],
          "Gather slices from tensor", "array"),
    
    OpDef("GatherV2", [OpInput("params", "tensor"), OpInput("indices", "tensor"),
                      OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("Tparams", "type"), OpAttr("Tindices", "type"),
           OpAttr("Taxis", "type"), OpAttr("batch_dims", "int", "0")],
          "Gather slices with axis", "array"),
    
    OpDef("GatherNd", [OpInput("params", "tensor"), OpInput("indices", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("Tparams", "type"), OpAttr("Tindices", "type")],
          "Gather slices with N-dimensional indices", "array"),
    
    OpDef("ScatterNd", [OpInput("indices", "tensor"), OpInput("updates", "tensor"),
                       OpInput("shape", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tindices", "type")],
          "Scatter updates into tensor", "array"),
    
    OpDef("Tile", [OpInput("input", "tensor"), OpInput("multiples", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tmultiples", "type")],
          "Tile tensor", "array"),
    
    OpDef("Pad", [OpInput("input", "tensor"), OpInput("paddings", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tpaddings", "type")],
          "Pad tensor with zeros", "array"),
    
    OpDef("PadV2", [OpInput("input", "tensor"), OpInput("paddings", "tensor"),
                   OpInput("constant_values", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tpaddings", "type")],
          "Pad tensor with constant value", "array"),
    
    OpDef("MirrorPad", [OpInput("input", "tensor"), OpInput("paddings", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tpaddings", "type"), OpAttr("mode", "string")],
          "Pad tensor with mirrored values", "array"),
    
    OpDef("ReverseV2", [OpInput("tensor", "tensor"), OpInput("axis", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("Tidx", "type")],
          "Reverse tensor along axes", "array"),
    
    OpDef("Pack", [OpInput("values", "tensor_list")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("N", "int"), OpAttr("axis", "int", "0")],
          "Stack tensors along axis (pack)", "array"),
    
    OpDef("Unpack", [OpInput("value", "tensor")],
          [OpOutput("output", "tensor_list")],
          [OpAttr("T", "type"), OpAttr("num", "int"), OpAttr("axis", "int", "0")],
          "Unstack tensor along axis (unpack)", "array"),
    
    OpDef("Shape", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type"), OpAttr("out_type", "type")],
          "Get tensor shape", "array"),
    
    OpDef("ShapeN", [OpInput("input", "tensor_list")],
          [OpOutput("output", "tensor_list")],
          [OpAttr("T", "type"), OpAttr("N", "int"), OpAttr("out_type", "type")],
          "Get shapes of multiple tensors", "array"),
    
    OpDef("Rank", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type")],
          "Get tensor rank", "array"),
    
    OpDef("Size", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type"), OpAttr("out_type", "type")],
          "Get number of elements", "array"),
    
    OpDef("Fill", [OpInput("dims", "tensor"), OpInput("value", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("index_type", "type")],
          "Fill tensor with scalar value", "array"),
    
    OpDef("Zeros", [OpInput("shape", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type")],
          "Create tensor of zeros", "array"),
    
    OpDef("ZerosLike", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Create tensor of zeros with same shape", "array"),
    
    OpDef("OnesLike", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")], [OpAttr("T", "type")],
          "Create tensor of ones with same shape", "array"),
    
    OpDef("Range", [OpInput("start", "tensor"), OpInput("limit", "tensor"),
                   OpInput("delta", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("Tidx", "type")],
          "Create range [start, limit) with delta step", "array"),
    
    OpDef("LinSpace", [OpInput("start", "tensor"), OpInput("stop", "tensor"),
                      OpInput("num", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type"), OpAttr("Tidx", "type")],
          "Create linearly spaced values", "array"),
    
    OpDef("BroadcastTo", [OpInput("input", "tensor"), OpInput("shape", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type"), OpAttr("Tidx", "type")],
          "Broadcast to shape", "array"),
    
    OpDef("Where", [OpInput("condition", "tensor")],
          [OpOutput("index", "tensor")], [OpAttr("T", "type")],
          "Returns indices of true elements", "array"),
    
    OpDef("SelectV2", [OpInput("condition", "tensor"), OpInput("t", "tensor"),
                      OpInput("e", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type")],
          "Select elements based on condition", "array"),

    # =========================================================================
    # Cast and Type Operations
    # =========================================================================
    OpDef("Cast", [OpInput("x", "tensor")],
          [OpOutput("y", "tensor")],
          [OpAttr("SrcT", "type"), OpAttr("DstT", "type"), OpAttr("Truncate", "bool", "false")],
          "Cast tensor to different type", "cast"),
    
    OpDef("Bitcast", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type"), OpAttr("type", "type")],
          "Bitcast without copying data", "cast"),
    
    OpDef("CheckNumerics", [OpInput("tensor", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("message", "string")],
          "Check for NaN/Inf values", "cast"),

    # =========================================================================
    # Random Operations
    # =========================================================================
    OpDef("RandomUniform", [OpInput("shape", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("seed", "int", "0"), OpAttr("seed2", "int", "0"),
           OpAttr("dtype", "type"), OpAttr("T", "type")],
          "Uniform random values [0, 1)", "random"),
    
    OpDef("RandomStandardNormal", [OpInput("shape", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("seed", "int", "0"), OpAttr("seed2", "int", "0"),
           OpAttr("dtype", "type"), OpAttr("T", "type")],
          "Standard normal random values", "random"),
    
    OpDef("TruncatedNormal", [OpInput("shape", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("seed", "int", "0"), OpAttr("seed2", "int", "0"),
           OpAttr("dtype", "type"), OpAttr("T", "type")],
          "Truncated normal random values", "random"),
    
    OpDef("RandomShuffle", [OpInput("value", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("seed", "int", "0"), OpAttr("seed2", "int", "0"), OpAttr("T", "type")],
          "Randomly shuffle tensor", "random"),
    
    OpDef("Multinomial", [OpInput("logits", "tensor"), OpInput("num_samples", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("seed", "int", "0"), OpAttr("seed2", "int", "0"),
           OpAttr("T", "type"), OpAttr("output_dtype", "type")],
          "Draw samples from multinomial distribution", "random"),

    # =========================================================================
    # Image Operations
    # =========================================================================
    OpDef("ResizeBilinear", [OpInput("images", "tensor"), OpInput("size", "tensor")],
          [OpOutput("resized_images", "tensor")],
          [OpAttr("T", "type"), OpAttr("align_corners", "bool", "false"),
           OpAttr("half_pixel_centers", "bool", "false")],
          "Resize images using bilinear interpolation", "image"),
    
    OpDef("ResizeNearestNeighbor", [OpInput("images", "tensor"), OpInput("size", "tensor")],
          [OpOutput("resized_images", "tensor")],
          [OpAttr("T", "type"), OpAttr("align_corners", "bool", "false"),
           OpAttr("half_pixel_centers", "bool", "false")],
          "Resize images using nearest neighbor", "image"),
    
    OpDef("ResizeBicubic", [OpInput("images", "tensor"), OpInput("size", "tensor")],
          [OpOutput("resized_images", "tensor")],
          [OpAttr("T", "type"), OpAttr("align_corners", "bool", "false"),
           OpAttr("half_pixel_centers", "bool", "false")],
          "Resize images using bicubic interpolation", "image"),
    
    OpDef("CropAndResize", [OpInput("image", "tensor"), OpInput("boxes", "tensor"),
                            OpInput("box_ind", "tensor"), OpInput("crop_size", "tensor")],
          [OpOutput("crops", "tensor")],
          [OpAttr("T", "type"), OpAttr("method", "string", "bilinear"),
           OpAttr("extrapolation_value", "float", "0")],
          "Extract and resize crops from images", "image"),
    
    OpDef("NonMaxSuppression", [OpInput("boxes", "tensor"), OpInput("scores", "tensor"),
                                OpInput("max_output_size", "tensor")],
          [OpOutput("selected_indices", "tensor")],
          [OpAttr("iou_threshold", "float", "0.5")],
          "Non-maximum suppression for object detection", "image"),
    
    OpDef("NonMaxSuppressionV3",
          [OpInput("boxes", "tensor"), OpInput("scores", "tensor"),
           OpInput("max_output_size", "tensor"), OpInput("iou_threshold", "tensor"),
           OpInput("score_threshold", "tensor")],
          [OpOutput("selected_indices", "tensor")], [],
          "Non-maximum suppression V3", "image"),
    
    OpDef("DecodeJpeg", [OpInput("contents", "tensor")],
          [OpOutput("image", "tensor")],
          [OpAttr("channels", "int", "0"), OpAttr("ratio", "int", "1"),
           OpAttr("fancy_upscaling", "bool", "true"),
           OpAttr("try_recover_truncated", "bool", "false"),
           OpAttr("acceptable_fraction", "float", "1"),
           OpAttr("dct_method", "string", "")],
          "Decode JPEG image", "image"),
    
    OpDef("DecodePng", [OpInput("contents", "tensor")],
          [OpOutput("image", "tensor")],
          [OpAttr("channels", "int", "0"), OpAttr("dtype", "type")],
          "Decode PNG image", "image"),
    
    OpDef("EncodeJpeg", [OpInput("image", "tensor")],
          [OpOutput("contents", "tensor")],
          [OpAttr("format", "string", ""), OpAttr("quality", "int", "95"),
           OpAttr("progressive", "bool", "false"), OpAttr("optimize_size", "bool", "false"),
           OpAttr("chroma_downsampling", "bool", "true"),
           OpAttr("density_unit", "string", "in"), OpAttr("x_density", "int", "300"),
           OpAttr("y_density", "int", "300"), OpAttr("xmp_metadata", "string", "")],
          "Encode image as JPEG", "image"),
    
    OpDef("EncodePng", [OpInput("image", "tensor")],
          [OpOutput("contents", "tensor")],
          [OpAttr("compression", "int", "-1"), OpAttr("T", "type")],
          "Encode image as PNG", "image"),

    # =========================================================================
    # Control Flow Operations
    # =========================================================================
    OpDef("NoOp", [], [], [], "No operation (placeholder)", "control"),
    
    OpDef("StopGradient", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")], [OpAttr("T", "type")],
          "Stop gradient propagation", "control"),
    
    OpDef("PreventGradient", [OpInput("input", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("message", "string", "")],
          "Prevent gradient propagation with message", "control"),
    
    OpDef("Print", [OpInput("input", "tensor"), OpInput("data", "tensor_list")],
          [OpOutput("output", "tensor")],
          [OpAttr("T", "type"), OpAttr("U", "list(type)"),
           OpAttr("message", "string", ""), OpAttr("first_n", "int", "-1"),
           OpAttr("summarize", "int", "3")],
          "Print tensor values for debugging", "control"),
    
    OpDef("Assert", [OpInput("condition", "tensor"), OpInput("data", "tensor_list")],
          [], [OpAttr("T", "list(type)"), OpAttr("summarize", "int", "3")],
          "Assert condition is true", "control"),

    # =========================================================================
    # String Operations
    # =========================================================================
    OpDef("StringJoin", [OpInput("inputs", "tensor_list")],
          [OpOutput("output", "tensor")],
          [OpAttr("N", "int"), OpAttr("separator", "string", "")],
          "Join strings", "string"),
    
    OpDef("StringSplit", [OpInput("input", "tensor"), OpInput("delimiter", "tensor")],
          [OpOutput("indices", "tensor"), OpOutput("values", "tensor"),
           OpOutput("shape", "tensor")],
          [OpAttr("skip_empty", "bool", "true")],
          "Split strings", "string"),
    
    OpDef("RegexReplace", [OpInput("input", "tensor"), OpInput("pattern", "tensor"),
                          OpInput("rewrite", "tensor")],
          [OpOutput("output", "tensor")],
          [OpAttr("replace_global", "bool", "true")],
          "Replace regex pattern", "string"),

    # =========================================================================
    # File I/O Operations
    # =========================================================================
    OpDef("ReadFile", [OpInput("filename", "tensor")],
          [OpOutput("contents", "tensor")], [],
          "Read entire file contents", "io"),
    
    OpDef("WriteFile", [OpInput("filename", "tensor"), OpInput("contents", "tensor")],
          [], [], "Write contents to file", "io"),
    
    OpDef("MatchingFiles", [OpInput("pattern", "tensor")],
          [OpOutput("filenames", "tensor")], [],
          "Find files matching pattern", "io"),

    # =========================================================================
    # Variable Operations
    # =========================================================================
    OpDef("Variable", [],
          [OpOutput("ref", "tensor")],
          [OpAttr("shape", "shape"), OpAttr("dtype", "type"),
           OpAttr("container", "string", ""), OpAttr("shared_name", "string", "")],
          "Create variable", "variable"),
    
    OpDef("VariableV2", [],
          [OpOutput("ref", "tensor")],
          [OpAttr("shape", "shape"), OpAttr("dtype", "type"),
           OpAttr("container", "string", ""), OpAttr("shared_name", "string", "")],
          "Create variable V2", "variable"),
    
    OpDef("VarHandleOp", [],
          [OpOutput("resource", "tensor")],
          [OpAttr("container", "string", ""), OpAttr("shared_name", "string", ""),
           OpAttr("dtype", "type"), OpAttr("shape", "shape")],
          "Create variable handle", "variable"),
    
    OpDef("ReadVariableOp", [OpInput("resource", "tensor")],
          [OpOutput("value", "tensor")], [OpAttr("dtype", "type")],
          "Read variable value", "variable"),
    
    OpDef("AssignVariableOp", [OpInput("resource", "tensor"), OpInput("value", "tensor")],
          [], [OpAttr("dtype", "type")],
          "Assign value to variable", "variable"),
    
    OpDef("AssignAddVariableOp", [OpInput("resource", "tensor"), OpInput("value", "tensor")],
          [], [OpAttr("dtype", "type")],
          "Add value to variable", "variable"),
    
    OpDef("AssignSubVariableOp", [OpInput("resource", "tensor"), OpInput("value", "tensor")],
          [], [OpAttr("dtype", "type")],
          "Subtract value from variable", "variable"),
]

# Total ops count
TOTAL_OPS = len(BUILTIN_OPS)

# =============================================================================
# Code Generator
# =============================================================================

def generate_header(ops: List[OpDef]) -> str:
    """Generate the C++ header file for all ops."""
    
    header = f'''// ============================================================================
// tf_wrap/ops.hpp - Auto-generated TensorFlow Operations Wrapper
// Generated: {datetime.datetime.now().isoformat()}
// Total operations: {len(ops)}
// ============================================================================
//
// This file provides type-safe C++20 wrappers for TensorFlow operations.
// Each op is wrapped as a function that creates the operation in a graph.
//
// Usage:
//   using namespace tf_wrap::ops;
//   
//   FastGraph graph;
//   auto t1 = FastTensor::FromScalar<float>(1.0f);
//   auto t2 = FastTensor::FromScalar<float>(2.0f);
//   
//   auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
//   auto c2 = Const(graph, "c2", t2.handle(), TF_FLOAT);
//   auto sum = Add(graph, "sum", c1, c2, TF_FLOAT);
//
// ============================================================================

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <stdexcept>
#include <cstdint>

extern "C" {{
#include <tensorflow/c/c_api.h>
}}

#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/status.hpp"

namespace tf_wrap {{
namespace ops {{

// ============================================================================
// Op Result - Wrapper for operation outputs
// ============================================================================

/// Result of an operation - holds the TF_Operation* and provides output access
class OpResult {{
public:
    explicit OpResult(TF_Operation* op) : op_(op) {{
        if (!op_) throw std::runtime_error("OpResult: null operation");
    }}
    
    /// Get the underlying operation
    [[nodiscard]] TF_Operation* op() const noexcept {{ return op_; }}
    
    /// Get output at index (default 0)
    [[nodiscard]] TF_Output output(int index = 0) const noexcept {{
        return TF_Output{{op_, index}};
    }}
    
    /// Implicit conversion to TF_Output (for output 0)
    operator TF_Output() const noexcept {{ return output(0); }}
    
    /// Get number of outputs
    [[nodiscard]] int num_outputs() const noexcept {{
        return TF_OperationNumOutputs(op_);
    }}
    
    /// Get operation name
    [[nodiscard]] std::string name() const {{
        return TF_OperationName(op_);
    }}

private:
    TF_Operation* op_;
}};

'''
    
    # Group ops by category
    categories: Dict[str, List[OpDef]] = {}
    for op in ops:
        if op.category not in categories:
            categories[op.category] = []
        categories[op.category].append(op)
    
    category_order = ["math", "matrix", "reduce", "compare", "nn", "array", 
                      "cast", "random", "image", "control", "string", "io", 
                      "variable", "misc"]
    
    category_names = {
        "math": "Math Operations",
        "matrix": "Matrix Operations", 
        "reduce": "Reduction Operations",
        "compare": "Comparison Operations",
        "nn": "Neural Network Operations",
        "array": "Array Operations",
        "cast": "Cast and Type Operations",
        "random": "Random Operations",
        "image": "Image Operations",
        "control": "Control Flow Operations",
        "string": "String Operations",
        "io": "File I/O Operations",
        "variable": "Variable Operations",
        "misc": "Miscellaneous Operations"
    }
    
    for cat in category_order:
        if cat not in categories:
            continue
        
        header += f'''
// ============================================================================
// {category_names.get(cat, cat.title())}
// ============================================================================

'''
        for op in categories[cat]:
            header += generate_op_function(op)
    
    header += '''
} // namespace ops
} // namespace tf_wrap
'''
    
    return header


def generate_op_function(op: OpDef) -> str:
    """Generate a single op function."""
    
    # Build parameter list
    params = ["Graph<GraphPolicy>& graph", "std::string_view name"]
    
    # Add input parameters  
    for inp in op.inputs:
        if inp.dtype == "tensor":
            params.append(f"TF_Output {inp.name}")
        elif inp.dtype == "tensor_list":
            params.append(f"std::span<const TF_Output> {inp.name}")
    
    # Collect required attrs (no default) and optional attrs (with default)
    required_attrs = []
    optional_attrs = []
    
    for attr in op.attrs:
        if attr.default is None:
            required_attrs.append(attr)
        else:
            optional_attrs.append(attr)
    
    # Add required attribute parameters
    for attr in required_attrs:
        if attr.dtype == "type":
            params.append(f"TF_DataType {attr.name}")
        elif attr.dtype == "int":
            params.append(f"int64_t {attr.name}")
        elif attr.dtype == "float":
            params.append(f"float {attr.name}")
        elif attr.dtype == "bool":
            params.append(f"bool {attr.name}")
        elif attr.dtype == "string":
            params.append(f'std::string_view {attr.name}')
        elif attr.dtype == "list(int)":
            params.append(f"std::span<const int64_t> {attr.name}")
        elif attr.dtype == "tensor":
            params.append(f"TF_Tensor* {attr.name}")
        elif attr.dtype == "shape":
            params.append(f"std::span<const int64_t> {attr.name}")
    
    # Build the fluent chain
    chain_parts = [f'graph.NewOperation("{op.name}", std::string(name))']
    
    # Add inputs
    for inp in op.inputs:
        if inp.dtype == "tensor":
            chain_parts.append(f'.AddInput({inp.name})')
        elif inp.dtype == "tensor_list":
            chain_parts.append(f'.AddInputList({inp.name})')
    
    # Add required attributes only
    for attr in required_attrs:
        if attr.dtype == "type":
            chain_parts.append(f'.SetAttrType("{attr.name}", {attr.name})')
        elif attr.dtype == "int":
            chain_parts.append(f'.SetAttrInt("{attr.name}", {attr.name})')
        elif attr.dtype == "float":
            chain_parts.append(f'.SetAttrFloat("{attr.name}", {attr.name})')
        elif attr.dtype == "bool":
            chain_parts.append(f'.SetAttrBool("{attr.name}", {attr.name})')
        elif attr.dtype == "string":
            chain_parts.append(f'.SetAttrString("{attr.name}", {attr.name})')
        elif attr.dtype == "list(int)":
            chain_parts.append(f'.SetAttrIntList("{attr.name}", {attr.name})')
        elif attr.dtype == "tensor":
            chain_parts.append(f'.SetAttrTensor("{attr.name}", {attr.name})')
        elif attr.dtype == "shape":
            chain_parts.append(f'.SetAttrShape("{attr.name}", {attr.name})')
    
    chain_parts.append('.Finish()')
    
    # Build function body with proper indentation
    if len(chain_parts) <= 3:
        body = f'    return OpResult({chain_parts[0]}{"".join(chain_parts[1:])});'
    else:
        body = f'    return OpResult(\n        {chain_parts[0]}\n'
        for part in chain_parts[1:]:
            body += f'        {part}\n'
        body = body.rstrip('\n') + ');'
    
    params_str = ',\n    '.join(params)
    
    return f'''/// {op.description}
template<typename GraphPolicy>
[[nodiscard]] inline OpResult {op.name}(
    {params_str}) {{
{body}
}}

'''


def try_extract_from_tf() -> Optional[List[OpDef]]:
    """Try to extract ops from TensorFlow (requires TF installed)."""
    try:
        import tensorflow as tf
        from tensorflow.python.framework.op_def_registry import get_registered_ops
        
        registered_ops = get_registered_ops()
        ops = []
        
        for name, op_def in sorted(registered_ops.items()):
            inputs = []
            for arg in op_def.input_arg:
                dtype = "tensor_list" if arg.number_attr or arg.type_list_attr else "tensor"
                inputs.append(OpInput(arg.name, dtype, arg.description))
            
            outputs = []
            for arg in op_def.output_arg:
                dtype = "tensor_list" if arg.number_attr or arg.type_list_attr else "tensor"
                outputs.append(OpOutput(arg.name, dtype, arg.description))
            
            attrs = []
            for attr in op_def.attr:
                dtype = attr.type
                default = str(attr.default_value) if attr.HasField('default_value') else None
                attrs.append(OpAttr(attr.name, dtype, default, attr.description))
            
            ops.append(OpDef(name, inputs, outputs, attrs, op_def.summary))
        
        return ops
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate TensorFlow ops C++ wrapper")
    parser.add_argument("--from-tf", action="store_true", 
                        help="Extract ops from TensorFlow (requires TF installed)")
    parser.add_argument("--builtin", action="store_true",
                        help="Use built-in op definitions (default)")
    parser.add_argument("--output", "-o", default="ops.hpp",
                        help="Output file path")
    parser.add_argument("--list", action="store_true",
                        help="List available ops and exit")
    
    args = parser.parse_args()
    
    # Determine which ops to use
    ops = None
    
    if args.from_tf:
        ops = try_extract_from_tf()
        if ops is None:
            print("Warning: Could not import TensorFlow. Falling back to built-in ops.",
                  file=sys.stderr)
    
    if ops is None:
        ops = BUILTIN_OPS
    
    if args.list:
        categories: Dict[str, List[str]] = {}
        for op in ops:
            cat = getattr(op, 'category', 'misc')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(op.name)
        
        print(f"Total operations: {len(ops)}\n")
        for cat, names in sorted(categories.items()):
            print(f"{cat.upper()} ({len(names)} ops):")
            for name in sorted(names):
                print(f"  - {name}")
            print()
        return
    
    # Generate header
    header = generate_header(ops)
    
    # Write to file
    with open(args.output, 'w') as f:
        f.write(header)
    
    print(f"Generated {args.output} with {len(ops)} operations")


if __name__ == "__main__":
    main()
