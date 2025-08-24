//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_CONVERSION_COMMON_H
#define TRITON_ADAPTER_CONVERSION_COMMON_H



namespace mlir {
namespace triton {


enum TensorKind { NONE = -1, INPUT = 0, OUTPUT = 1, INPUT_OUTPUT = 2 };

} // namespace triton
} // namespace mlir

#endif