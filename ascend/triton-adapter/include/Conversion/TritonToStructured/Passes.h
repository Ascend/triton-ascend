#ifndef TRITON_TO_STRUCTURED_CONVERSION_PASSES_H
#define TRITON_TO_STRUCTURED_CONVERSION_PASSES_H

#include "Conversion/TritonToStructured/TritonToStructured.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "Conversion/TritonToStructured/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
