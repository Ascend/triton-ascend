//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_CONVERSION_TRITONARITHTOLINALG_TRITONARITHTOLINALG_H
#define TRITON_ADAPTER_CONVERSION_TRITONARITHTOLINALG_TRITONARITHTOLINALG_H


#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

// #include "Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-arith-to-linalg"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONARITHTOLINALG
#include "Conversion/TritonArithToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

#include "Conversion/ConversionCommon.h"

namespace {
using namespace mlir;
using namespace triton;
const std::string globalKernelAttr = "global_kernel";
const std::string kernelMixModeName = "mix_mode";
const unsigned INT_BIT_WIDTH = 32;
const unsigned SET_INIT_SIZE = 16;

class TritonArithTypeConverter : public mlir::TypeConverter {
public:
  explicit TritonArithTypeConverter();
};

class TritonArithToLinalgPass
    : public triton::impl::TritonArithToLinalgBase<TritonArithToLinalgPass> {
  using TritonArithToLinalgBase<
      TritonArithToLinalgPass>::TritonArithToLinalgBase;

// class TritonArithToLinalgPass : public TritonArithToLinalgBase<TritonArithToLinalgPass> {
//   using TritonArithToLinalgBase<TritonArithToLinalgPass>::TritonArithToLinalgBase;

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

private:
  // grid构造 num_programs 3维, program_id 3维
  // remember 'xxxOp' is usually a Pointer, so that we can change target memory
  // without giving a reference argument
  void addProgramInfo(triton::FuncOp func, bool globalKernel=false);

  // 处理嵌套的if/else
  void transformNestedIfElse(Operation &nestedBranch, OpBuilder &builder);

  void convertTTFunc(triton::FuncOp func, const bool existDot);
  // // 处理嵌套的if/else

  void addDynamicLegal(ConversionTarget &target,
                        TritonArithTypeConverter &tritonTypeConverter);


  void
  populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns);

  void populateTritonToLinalgConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                unsigned int launchGridRank);

public:
  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;
};
} // namespace

namespace mlir {
namespace triton {

}
}

#endif // TRITON_ADAPTER_CONVERSION_TRITONARITHTOLINALG_H
