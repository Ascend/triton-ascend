//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
//#include "AnalysisStructured/PtrAnalysis.h"
#include "Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "Conversion/TritonArithToLinalg/ArgMinMaxConverter.h"
#include "Conversion/TritonArithToLinalg/FunctionConverter.h"
#include "Conversion/TritonArithToLinalg/TritonOpConverter.h"
#include "Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <type_traits>

//#define DEBUG_TYPE "triton-arith-to-linalg"
#include "Conversion/TritonArithToLinalg/ConversionPatterns.hpp"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "Conversion/TritonArithToLinalg/Passes.h.inc"


void mlir::triton::populateTritonArithToLinalgCanonicalizationPatterns(RewritePatternSet &patterns)
{
    patterns.add<ArithOpConverters::AssertCanonicalizer>(patterns.getContext());
    patterns.add<ArithOpConverters::BitcastCanonicalizer>(patterns.getContext());
    patterns.add<
        ArithOpConverters::ScalarMathCanonicalizer<math::AbsFOp>,
        ArithOpConverters::ScalarMathCanonicalizer<math::CeilOp>, ArithOpConverters::ScalarMathCanonicalizer<math::CosOp>,
        ArithOpConverters::ScalarMathCanonicalizer<math::ErfOp>, ArithOpConverters::ScalarMathCanonicalizer<math::ExpOp>,
        ArithOpConverters::ScalarMathCanonicalizer<math::Exp2Op>,
        ArithOpConverters::ScalarMathCanonicalizer<math::FloorOp>,
        ArithOpConverters::ScalarMathCanonicalizer<math::LogOp>,
        ArithOpConverters::ScalarMathCanonicalizer<math::Log2Op>,
        ArithOpConverters::ScalarMathCanonicalizer<math::RsqrtOp>, ArithOpConverters::ScalarMathCanonicalizer<math::SinOp>,
        ArithOpConverters::ScalarMathCanonicalizer<math::SqrtOp>,
        ArithOpConverters::ScalarMathCanonicalizer<math::TanhOp>,
        ArithOpConverters::ScalarMathCanonicalizer<arith::AddFOp>, ArithOpConverters::ScalarMathCanonicalizer<arith::SubFOp>,
        ArithOpConverters::ScalarMathCanonicalizer<arith::MulFOp>, ArithOpConverters::ScalarMathCanonicalizer<arith::DivFOp>,
        ArithOpConverters::ScalarMathCanonicalizer<arith::NegFOp>, ArithOpConverters::ScalarMathCanonicalizer<arith::RemFOp>,
        ArithOpConverters::ScalarMathCanonicalizer<arith::MaxNumFOp>,
        ArithOpConverters::ScalarMathCanonicalizer<arith::MaximumFOp>,
        ArithOpConverters::ScalarMathCanonicalizer<arith::MinNumFOp>,
        ArithOpConverters::ScalarMathCanonicalizer<arith::MinimumFOp>
        >(patterns.getContext());
    patterns.add<ArithOpConverters::ReduceSingleCanonicalizer>(patterns.getContext());
}

void mlir::triton::populateTritonArithToLinalgConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank) {
  populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
      patterns, typeConverter);

  patterns.add<AddPtrConverter>(patterns.getContext());
  patterns.add<ArithFunctionConverter::GetProgramIDConverter>(patterns.getContext());
  patterns.add<ArithFunctionConverter::GetNumProgramsConverter>(
      patterns.getContext());

  patterns.add<ArithOpConverters::MakeRangeConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::SplatConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::ClampFConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::PreciseDivConverter>(patterns.getContext());
  // reduce converters
  patterns.add<ArithOpConverters::ArgMinConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::ArgMaxConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::ReduceConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::ScanConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::ReshapeConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::ExpandDimsConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::BroadcastConverter>(patterns.getContext());

  patterns.add<ArithOpConverters::DenseConstantConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::ExternElementwiseClOpConverter>(
      patterns.getContext());
  patterns.add<ArithOpConverters::TritonMulhiuiConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::TritonPreciseSqrtConverter>(
      patterns.getContext());
  patterns.add<MakeTensorPtrConverter>(patterns.getContext());
  patterns.add<AdvanceConverter>(patterns.getContext());

  patterns.add<ArithOpConverters::TransposeConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::SplitConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::JoinConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::CatConverter>(patterns.getContext());
  // fixme 
  // patterns.add<BitcastConverter>(patterns.getContext());
 
  patterns.add<YieldConverter>(patterns.getContext());
  patterns.add<LoopConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::GatherConverter>(patterns.getContext());

  patterns.add<ArithOpConverters::DevicePrintConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::MatmulConverter>(patterns.getContext());

  patterns.add<ArithOpConverters::DeviceAssertConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::DevicePrintConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::MatmulConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::SortOpConverter>(patterns.getContext());
  patterns.add<ArithOpConverters::DotScaledConverter>(patterns.getContext());

  // if (!this->namedOps) {
  //   linalg::populateElementwiseToLinalgConversionPatterns(patterns);
  // }
}

