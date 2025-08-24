//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonArithToLinalg/TritonOpConverter.h"
// #include "TritonToLinalg/TritonToLinalgPass.h"
// #include "TritonToLinalg/BlockPtrAnalysis.h"
#include "Analysis/MaskAnalysis.h"

#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
extern int nd2nzFlag;
namespace ArithOpConverters {
using namespace mlir;
using namespace triton;

MetaUseEraser::MetaUseEraser(MLIRContext *context)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

LogicalResult MetaUseEraser::matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
  if (isa<triton::AddPtrOp>(op)) {
    return rewriter.notifyMatchFailure(op,
                                       "AddPtrOp will be handled separately");
  }
  if (isMetaUse(op)) {
    rewriter.eraseOp(op);
    return success();
  }
  return rewriter.notifyMatchFailure(op, "requires meta ops");
}

LogicalResult
AssertCanonicalizer::matchAndRewrite(triton::AssertOp op,
                                     PatternRewriter &rewriter) const {
  // TODO: update assert converter to support llvm20
  LLVM_DEBUG(llvm::dbgs()
             << "we do not support assertion in kernel in llvm-20 yet \n");
  rewriter.eraseOp(op);
  return success();
}

LogicalResult
BitcastConverter::matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto arithBitcast = rewriter.create<arith::BitcastOp>(
      op.getLoc(), op.getType(), op.getOperand());
  
  rewriter.replaceOp(op, arithBitcast.getResult());
  return success();
}

LogicalResult
TransposeConverter::matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto src = adaptor.getSrc();
  auto srcRank = cast<ShapedType>(src.getType()).getRank();
  auto res = ConverterUtils::getTransposedValue(src, op.getLoc(), rewriter,
                                                op.getOrder());
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult
YieldConverter::matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
  return success();
}

LogicalResult
LoopConverter::matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, BlockData> known;

  BlockDataParser::rewriteForOp(op, rewriter, known);
  return success();
}

LogicalResult
AdvanceConverter::matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, BlockData> known;
  BlockDataParser::rewriteAdvanceOp(op, rewriter, known);
  return success();
}

OpFoldResult MakeTensorPtrConverter::accumulatePotentialOffsetOnBase(
    triton::MakeTensorPtrOp op, Value base, OpFoldResult offset,
    ConversionPatternRewriter &rewriter) const {
  if (auto baseRecast = base.getDefiningOp<memref::ReinterpretCastOp>()) {
    assert(isa<triton::AddPtrOp>(op.getBase().getDefiningOp()) &&
           "base of MakeTensorPtrOp only comes from native ptr or AddPtrOp");

    return addOpFoldResult(offset, baseRecast.getConstifiedMixedOffset(),
                           op.getLoc(), rewriter);
  }

  return offset;
}

// Design for load/store boundary_check.
memref::ReinterpretCastOp
MakeTensorPtrConverter::createRedundantOp(triton::MakeTensorPtrOp op,
                                          ConversionPatternRewriter &rewriter,
                                          BlockData &data) const {
  auto loc = op.getLoc();
  // to do boundary_check in tt.load, we need to keep the parent tensor's
  // shape info in the IR.
  // use the parent tensor's shape to create a cast
  auto resultSizes = data.getSizes();
  auto resultOffsets = data.getOffsets();
  data.getSizesRef().clear();
  data.getOffsetsRef().clear();
  data.getSizesRef() =
      std::move(llvm::map_to_vector(op.getShape(), [&](Value v) {
        return getOpFoldResultOfLayoutInfo(v, rewriter);
      }));

  // This redundant ReinterpretCastOp is to describe full tensor_ptr, so each
  // dim offset from base is initialized as zero.
  SmallVector<OpFoldResult> curOffsets(op.getOffsets().size(),
                                       rewriter.getIndexAttr(0));
  // Just accumulate base potential offset
  curOffsets.front() = accumulatePotentialOffsetOnBase(
      op, rewriter.getRemappedValue(op.getBase()), curOffsets.front(),
      rewriter);

  for (auto offset : curOffsets) {
    data.getOffsetsRef().push_back(offset);
  }

  SmallVector<int64_t> staticShapes;
  SmallVector<Value> dynamicShapes;
  dispatchIndexOpFoldResults(data.getSizesRef(), dynamicShapes, staticShapes);
  auto castOp = data.createCastOp(staticShapes, loc, rewriter);
  // restore sizes and offsets
  data.getSizesRef().clear();
  for (auto &s : resultSizes) {
    data.getSizesRef().push_back(s);
  }
  data.getOffsetsRef().clear();
  for (auto &offset : resultOffsets) {
    data.getOffsetsRef().push_back(offset);
  }
  return castOp;
}

// ToDo:
// 1. Refactor MakeTensorPtrConverter and AdvanceConverter with
// memref::ReinterpretCastOp and memref::SubViewOp.
// Use recast to describe full shape of tensor, and use subview to represent
// current block tensor.
LogicalResult MakeTensorPtrConverter::matchAndRewrite(
    triton::MakeTensorPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  BlockData data;

  auto orderSize = op.getOrder().size();
  if (orderSize > 1) {
    // Declaration of llvm::ArrayRef::slice(n, m)
    // - Chop off the first N elements of the array, and keep M elements
    //   in the array.
    // Take care that 'm' means chunk length
    for (auto [first, second] :
         llvm::zip(op.getOrder().slice(0, orderSize - 1),
                   op.getOrder().slice(1, orderSize - 1))) {
      if (first != second + 1) {
        op->emitError("Currently only support default order on block pointers");
        return failure();
      }
    }
  }

  // Handle base is defined by tt.bitcast
  llvm::SmallDenseMap<Value, BlockData> known;
  BlockDataParser::parse(op.getBase(), data, loc, rewriter, known);
  if (data.hasResElemTy()) {
    auto memrefType = dyn_cast<BaseMemRefType>(data.getSourceRef().getType())
                          .cloneWith(std::nullopt, data.getResElemTyRef());
    UnrealizedConversionCastOp castOp =
        rewriter.create<mlir::UnrealizedConversionCastOp>(loc, memrefType,
                                                          data.getSourceRef());
    data.setSource(castOp.getOutputs()[0]);
  } else {
    data.setSource(rewriter.getRemappedValue(op.getBase()));
  }

  data.getOffsetsRef() =
      std::move(llvm::map_to_vector(op.getOffsets(), [&](Value v) {
        return getOpFoldResultOfLayoutInfo(v, rewriter);
      }));
  data.getStridesRef() =
      std::move(llvm::map_to_vector(op.getStrides(), [&](Value v) {
        return getOpFoldResultOfLayoutInfo(v, rewriter);
      }));

  SmallVector<OpFoldResult> newOffsets;
  for (auto [offset, stride] :
       llvm::zip(data.getOffsetsRef(), data.getStridesRef()))
    newOffsets.push_back(mulOpFoldResult(offset, stride, loc, rewriter));

  // 1. Consider that current base ptr may comes from `triton::AddPtrOp`,
  // which have been converted to `memref::ReinterpretCastOp` with 1D
  // shape([1,]) by `AddPtrConverter`.
  // 2. While here would also convert `triton::MakeTensorPtrOp` to
  // `memref::ReinterpretCastOp`, it will create use-def on double recast
  // which means offset&size&stride info of first one will be dropped in terms
  // of memref recast op fold specification.
  //
  // Conclusion with above two:
  // Base of MakeTensorPtrOp has been seen as origin base, so it should
  // reserve offset of first recast if it exists.
  // Here extract the offset of first recast and add it to highest dimension
  newOffsets.front() = accumulatePotentialOffsetOnBase(
      op, adaptor.getBase(), newOffsets.front(), rewriter);

  data.getOffsetsRef().clear();

  for (auto offset : newOffsets) {
    data.getOffsetsRef().push_back(offset);
  }

  ArrayRef<int64_t> resultShape;
  auto pointerType = cast<mlir::triton::PointerType>(op.getResult().getType());
  if (auto shapedType = dyn_cast<ShapedType>(pointerType.getPointeeType())) {
    resultShape = shapedType.getShape();
    for (auto dim_size : resultShape) {
      data.getSizesRef().push_back(
          IntegerAttr::get(IntegerType::get(op.getContext(), 64), dim_size));
    }
  } else {
    // scalar pointer, should produce a one dimensional memref
    SmallVector<int64_t> scalarShape(1, 1);
    resultShape = scalarShape;
    assert(data.getRank() == 1);
  }

  // special handling for davinci
  // create redundant reinterpret_cast op for record shape info
  auto redundantOp = createRedundantOp(op, rewriter, data);
  redundantOp->setAttr("tensor_ptr_full_shape", rewriter.getUnitAttr());

  // create reinterpret_cast op for the target block
  data.setSource(redundantOp.getResult());
  auto castOp = data.createCastOp(resultShape, loc, rewriter);
  rewriter.replaceOp(op, castOp.getResult());
  
  if (nd2nzFlag) {
    auto basePtr = castOp.getResult();
    int original_rank = op.getShape().size() + 1;
    std::string shapeStr;
    if (auto memrefType = mlir::dyn_cast<MemRefType>(basePtr.getType())) {
        for (auto dim : memrefType.getShape()) {
            shapeStr += llvm::formatv("_{0}", dim);
        }
    }
    auto funcName = rewriter.getStringAttr(
        llvm::formatv("__hmf_original_shape{0}d{1}", original_rank, shapeStr)
    );
    const int vectorSize = 4;
    SmallVector<Type, vectorSize> srcElemTys;
    for (auto sz : op.getShape()) {
      srcElemTys.push_back(sz.getType());
    }
    srcElemTys.push_back(basePtr.getType());
    Type dstElemTy = rewriter.getNoneType();
    FunctionType hintFuncType =
        FunctionType::get(rewriter.getContext(), srcElemTys, {dstElemTy});

    auto mod = SymbolTable::getNearestSymbolTable(op);
    auto extFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(mod, funcName));
    SmallVector<Value, vectorSize> args;
    for (auto sz : op.getShape()) {
      args.push_back(sz);
    }
    args.push_back(basePtr);
    if (!extFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&mod->getRegion(0).front());
      extFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(),
                                              funcName, hintFuncType);
      extFunc.setPrivate();
      extFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                       UnitAttr::get(rewriter.getContext()));
      rewriter.setInsertionPoint(op);
    }
    rewriter.create<func::CallOp>(loc, funcName, dstElemTy, args);
  }
  return success();
}

LogicalResult PreciseDivConverter::matchAndRewrite(
    triton::PreciseDivFOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value opa = op.getX();
  Value opb = op.getY();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto divOp = rewriter.create<arith::DivFOp>(loc, resType, opa, opb);

  rewriter.replaceOp(op, divOp);
  return success();
}

LogicalResult
SelectCanonicalizer::matchAndRewrite(arith::SelectOp op,
                                     PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // 0. Shortcut for scalars
  auto type = dyn_cast<TensorType>(op.getResult().getType());
  if (!type) {
    // do nothing non-tensor select
    return failure();
  }
  auto mask = op.getCondition();
  if (!isa<ShapedType>(mask.getType())) {
    // do nothing for scalar mask
    return failure();
  }

  // 1. Check for continuous masked loads.
  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  triton::MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);

  if (isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return rewriter.notifyMatchFailure(
        op, "Cannot lower continuous masked selects");
  }

  // 2. Slice out the masked part of true tensor
  auto trueTensor = op.getTrueValue();
  auto trueSlice = mstate.getExtractSlice(trueTensor, loc, rewriter);

  // 3. Insert out the sliced true tensor into false tensor
  auto falseTensor = op.getFalseValue();
  auto result = mstate.getInsertSlice(trueSlice, falseTensor, loc, rewriter);

  rewriter.replaceOp(op, result);
  return success();
}

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly applied
 * on function arguments
 */
LogicalResult
BitcastCanonicalizer::matchAndRewrite(triton::BitcastOp bitcastOp,
                                      PatternRewriter &rewriter) const {
  Value castSrc = bitcastOp.getSrc();
  Value castRes = bitcastOp.getResult();
  Type castSrcTy = castSrc.getType();
  Type castSrcPtrTy = isa<ShapedType>(castSrcTy)
                          ? cast<ShapedType>(castSrcTy).getElementType()
                          : castSrcTy;
  if (!isa<triton::PointerType>(castSrcPtrTy))
    return failure();

  auto origBitwidth = getPointeeBitWidth(castSrc.getType());
  auto castBitwidth = getPointeeBitWidth(castRes.getType());

  if (origBitwidth == 1)
    origBitwidth = 8;
  if (castBitwidth == 1)
    castBitwidth = 8;
  if (origBitwidth != castBitwidth) {
    bitcastOp.emitError() << "Casting pointers with unmatched bitwidth!\n";
    return failure();
  }

  // llvm::dbgs()<<bitcastOp<<"\n";
  Operation *beforeCastOp = castSrc.getDefiningOp();
  if (beforeCastOp == nullptr) {
    return failure();
  }

  auto newRes =
      TypeSwitch<Operation *, FailureOr<Operation *>>(beforeCastOp)
          // before: addptr - bitcast - load/store
          // after: bitcast - addptr - load/store
          .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptrOp) {
            auto newCastOp = rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), castRes.getType(), addptrOp.getPtr());
            return rewriter.create<triton::AddPtrOp>(
                bitcastOp.getLoc(), castRes.getType(), newCastOp.getResult(),
                addptrOp.getOffset());
          })
          .Case<triton::SplatOp>([&](triton::SplatOp splatOp) {
            Type newCastSrcTy =
                cast<RankedTensorType>(castRes.getType()).getElementType();

            Value splatSrc = splatOp.getSrc();
            Type splatSrcTy = splatSrc.getType();
            if (auto splatSrcTensorTy = dyn_cast<RankedTensorType>(splatSrcTy))
              newCastSrcTy =
                  splatSrcTensorTy.cloneWith(std::nullopt, newCastSrcTy);
            auto newCastOp = rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), newCastSrcTy, splatSrc);
            return rewriter.create<triton::SplatOp>(
                bitcastOp.getLoc(), castRes.getType(), newCastOp);
          })
          // before: bitcast - bitcast
          // after(fusion optimization): bitcast
          .Case<triton::BitcastOp>([&](triton::BitcastOp prevCastOp) {
            return rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), castRes.getType(), prevCastOp.getSrc());
          })
          .Default([&](Operation *op) {
            return rewriter.notifyMatchFailure(bitcastOp,
                                               "Unknown bitcast pattern");
          });
  if (succeeded(newRes)) {
    rewriter.replaceOp(bitcastOp, newRes.value());
    if (beforeCastOp->use_empty()) {
      rewriter.eraseOp(beforeCastOp);
    }
    return success();
  }
  return failure();
}

LogicalResult ReduceSingleCanonicalizer::matchAndRewrite(triton::ReduceOp reduceOp, PatternRewriter &rewriter) const
{
    auto srcs = reduceOp.getSrcs();
    bool allSrcSingleElem = true;
    for (auto src : srcs) {
        auto srcType = cast<RankedTensorType>(src.getType());
        auto srcShape = srcType.getShape();
        int64_t numel = 1;
        for (auto s : srcShape) {
            numel *= s;
        }
        if (numel != 1) {
            allSrcSingleElem = false;
            break;
        }
    }

    if (!allSrcSingleElem) {
        return rewriter.notifyMatchFailure(reduceOp, "reduce's srcs are not all with single element");
    }

    auto results = reduceOp.getResult();
    auto loc = reduceOp->getLoc();
    auto zero = rewriter
                    .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                               rewriter.getIntegerAttr(rewriter.getIndexType(), 0))
                    .getResult();
    for (int i = 0; i < srcs.size(); i++) {
        auto src = srcs[i];
        auto srcType = cast<RankedTensorType>(src.getType());
        auto srcRank = srcType.getRank();
        auto res = results[i];
        Value extracted;
        if (srcRank == 1) {
            // vector reduce generates a scalar result
            extracted = rewriter.create<tensor::ExtractOp>(loc, src, zero).getResult();
        } else {
            auto srcShape = srcType.getShape();
            auto resType = cast<RankedTensorType>(res.getType());
            auto resShape = resType.getShape();
            auto collapseReassociationIndicesOptional = getReassociationIndicesForCollapse(srcShape, resShape);
            if (!collapseReassociationIndicesOptional.has_value()) {
                return rewriter.notifyMatchFailure(reduceOp, "Failure with getReassociationIndicesForCollapse call");
            }
            auto collapseReassociationIndices = collapseReassociationIndicesOptional.value();
            extracted = rewriter.create<tensor::CollapseShapeOp>(loc, src, collapseReassociationIndices).getResult();
        }
        res.replaceAllUsesWith(extracted);
    }

    return success();
}

LogicalResult DenseConstantConverter::matchAndRewrite(
    arith::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto denseAttr = cast<DenseElementsAttr>(op.getValue());
  auto loc = op.getLoc();
  auto constSplatOp = arith::ConstantOp::materialize(
      rewriter, denseAttr.getSplatValue<Attribute>(),
      denseAttr.getElementType(), loc);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, cast<RankedTensorType>(op.getResult().getType()).getShape(),
      denseAttr.getElementType());

  rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{constSplatOp},
                                              ValueRange{emptyOp});

  return success();
}

LogicalResult
MakeRangeConverter::matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto type = cast<TensorType>(op.getResult().getType());
  auto shape = type.getShape();
  auto elementType = type.getElementType();
  auto context = op.getContext();

  assert(type.getShape().size() == 1 &&
         isa<IntegerType>(type.getElementType()) &&
         type.getElementType().getIntOrFloatBitWidth() == 32 &&
         "make range can only return 1D int32 tensor");

  SmallVector<AffineMap> indexingMaps{AffineMap::get(
      /* dimCount */ 1, /* symbolCount */ 0,
      {mlir::getAffineDimExpr(0, context)}, context)};

  auto init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);

  auto nestedBody = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                        ValueRange blockArgs) {
    Value index = nestedBuilder.create<linalg::IndexOp>(loc, 0);
    Value res = nestedBuilder.create<arith::IndexCastOp>(
        loc, type.getElementType(), index);
    nestedBuilder.create<linalg::YieldOp>(loc, res);
  };

  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, op->getResultTypes(), /* operands */ ValueRange{}, ValueRange{init},
      indexingMaps, ConverterUtils::getNParallelLoopsAttrs(1), nestedBody);

  rewriter.replaceOp(op, linalgOp->getResults());
  return success();
}

LogicalResult
SplatConverter::matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto init = rewriter.create<tensor::EmptyOp>(loc, op.getType().getShape(),
                                               op.getType().getElementType());
  rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{adaptor.getSrc()},
                                              ValueRange{init});
  return success();
}

LogicalResult
ReshapeConverter::matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto src = op.getSrc();
  auto dst = op.getResult();
  Value shape = rewriter.create<arith::ConstantOp>(
      loc,
      rewriter.getI64TensorAttr(cast<ShapedType>(dst.getType()).getShape()));
  auto reshapeOp =
      rewriter.create<tensor::ReshapeOp>(loc, dst.getType(), src, shape);
  rewriter.replaceOp(op, reshapeOp.getResult());
  return success();
}

LogicalResult ExpandDimsConverter::matchAndRewrite(
    triton::ExpandDimsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto src = op.getSrc();
  auto resShape = cast<ShapedType>(op.getResult().getType()).getShape();
  auto axis = op.getAxis();

  SmallVector<ReassociationIndices> reassociation;

  auto src_last_dim = resShape.size() - 2;
  auto map_func = [&](unsigned i) -> ReassociationIndices {
    if (i < axis) {
      return i == src_last_dim ? ReassociationIndices{i, i + 1}
                               : ReassociationIndices{i};
    }
    return i == axis ? ReassociationIndices{i, i + 1}
                     : ReassociationIndices{i + 1};
  };

  reassociation = llvm::to_vector(
      llvm::map_range(llvm::seq<unsigned>(0, src_last_dim + 1), map_func));

  auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      op.getLoc(), op.getResult().getType(), src, reassociation);
  rewriter.replaceOp(op, expandShapeOp.getResult());
  return success();
}

LogicalResult
ClampFConverter::matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto input = adaptor.getX();
  auto min_para = adaptor.getMin();
  auto max_para = adaptor.getMax();
  auto propagateNan_para = adaptor.getPropagateNan();

  if (auto input_type = dyn_cast<RankedTensorType>(input.getType())) {
    if (isa<FloatType>(min_para.getType())) {
      auto minEmptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, input_type.getShape(), input_type.getElementType());
      min_para = rewriter
                     .create<linalg::FillOp>(loc, ValueRange{min_para},
                                             ValueRange{minEmptyTensor})
                     .result();
    }
    if (isa<FloatType>(max_para.getType())) {
      auto maxEmptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, input_type.getShape(), input_type.getElementType());
      max_para = rewriter
                     .create<linalg::FillOp>(loc, ValueRange{max_para},
                                             ValueRange{maxEmptyTensor})
                     .result();
    }
  }

  if (propagateNan_para == PropagateNan::NONE) {
    auto minOp = rewriter.create<arith::MinNumFOp>(loc, input, max_para);
    auto maxOp = rewriter.create<arith::MaxNumFOp>(loc, min_para, minOp);
    rewriter.replaceOp(op, ValueRange{maxOp});
  } else if (propagateNan_para == PropagateNan::ALL) {
    auto minOp = rewriter.create<arith::MinimumFOp>(loc, input, max_para);
    auto maxOp = rewriter.create<arith::MaximumFOp>(loc, min_para, minOp);
    rewriter.replaceOp(op, ValueRange{maxOp});
  } else {
    return failure();
  }

  return success();
}

// Here convert tt.broadcast to linalg.broadcast
//
// before
// %out = tt.broadcast %in : tensor<1x4x8xf32> -> tensor<128x4x8xf32>
//
// after
// %collpased = tensor.collapse_shape %in [[0, 1], [2]] :
//                                    tensor<1x4x8xf32> into tensor<4x8xf32>
// %out = linalg.broadcast ins(%collpased : tensor<4x8xf32>)
//                         outs(%empty : tensor<128x4x8xf32>) dimensions = [0]
LogicalResult
BroadcastConverter::matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  assert(op->getNumResults() == 1 && "BroadcastOp assumes single result");

  RankedTensorType sourceType =
      cast<RankedTensorType>(adaptor.getSrc().getType());
  RankedTensorType resultType = cast<RankedTensorType>(op.getType());
  auto elementType = resultType.getElementType();
  size_t resultRank = resultType.getRank();
  auto loc = op.getLoc();

  auto initEmpty =
      rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elementType);

  SmallVector<int64_t> broadcastDims =
      ConverterUtils::getBroadcastDims(sourceType, resultType);
  SmallVector<int64_t> unbroadcastDims =
      ConverterUtils::getUnbroadcastDims(sourceType, resultType);

  SmallVector<ReassociationIndices> collapseReassociationIndices;
  auto collapseReassociationIndicesOptional =
      getReassociationIndicesForCollapse(sourceType.getShape(),
                                         unbroadcastDims);
  if (!collapseReassociationIndicesOptional.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "Failure with getReassociationIndicesForCollapse call");
  }
  collapseReassociationIndices = collapseReassociationIndicesOptional.value();

  RankedTensorType collapseResultType =
      RankedTensorType::get(unbroadcastDims, sourceType.getElementType());

  auto collpasedOp = rewriter.create<tensor::CollapseShapeOp>(
      loc, collapseResultType, adaptor.getSrc(), collapseReassociationIndices);

  auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
      loc, collpasedOp, initEmpty,
      rewriter.getDenseI64ArrayAttr(broadcastDims));

  rewriter.replaceOp(op, broadcastOp.getResults());
  return success();
}

// Reduce Converter
bool ReduceConverter::isReductionOpSupported(Operation *redOp) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MaximumFOp,
          arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
          arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
          arith::AndIOp, arith::OrIOp, arith::XOrIOp>(redOp);
}

LogicalResult ReduceConverter::convertToTargetOp(
    triton::ReduceOp op, typename triton::ReduceOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto source = adaptor.getOperands().front();
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto elemType = sourceType.getElementType();
  auto resType = op.getResult().front().getType();
  auto loc = op.getLoc();
  auto reductionOps = this->getRedOps(op);

  // Reduction of arbitrary operations isn't supported because using the first
  // element across the reduction dimension requires us to iterate over a
  // subview that skips over each first element.
  if (!this->isReductionOpSupported(reductionOps.front())) {
    return rewriter.notifyMatchFailure(
        op, "Only support lowering reduction with single op and limited types of reducetion");
  }

  auto rop = reductionOps.front();
  auto axis = op.getAxis();
  auto isVectorReduce = sourceType.getRank() == 1;

  auto constantType = elemType;

  auto accBaseConstOp = this->getRedBaseConstOp(rewriter, rop, constantType);
  Value initTensor;

  if (isVectorReduce) {
    auto holder = rewriter.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get({}, constantType), ValueRange{});
    initTensor = rewriter
                     .create<linalg::FillOp>(loc, accBaseConstOp.getResult(),
                                             holder.getResult())
                     .getResult(0);
  } else {
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(resType).getShape(), constantType);
    initTensor =
        rewriter.create<linalg::FillOp>(loc, accBaseConstOp.getResult(), init)
            .getResult(0);
  }

  Value finalResult = rewriter.create<linalg::ReduceOp>(
              loc, ValueRange{source}, ValueRange{initTensor},
              SmallVector<int64_t>{axis},
              [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                assert(inputs.size() == 2);
                Value result = this->getRedElement(inputs[0], inputs[1], loc, rop,
                                             opBuilder, false);
                opBuilder.create<linalg::YieldOp>(loc, result);
              })
          .getResult(0);

  if (sourceType.getRank() == 1) {
    finalResult = rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
  }

  rewriter.replaceOp(op, finalResult);
  return success();
}

LogicalResult ReduceConverter::convertToTargetOpExtended(
    triton::ReduceOp op, typename triton::ReduceOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto elemTypes = op.getElementTypes();

  auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
  const auto isScalarReduce = valueResultType == nullptr;

  SmallVector<Value> outputs;
  for (auto i = 0; i < op.getResult().size() && i < elemTypes.size(); i++) {
    auto result = dyn_cast<RankedTensorType>(op.getType(i));
    SmallVector<int64_t> resultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(result.getShape())};
    outputs.push_back(
        rewriter.create<tensor::EmptyOp>(loc, resultShape, elemTypes[i]));
  }

  auto linalgOp = rewriter.create<linalg::ReduceOp>(
      loc, adaptor.getOperands(), outputs,
      SmallVector<int64_t>{adaptor.getAxis()},
      [&](OpBuilder &b, Location loc, ValueRange inputs) {
        auto tritonReduceBlock = op.getBody();
        IRMapping mapping;
        mapping.map(tritonReduceBlock->getArguments(), inputs);

        for (auto &op : tritonReduceBlock->without_terminator()) {
          b.clone(op, mapping);
        }

        auto tritonYield = tritonReduceBlock->getTerminator();
        auto results =
            llvm::map_to_vector(tritonYield->getOperands(),
                                [&](Value val) { return mapping.lookup(val); });
        b.create<linalg::YieldOp>(loc, results);
      });

  if (failed(addReduceWithIndexAttrIfNeeded(rewriter, linalgOp))) {
    return rewriter.notifyMatchFailure(op, "meaningless reduce operation");
  }

  if (isScalarReduce) {
    SmallVector<Value> reduceResults;
    for (auto i = 0; i < linalgOp.getResults().size() && i < elemTypes.size();
         i++) {
      reduceResults.push_back(rewriter.create<tensor::ExtractOp>(
          loc, elemTypes[i], linalgOp.getResults()[i], ValueRange{}));
    }
    rewriter.replaceOp(op, reduceResults);
  } else {
    rewriter.replaceOp(op, linalgOp);
  }
  return success();
}

bool ScanConverter::isReductionOpSupported(Operation *redOp) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp>(redOp);
}

LogicalResult ScanConverter::convertToTargetOp(
    triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto reductionOps = this->getRedOps(op);
  // Reduction of arbitrary operations isn't supported because using the first
  // element across the reduction dimension requires us to iterate over a
  // subview that skips over each first element.
  if (!this->isReductionOpSupported(reductionOps.front())) {
    return rewriter.notifyMatchFailure(
        op, "Only support lowering reduction with single op and limited types of reducetion");
  }

  llvm::SmallString<64> funcName;
  auto rop = reductionOps.front();
  if (isa<arith::AddFOp, arith::AddIOp>(rop)) {
    funcName = "triton_cumsum";
  } else if (isa<arith::MulFOp, arith::MulIOp>(rop)) {
    funcName = "triton_cumprod";
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName += "_" + std::to_string(uniqueId++);
  }

  auto loc = op.getLoc();
  auto src = adaptor.getOperands().front();
  auto resTy = op.getResult().front().getType();
  auto libFnType = rewriter.getFunctionType(
      {src.getType(), rewriter.getI32Type(), rewriter.getI1Type()}, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  auto scanAxis = op.getAxis();
  auto scanReverse = op.getReverse();
  Value axis = rewriter.create<arith::ConstantIntOp>(loc, scanAxis, 32);
  Value reverse = rewriter.create<arith::ConstantIntOp>(loc, scanReverse, 1);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}),
                                              ValueRange({src, axis, reverse}));

  rewriter.replaceOp(op, callOp);

  return success();
}

LogicalResult ScanConverter::convertToTargetOpExtended(
    triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  return op->emitError("tt.scan with multiple ops inside the body unsupported!");
}

LogicalResult ExternElementwiseClOpConverter::matchAndRewrite(
    triton::ExternElementwiseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  if (!op.getPure()) {
    op->emitWarning() << "impure elementwise op!";
    return failure();
  }
  if (op.getSymbol().contains("__hmf_")) {
    // 1. get or create the declaration of external elementwise function
    Type dstTy = op.getResult().getType();
    bool isDstScalar = !isa<RankedTensorType>(dstTy);
    Type dstElemTy =
        isDstScalar ? dstTy : cast<RankedTensorType>(dstTy).getElementType();
    SmallVector<Type, 4> srcElemTys;
    SmallVector<Value, 4> srcs;
    for (auto src : op.getSrcs()) {
      if (!isa<RankedTensorType>(src.getType())) {
        src = rewriter.create<tensor::FromElementsOp>(
            op.getLoc(), RankedTensorType::get({(int64_t)1}, src.getType()),
            src);
      }
      srcs.push_back(src);
      srcElemTys.push_back(
          cast<RankedTensorType>(src.getType()).getElementType());
    }
    FunctionType elemFuncType =
        FunctionType::get(rewriter.getContext(), srcElemTys, {dstElemTy});
    auto mod = SymbolTable::getNearestSymbolTable(op);
    auto extFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(mod, op.getSymbol()));
    if (!extFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&mod->getRegion(0).front());
      extFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(),
                                              op.getSymbol(), elemFuncType);
      extFunc.setPrivate();
      extFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                       UnitAttr::get(rewriter.getContext()));
    }
    assert(isa<FunctionOpInterface>(
        SymbolTable::lookupSymbolIn(mod, op.getSymbol())));
    // 2. prepare the output tensor
    Value output;
    if (isDstScalar) {
      dstTy = RankedTensorType::get({(int64_t)1}, dstElemTy);
    }
    bool found = false;
    for (Value v : srcs) {
      if (v.getType() == dstTy) {
        found = true;
        output = v;
        break;
      }
    }
    if (!found) {
      output = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), cast<RankedTensorType>(dstTy).getShape(), dstElemTy);
    }
    // 3. create the linalg.map op
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc,
        /*inputs=*/srcs,
        /*init=*/output,
        /*bodyBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto elemOp = builder.create<func::CallOp>(loc,
                                                     /*name=*/op.getSymbol(),
                                                     /*resultType=*/dstElemTy,
                                                     /*operands=*/regionArgs);
          builder.create<linalg::YieldOp>(loc, elemOp->getResults());
        });
    if (isDstScalar) {
      // need to convert tensor back to scalar
      auto indexType = rewriter.getIndexType();
      Value zeroConstant = rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIntegerAttr(indexType, 0));
      auto extractOp = rewriter.create<tensor::ExtractOp>(
          loc, mapOp.getResults()[0], zeroConstant);
      rewriter.replaceOp(op, extractOp);
    } else {
      rewriter.replaceOp(op, mapOp);
    }
    return success();
  }
  return failure();
}

LogicalResult UnrealizedCastConverter::matchAndRewrite(
    UnrealizedConversionCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult
JoinConverter::matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value opa = op.getLhs();
  Value opb = op.getRhs();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  Value emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                   resType.getElementType());

  auto shape = dyn_cast<RankedTensorType>(opa.getType()).getShape();
  auto sizes = llvm::map_to_vector(shape, [&](int64_t t) {
    return OpFoldResult(rewriter.getI64IntegerAttr(t));
  });
  sizes.push_back(rewriter.getI64IntegerAttr(1));

  int64_t rank = resType.getRank();

  // Set last dimension stride to 2 in layout
  // As last dimension size is always 1, last dimension stride here could be
  // either 1 or 2, while stride `2` could carry interleave trait and it's
  // convenient for next lower.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  strides.back() = rewriter.getIndexAttr(2);

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));

  auto insert0 = rewriter.create<tensor::InsertSliceOp>(
      loc, opa, emptyOp, offsets, sizes, strides);

  offsets.back() = rewriter.getIndexAttr(1);
  auto insert1 = rewriter.create<tensor::InsertSliceOp>(
      loc, opb, insert0, offsets, sizes, strides);
  rewriter.replaceOp(op, insert1);
  return success();
}

LogicalResult
CatConverter::matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Value opa = op.getLhs();
  Value opb = op.getRhs();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                  resType.getElementType());

  auto rank = resType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  auto inputType = dyn_cast<RankedTensorType>(opa.getType());

  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      });

  auto insert0 = rewriter.create<tensor::InsertSliceOp>(
      loc, opa, emptyOp, offsets, sizes, strides);

  offsets[0] =
      rewriter.getIndexAttr(inputType.getRank() ? inputType.getShape()[0] : 1);
  auto insert1 = rewriter.create<tensor::InsertSliceOp>(
      loc, opb, insert0, offsets, sizes, strides);

  rewriter.replaceOp(op, insert1);
  return success();
}

/// @brief Convert tt.gather to func.call. BiShengIR captures the func
///        with assumed semantics.
/// @param op The `triton::GatherOp` operation to be rewritten.
/// @param adaptor An adaptor for the operation's operands.
/// @param rewriter A pattern rewriter used to modify the IR.
/// @return A `LogicalResult` indicating whether the rewrite was successful.
LogicalResult
GatherConverter::matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value src = adaptor.getSrc();
  Value idx = adaptor.getIndices();
  Value res = op.getResult();
  auto gatherAxis = op.getAxis();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  llvm::SmallString<128> funcName = gatherFuncNameBase;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName += "_" + std::to_string(uniqueId++);
  }

  auto resTy = res.getType();
  auto libFnType = rewriter.getFunctionType(
      {src.getType(), idx.getType(), rewriter.getI32Type()}, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  Value axis = rewriter.create<arith::ConstantIntOp>(loc, gatherAxis, 32);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}),
                                              ValueRange({src, idx, axis}));

  rewriter.replaceOp(op, callOp);

  return success();
}

LogicalResult
SplitConverter::matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Value input = op.getSrc();
  auto loc = op.getLoc();
  auto inputType = cast<RankedTensorType>(input.getType());

  int64_t rank = inputType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  // Similar to JoinConverter, here adjust last dimension stride
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  strides.back() = rewriter.getIndexAttr(2);

  auto outType = dyn_cast<RankedTensorType>(op.getOutLHS().getType());
  auto sizes = llvm::map_to_vector(outType.getShape(), [&](int64_t t) {
    return OpFoldResult(rewriter.getIndexAttr(t));
  });
  sizes.push_back(rewriter.getIndexAttr(1));

  auto slice0 = rewriter.create<tensor::ExtractSliceOp>(
      loc, outType, input, offsets, sizes, strides);

  offsets.back() = rewriter.getIndexAttr(1);
  auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
      loc, outType, input, offsets, sizes, strides);

  SmallVector<Value, 2> slices = {slice0.getResult(), slice1.getResult()};
  rewriter.replaceOp(op, ValueRange(slices));
  return success();
}

/*
the element-wise most significant N bits of the 2N-bit product of x and y
%x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi32>
*/
LogicalResult TritonMulhiuiConverter::matchAndRewrite(
    triton::MulhiUIOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value opl = op.getX();
  Value opr = op.getY();
  Value res = op.getResult();
  auto newMulOp = rewriter.create<arith::MulSIExtendedOp>(
      loc, res.getType(), res.getType(), opl, opr);
  // triton only need the high value
  rewriter.replaceOp(op, ValueRange{newMulOp.getHigh()});
  return success();
}

LogicalResult TritonPreciseSqrtConverter::matchAndRewrite(
    triton::PreciseSqrtOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<math::SqrtOp>(op, adaptor.getOperands());
  return success();
}

LogicalResult DevicePrintConverter::matchAndRewrite(
    triton::PrintOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));
  SmallVector<Type, 4> inputTypes;
  for (auto arg : op.getArgs()) {
    inputTypes.push_back(arg.getType());
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp =
      rewriter.create<func::FuncOp>(op.getLoc(), printFuncNameBase, libFnType);
  SymbolTable symTab(moduleOp);
  auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
  if (failed(maybePrintFuncNameAttr)) {
    return op->emitError(
        "failed to create a unique func name for device_print");
  }
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  auto prefixAttr = op.getPrefixAttr();
  funcOp->setAttr(prefixAttrName, prefixAttr);
  auto hexAttr = op.getHexAttr();
  funcOp->setAttr(hexAttrName, hexAttr);

  rewriter.setInsertionPoint(op);
  rewriter.create<func::CallOp>(op.getLoc(), funcOp, op.getArgs());

  rewriter.eraseOp(op);
  return success();
}


LogicalResult DeviceAssertConverter::matchAndRewrite(
    triton::AssertOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto msgAttr = op.getMessageAttr();
  // Filter out automatically inserted assert ops
  if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(msgAttr)) {
    llvm::StringRef msg = strAttr.getValue();
    if (msg.contains("overflow detected for operation")) {
      rewriter.eraseOp(op);
      return success();
    }
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));
  auto conditionType = op.getCondition().getType();

  auto libFnType = rewriter.getFunctionType({conditionType}, {});
  auto funcOp =
      rewriter.create<func::FuncOp>(op.getLoc(), printFuncNameBase, libFnType);
  mlir::SymbolTable symTab(moduleOp);
  auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
  if (failed(maybePrintFuncNameAttr)) {
    return op->emitError(
        "failed to create a unique func name for device_assert");
  }
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  funcOp->setAttr(msgAttrName, msgAttr);

  rewriter.setInsertionPoint(op);
  rewriter.create<func::CallOp>(op.getLoc(), funcOp, ValueRange{op.getCondition()});

  rewriter.eraseOp(op);
  return success();
}

LogicalResult
MatmulConverter::matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto opa = adaptor.getA();
  auto opb = adaptor.getB();
  auto opc = adaptor.getC();
  auto dstType = cast<RankedTensorType>(op.getType());
  auto inputPrec = op.getInputPrecision();

  if (dstType.getRank() == 2) {
    auto matmulOp = rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, ValueRange{opa, opb}, ValueRange{opc});
    matmulOp->setAttr(
        "input_precison",
        rewriter.getStringAttr(stringifyInputPrecision(inputPrec)));
  } else if (dstType.getRank() == 3) {
    auto matmulOp = rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
        op, ValueRange{opa, opb}, ValueRange{opc});
    matmulOp->setAttr(
        "input_precison",
        rewriter.getStringAttr(stringifyInputPrecision(inputPrec)));
  } else {
    llvm_unreachable("Datatype of DotOp operands could only be 2D or 3D");
  }
  return success();
}


LogicalResult SortOpConverter::matchAndRewrite(
    triton::SortOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const
    {
  Value src = adaptor.getSrc();
  auto rankedSrcTy = cast<RankedTensorType>(src.getType());
  auto srcElemTy = rankedSrcTy.getElementType();
  auto srcShape = rankedSrcTy.getShape();
  auto srcEnc = rankedSrcTy.getEncoding();

  MLIRContext *ctx = rewriter.getContext();

  Type backendElemTy = srcElemTy;
  if (srcElemTy.isInteger(8)) {
    backendElemTy = Float16Type::get(ctx);   // i8 -> f16
  } else if (srcElemTy.isInteger(16)) {
    backendElemTy = Float32Type::get(ctx);   // i16 -> f32
  }
  Type backendTensorTy = RankedTensorType::get(srcShape, backendElemTy, srcEnc);

  Type valuesTy = src.getType();

  Location loc = op.getLoc();
  auto dimAttr = op->getAttrOfType<IntegerAttr>("dim");
  auto descAttr = op->getAttrOfType<BoolAttr>("descending");
  if (!dimAttr || !descAttr) {
    op->emitError("missing 'dim' or 'descending' attribute");
    return failure();
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    op->emitError("must be inside a module");
    return failure();
  }

  llvm::SmallString<64> baseName("triton_sort");
  llvm::SmallString<64> funcName = baseName;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName = baseName;
    funcName += ("_" + std::to_string(uniqueId++));
  }

  auto i64Ty = IntegerType::get(ctx, 64);
  auto i1Ty  = IntegerType::get(ctx, 1);
  auto libFnType = rewriter.getFunctionType(
      {backendTensorTy, i64Ty, i1Ty},
      {backendTensorTy});

  auto moduleIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  rewriter.restoreInsertionPoint(moduleIP);

  Value srcForCall = src;
  if (backendElemTy != srcElemTy) {
    srcForCall = rewriter.create<arith::SIToFPOp>(loc, backendTensorTy, src);
  }

  Value dimVal = rewriter.create<arith::ConstantIntOp>(loc, dimAttr.getInt(), 64);
  Value descVal = rewriter.create<arith::ConstantIntOp>(loc, descAttr.getValue() ? 1 : 0, 1);

  auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
  auto callOp = rewriter.create<func::CallOp>(
      loc,
      TypeRange({backendTensorTy}),
      callee,
      ValueRange({srcForCall, dimVal, descVal})
  );

  Value valuesFloat = callOp.getResult(0);   // tensor<f16/f32>

  Value finalValues = valuesFloat;
  if (backendElemTy != srcElemTy) {
    finalValues = rewriter.create<arith::FPToSIOp>(loc, valuesTy, valuesFloat);
  }

  rewriter.replaceOp(op, {finalValues});

  return success();
}


LogicalResult
DotScaledConverter::matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const
                                    {
  Value lhs = adaptor.getLhs();
  Value lhsScale = adaptor.getLhsScale();
  Value rhsScale = adaptor.getRhsScale();
  Value rhs = adaptor.getRhs();
  Value c = adaptor.getC();
  RankedTensorType dstType = cast<RankedTensorType>(op.getType());

  RankedTensorType lhsTy = cast<RankedTensorType>(lhs.getType());
  RankedTensorType lhsScaleTy = cast<RankedTensorType>(lhsScale.getType());
  RankedTensorType rhsScaleTy = rhsScale ? cast<RankedTensorType>(rhsScale.getType()) : nullptr;
  RankedTensorType rhsTy = cast<RankedTensorType>(rhs.getType());

  Type fp32Ty = rewriter.getF32Type();
  RankedTensorType lhsF32Ty = RankedTensorType::get(lhsTy.getShape(), fp32Ty);
  RankedTensorType rhsF32Ty = RankedTensorType::get(rhsTy.getShape(), fp32Ty);

  Value lhsF32 = rewriter.create<arith::ExtFOp>(op.getLoc(), lhsF32Ty, lhs);
  Value rhsF32 = rewriter.create<arith::ExtFOp>(op.getLoc(), rhsF32Ty, rhs);

  Value lhsScaleF32;
  Value rhsScaleF32;
  Value c127 = rewriter.create<arith::ConstantOp>(
    op.getLoc(),
    rewriter.getI16Type(),
    rewriter.getI16IntegerAttr(127)
    );
  Value c7 = rewriter.create<arith::ConstantOp>(
    op.getLoc(),
    rewriter.getI16Type(),
    rewriter.getI16IntegerAttr(7)
  );
  Type i16Ty = rewriter.getI16Type();
  Type bf16Ty = rewriter.getBF16Type();

  if (lhsScaleTy.getElementType().isIntOrIndex()) {
    RankedTensorType lhsScaleI16Ty = RankedTensorType::get(lhsScaleTy.getShape(), i16Ty);
    Value lhsScaleI16 = rewriter.create<arith::ExtUIOp>(
      op.getLoc(),
      lhsScaleI16Ty,
      lhsScale
    );

    Value lhsShift127Empty = rewriter.create<tensor::EmptyOp>(
      op.getLoc(),
      lhsScaleI16Ty.getShape(),
      i16Ty
    );
    Value lhsShift127 = rewriter.create<linalg::FillOp>(
      op.getLoc(),
      ValueRange{c127},
      ValueRange{lhsShift127Empty}
    ).getResult(0);

    Value lhsScaleI16Add127 = rewriter.create<arith::AddIOp>(
      op.getLoc(),
      lhsScaleI16,
      lhsShift127
    );

    Value lhsShift7Empty = rewriter.create<tensor::EmptyOp>(
      op.getLoc(),
      lhsScaleI16Ty.getShape(),
      i16Ty
    );
    Value lhsShift7 = rewriter.create<linalg::FillOp>(
      op.getLoc(),
      ValueRange{c7},
      ValueRange{lhsShift7Empty}
    ).getResult(0);
    Value lhsScaleI16Shifted = rewriter.create<arith::ShLIOp>(
      op.getLoc(),
      lhsScaleI16Add127,
      lhsShift7
    );

    RankedTensorType lhsScaleBF16Ty = RankedTensorType::get(lhsScaleTy.getShape(), bf16Ty);
    Value lhsScaleBF16 = rewriter.create<arith::BitcastOp>(
      op.getLoc(),
      lhsScaleBF16Ty,
      lhsScaleI16Shifted
    );

    lhsScaleF32 = rewriter.create<arith::ExtFOp>(
      op.getLoc(),
      RankedTensorType::get(lhsScaleTy.getShape(), fp32Ty),
      lhsScaleBF16
    );
  } else {
      lhsScaleF32 = rewriter.create<arith::ExtFOp>(
      op.getLoc(),
      RankedTensorType::get(lhsScaleTy.getShape(), fp32Ty),
      lhsScale
    ).getResult();
  }

  int64_t D0 = lhsScaleTy.getShape()[0];
  int64_t D1 = lhsScaleTy.getShape()[1];
  SmallVector<int64_t> expandedShape1 = {D0, D1, 1};
  RankedTensorType expandedTy1 = RankedTensorType::get(expandedShape1, fp32Ty);
  Value expanded1 = rewriter.create<triton::ExpandDimsOp>(
    op.getLoc(),
    expandedTy1,
    lhsScaleF32,
    rewriter.getI32IntegerAttr(2)
  ).getResult();

  int64_t lhsDim1 = lhsTy.getShape()[1];
  if (lhsDim1 % D1 != 0) {
    return op.emitError("lhs dim1 must be an integer multiple of lhsScale dim1");
  }
  int64_t D2 = lhsDim1 / D1;
  SmallVector<int64_t> broadcastShape = {D0, D1, D2};
  RankedTensorType broadcastTy = RankedTensorType::get(broadcastShape, fp32Ty);
  Value broadcasted = rewriter.create<triton::BroadcastOp>(
    op.getLoc(),
    broadcastTy,
    expanded1
  ).getResult();

  SmallVector<ReassociationIndices> reassociation;
  reassociation.push_back({0});
  reassociation.push_back({1, 2});

  Value scaledLhs = rewriter.create<tensor::CollapseShapeOp>(
    op.getLoc(),
    RankedTensorType::get({D0, D1 * D2}, fp32Ty),
    broadcasted,
    reassociation
  ).getResult();

  Value scaledLhsFinal = rewriter.create<arith::MulFOp>(
    op.getLoc(),
    lhsF32,
    scaledLhs
  ).getResult();

  Operation *matmulOp;
  if (dstType.getRank() == 2) {
    matmulOp = rewriter.create<linalg::MatmulOp>(
      op.getLoc(), ValueRange{scaledLhsFinal, rhsF32}, ValueRange{c}
    );
  } else if (dstType.getRank() == 3) {
    matmulOp = rewriter.create<linalg::BatchMatmulOp>(
      op.getLoc(), ValueRange{scaledLhsFinal, rhsF32}, ValueRange{c}
    );
  } else {
    return op.emitError("DotScaledOp only support 2D or 3D tensor");
  }

  rewriter.replaceOp(op, matmulOp->getResults());
  return success();
}


} // namespace TTOpConverters
