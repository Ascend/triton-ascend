//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "Conversion/StructuredToMemref/StructuredToMemref.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Analysis/OpFoldResultUtils.h"

#include "Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Conversion/TritonArithToLinalg/Passes.h.inc"
#include "Utils/Utils.h"
#include "Analysis/MaskAnalysis.h"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims, ArrayRef<int64_t> dimMode,
                                    Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto tensorDim = sourceType.getShape();
  if(!dimMode.empty()){
    assert(dimMode.size() == rank);
    for(size_t i = 0; i < rank; ++i){
      if(dimMode[i] == 0) continue;
      offsets[i] = subOFRs(b.getIndexAttr(tensorDim[i]), dims[i], loc, b);
    }
  }

  // TODO 找到引入多余维度的地方并删除多余的1
  SmallVector<OpFoldResult> tempDim(dims);
  if(sourceType.getRank() != dims.size()){
    tempDim.clear();
    for(auto x : dims){
      auto s = getIntAttr(x);
      if(s.has_value() && s.value() == 1) continue;
      tempDim.push_back(x);
    }
  }
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, tempDim, strides);

  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), source,
                                     offsets, dims, strides);
}

namespace {

struct MakeTensorPtrConverter
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
private:
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
    assert(!op.isBlockPtr());
    // tensor<1024x!tt.ptr<f32>>
    auto ptrType = cast<triton::PointerType>(
        cast<RankedTensorType>(op.getType()).getElementType());
    return ptrType.getPointeeType();
  }

  static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
    assert(op.isBlockPtr());
    // !tt.ptr<tensor<128x64xbf16>, 1>
    auto shapedType = cast<ShapedType>(
        cast<triton::PointerType>(op.getType()).getPointeeType());
    return shapedType.getElementType();
  }

  static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                        ArrayRef<int64_t> staticStrides,
                                        ArrayRef<int64_t> resultShape) {
    auto layout =
        StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
    Type elemType;
    if (op.isBlockPtr()) {
      elemType = getElementTypeBlockPtr(op);
    } else {
      elemType = getElementTypeStructuredPtr(op);
    }
    return MemRefType::get(resultShape, elemType, layout);
  }

  // If there are dimensions with size 1 and stride 0, replace 0 stride with
  // the product of sizes of all lower dimensions. This avoids creating memref
  // with zero stride.
  static llvm::SmallVector<OpFoldResult>
  getMixedStridesForMemref(tts::MakeTensorPtrOp op, OpBuilder &b) {
    llvm::SmallVector<OpFoldResult> strides;
    auto accumulate = 1;
    for (auto [size, stride] :
         llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
      auto strideIntAttr = getIntAttr(stride);
      if (size == 1 && strideIntAttr && strideIntAttr.value() == 0) {
        strides.push_back(b.getIndexAttr(accumulate));
      } else {
        strides.push_back(stride);
      }
      accumulate *= size;
    }
    std::reverse(strides.begin(), strides.end());
    return strides;
  }

  static OpFoldResult accumulateTargetOffset(tts::MakeTensorPtrOp op,
                                             OpBuilder &b) {
    Location loc = op->getLoc();
    OpFoldResult targetOffset = b.getIndexAttr(0);
    for (auto o : op.getMixedOffsets()) {
      targetOffset = addOFRs(targetOffset, o, loc, b);
    }
    return targetOffset;
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createSideBySideCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    auto targetOffset =
        ofrToIndexValue(accumulateTargetOffset(op, rewriter), loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling side-by-side wraparound
    //
    // Note: We do not support cases where the target has already overflown the
    // number of columns! This is because in PtrAnalysis, the offset has already
    // been collapsed into a single dimension, so it is ambiguous to determine
    // whether the offset actually overflows or just refers to an element on the
    // subsequent rows.
    //
    // Same limitations apply to the stacked wraparound case.
    //
    ////////////////////////////////////////////////////////////////////////////
    //
    //    nextOffset - targetOffset = colSize
    //    d1 + d2 = colSize
    //                          N
    //                                x            clampedOffset
    //      --------------------------*----------------*-----*
    //      |                                          |     nextOffset (might
    //      |                    targetOffset          |             overflow)
    //  y   *-----                    *----------------|
    //      |    |                    |                |
    //  M   |-----                    -----------------|
    //      | d2                              d1       |
    //      --------------------------------------------
    //
    //    x = targetOffset % N
    //    nextOffset = x + colSize
    //    clampedOffset = min(nextOffset, N)
    //    d1 = clampedOffset - x
    //
    ////////////////////////////////////////////////////////////////////////////

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{

            // Row stays the same
            resultShape[0],

            // Column is dynamic, in most cases, this
            // should be the same as the original column.
            // The last chunk may be smaller due to
            // wrapping around.
            ShapedType::kDynamic});

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value modN = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);

    Value x = rewriter.create<arith::RemSIOp>(loc, targetOffset, modN);
    Value y = rewriter.create<arith::SubIOp>(loc, targetOffset, x);

    SmallVector<Value> strideVals =
        ofrsToIndexValues(op.getMixedStrides(), loc, rewriter);

    // First chunk
    Value nextOffset = rewriter.create<arith::AddIOp>(loc, x, colSize);
    Value clampedOffset =
        rewriter.create<arith::MinSIOp>(loc, nextOffset, modN);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOffset, x);
    SmallVector<Value> sizes1{rowSize, d1};

    auto cast1 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), targetOffset, sizes1, strideVals);

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, colSize, d1);
    SmallVector<Value> sizes2{rowSize, d2};

    auto cast2 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), y, sizes2, strideVals);

    return {cast1, cast2};
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createStackedCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {

    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    assert(resultShape.size() == 2);

    auto targetOffset =
        ofrToIndexValue(accumulateTargetOffset(op, rewriter), loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling stacked wraparound
    //
    // We do not support cases where the target offset has already overflown the
    // number of rows. See side-by-side wraparound for details.
    //
    ////////////////////////////////////////////////////////////////////////////
    //    We're loading a tensor of dim (rowSize, colSize)
    //    d1 + d2 = rowSize
    //    d2 is the number of rows that overflow
    //
    //                       cols
    //
    //               wrappedAroundOff
    //      --------------*------------*--------
    //      |        d2   |            |       |
    //      |             |------------|       |
    //  rows|                                  |
    //      |                                  |
    //      |           targetOffset           |
    //      |             *------------|       |
    //      |             |            |       |
    //      |         d1  |            |       |
    //      |             | clampedOff |       |
    //      --------------*---------------------
    //                    |  overflow  |
    //                    *-------------
    //                 nextOff
    //
    //    wrappedAroundOff = targetOffset % cols
    //    clampedOff = (rows * strideRows) + wrappedAroundOff
    //                  ~~~~~~~~~~~~~~~~~
    //                         ^
    //                         |
    //          We have already computed
    //          rows * strideRows = modRow = shape[1]
    //          in TritonToStructured
    //
    //          clampedOff - targetOffset
    //    d1 = --------------------
    //              strideRows

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{
            // Row is dynamic, in most cases, this should
            // be the same as the original row. The last
            // chunk may be smaller due to wrapping
            // around.
            ShapedType::kDynamic,

            // Col stays the same.
            resultShape[1],
        });

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);

    Value modRow = op.getShape()[0];

    // First chunk
    Value wrappedAroundOff =
        rewriter.create<arith::RemSIOp>(loc, targetOffset, strideRow);
    Value clampedOff =
        rewriter.create<arith::AddIOp>(loc, modRow, wrappedAroundOff);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOff, targetOffset);
    d1 = rewriter.create<arith::DivSIOp>(loc, d1, strideRow);

    SmallVector<Value> sizes1{d1, colSize};
    memref::ReinterpretCastOp cast1 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), targetOffset, sizes1,
            ValueRange{strideRow, strideCol});

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, rowSize, d1);
    SmallVector<Value> sizes2{d2, colSize};
    memref::ReinterpretCastOp cast2 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), wrappedAroundOff, sizes2,
            ValueRange{strideRow, strideCol});

    return {cast1, cast2};
  }

  LogicalResult rewriteSplitPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {

    auto parentShape = op.getStaticShape();

    SmallVector<Value> casts;
    StringRef wrapType;

    if (parentShape[0] == ShapedType::kDynamic) {
      // Stacked case
      assert(parentShape[1] == 0);
      auto [cast1, cast2] = createStackedCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_STACKED;
    } else {
      assert(parentShape[0] == 0);
      auto [cast1, cast2] = createSideBySideCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_SIDE_BY_SIDE;
    }

    auto combinedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getType(), casts);

    combinedCast->setAttr(wrapType, rewriter.getUnitAttr());

    rewriter.replaceOp(op, combinedCast);

    return success();
  }

  LogicalResult rewritePtr(ArrayRef<int64_t> resultShape, bool isBlockPtr,
                           tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto targetOffset = accumulateTargetOffset(op, rewriter);
    auto staticTargetOffset = getIntAttr(targetOffset);
    auto resultType = getResultMemrefType(
        op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides,
        resultShape);

    // The base ptr, which is from one of the args, would have already been
    // converted to memref<*> at this point, so get the base from adaptor.
    //
    // For block pointers, the base could come from a sequence of `tt.addptr`,
    // which at this point has already been lowered to a sequence of
    // `memref.reinterpret_cast` ops. The offset in such cases are dynamic.
    // (see test/Conversion/StructuredToMemref/block_ptr_complex_offset.mlir)
    //
    // For non-block pointer cases, the base is the reinterpret_cast of a
    // function argument. Assert that the offset is a constant 0 in such cases.
    auto ptr = adaptor.getBase();
    if (auto reinterpretCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
      auto offset = reinterpretCast.getMixedOffsets()[0];
      auto intAttr = getIntAttr(offset);
      // TODO 更改assert
      // assert(isBlockPtr || (intAttr.has_value() && intAttr.value() == 0));
      targetOffset = addOFRs(targetOffset, reinterpretCast.getMixedOffsets()[0],
                             op->getLoc(), rewriter);
    }

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        op.getLoc(), resultType, ptr, targetOffset, op.getMixedSizes(),
        mixedStrides);

    rewriter.replaceOp(op, castOp);

    return success();
  }

  LogicalResult
  rewriteStructuredPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
    return rewritePtr(resultShape, false, op, adaptor, rewriter);
  }

  LogicalResult rewriteBlockPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Block pointers are basically the same as structured pointers except that
    // the return types are !tt.ptr<tensor<AxBxCxbf16>> instead of
    // tensor<AxBxCx!tt.ptr<bf16>>
    ArrayRef<int64_t> resultShape =
        cast<ShapedType>(
            cast<triton::PointerType>(op.getType()).getPointeeType())
            .getShape();
    return rewritePtr(resultShape, true, op, adaptor, rewriter);
  }

public:
  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::is_sorted(op.getOrder(), std::greater<>())) {
      emitError(op.getLoc()) << "non-decreasing dimension order on tensor "
                                "pointers are not yet supported";
      return failure();
    }

    if (op.isBlockPtr()) {
      return rewriteBlockPtr(op, adaptor, rewriter);
    }

    if (op.isStructuredPtr()) {
      return rewriteStructuredPtr(op, adaptor, rewriter);
    }

    if (op.isSplitPtr()) { // TODO: WIP
      return rewriteSplitPtr(op, adaptor, rewriter);
    }

    return failure();
  }
};

struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
private:
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{zero, block1Col},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{block1Row, zero},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  memref::SubViewOp createSubview(Value src, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType =
        memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
    return rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType),
                                              src, offsets, sizes, strides);
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getSideBySideSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                        Location loc,
                        ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult col1 =
        rewriter.create<memref::DimOp>(loc, block1, 1).getResult();
    OpFoldResult subviewCol1 = minOFRs(col1, subviewColFull, loc, rewriter);
    OpFoldResult subviewCol2 =
        subOFRs(subviewColFull, subviewCol1, loc, rewriter);

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRowFull, subviewCol1},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRowFull, subviewCol2},
                             strides, loc, rewriter);

    return {sv1, sv2};
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getStackedSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                     const Location loc,
                     ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult row1 =
        rewriter.create<memref::DimOp>(loc, block1, 0).getResult();
    OpFoldResult subviewRow1 = minOFRs(row1, subviewRowFull, loc, rewriter);
    OpFoldResult subviewRow2 =
        subOFRs(subviewRowFull, subviewRow1, loc, rewriter);

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRow1, subviewColFull},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRow2, subviewColFull},
                             strides, loc, rewriter);
    return {sv1, sv2};
  }

  LogicalResult
  rewriteStructuredLoad(tts::LoadOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    assert(!op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();
    auto other = op.getOther();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(tensorType.getShape(), elemType));

    // No mask
    assert(!other && "other value used in non-masked load");

    if (auto unrealizedCast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      auto memrefs = unrealizedCast.getOperands();
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        createSideBySideCopies(block1, block2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        createStackedCopies(block1, block2, alloc, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }
    } else {
      rewriter.create<memref::CopyOp>(loc, ptr, alloc);
    }

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }

  LogicalResult rewriteMaskedLoad(tts::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();
    auto dimMode = adaptor.getMaskMode();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(tensorType.getShape(), elemType));

    SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();

    // TODO Exclude cases where the value of the mask is 1 during mask generation
    SmallVector<OpFoldResult> temp(mixedDims);
    if(tensorType.getRank() != mixedDims.size()){
      mixedDims.clear();
      for(auto x : temp){
        auto s = getIntAttr(x);
        if(s.has_value() && s.value() == 1) continue;
        mixedDims.push_back(x);
      }
    }

    // Fill load destination with other value
    if (op.getOther()) {
      // For each dimension check if dims[i] < shape[i], or-accumulate
      // the result
      auto shape = tensorType.getShape();
      auto accBase =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
              .getResult();
      for (size_t i = 0; i < shape.size(); i++) {
        auto shapei = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(shape[i]));

        Value dimi = dyn_cast<Value>(mixedDims[i]);
        if (!dimi) {
          dimi = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIndexAttr(op.getStaticMaskDims()[i]));
        }

        Value cmp = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, dimi, shapei);
        accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmp);
      }

      // condition the memset on the or-accumulation
      // initialize with padding prior to CopyOp
      rewriter.create<scf::IfOp>(loc, accBase, [&](OpBuilder &b, Location loc) {
        b.create<linalg::FillOp>(loc, ValueRange{op.getOther()},
                                 ValueRange{alloc});
        b.create<scf::YieldOp>(loc);
      });
    }

    if (auto unrealizedCast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {

      auto memrefs = unrealizedCast.getOperands();
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        auto [subview1, subview2] =
            getSideBySideSubviews(mixedDims, block1, block2, loc, rewriter);
        createSideBySideCopies(subview1, subview2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        auto [subview1, subview2] =
            getStackedSubviews(mixedDims, block1, block2, loc, rewriter);
        createStackedCopies(subview1, subview2, alloc, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }

      rewriter.eraseOp(unrealizedCast);

    } else {
      memref::SubViewOp srcSubview =
          getSubview(tensorType.getRank(), mixedDims, dimMode, ptr, loc, rewriter);
      memref::SubViewOp dstSubview =
          getSubview(tensorType.getRank(), mixedDims, dimMode, alloc, loc, rewriter);
      rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    }

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }

public:
  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.hasMask()) {
      return rewriteMaskedLoad(op, adaptor, rewriter);
    } else {
      return rewriteStructuredLoad(op, adaptor, rewriter);
    }
  }
};

struct StoreConverter : public OpConversionPattern<tts::StoreOp> {
private:
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  static tensor::ExtractSliceOp
  getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source,
                  const Location loc, OpBuilder &b) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

    auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                           dims, strides);

    return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims,
                                            strides);
  }

public:
  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    SmallVector<int64_t> dimMode;
    auto storeValue = op.getValue();
    auto rank = cast<RankedTensorType>(storeValue.getType()).getRank();

    if (op.hasMask()) {
      auto mixedDims = op.getMixedMaskDims();

      auto srcSlice =
          getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      auto dstSubview = getSubview(rank, mixedDims, dimMode, ptr, loc, rewriter);

      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    } else {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, storeValue, ptr);
      storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ScalarLoadConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = op->getLoc();
    auto memrefPtr = adaptor.getPtr();
    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
    auto loadOp = rewriter.create<affine::AffineLoadOp>(loc, memrefPtr, zeroMap,
                                                        std::nullopt);
    rewriter.replaceOp(op, loadOp.getResult());

    return success();
  }
};

struct ScalarStoreConverter : public OpConversionPattern<triton::StoreOp> {
private:
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getValue().getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = op->getLoc();
    auto memrefPtr = adaptor.getPtr();
    auto val = op.getValue();
    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    rewriter.create<affine::AffineStoreOp>(loc, val, memrefPtr, zeroMap,
                                           std::nullopt);
    rewriter.eraseOp(op);

    return success();
  }
};

struct UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
private:
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

public:
  UnrealizedCastConverter(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<UnrealizedConversionCastOp>(typeConverter,
                                                        context) {}

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = op->getResultTypes()[0];
    auto input = op.getInputs()[0];
    auto inputType = input.getType();

    if (!isa<triton::PointerType>(resType) ||
        !isa<MemRefType, UnrankedMemRefType>(inputType)) {
      return failure();
    }

    if (auto reinterpretCast =
            input.getDefiningOp<memref::ReinterpretCastOp>()) {
      rewriter.replaceOp(op, reinterpretCast);
    } else {
      auto ptrType = cast<triton::PointerType>(resType);
      auto memrefType =
          cast<MemRefType>(getTypeConverter()->convertType(ptrType));

      auto cast = rewriter.create<memref::ReinterpretCastOp>(
          op->getLoc(), memrefType, op.getInputs()[0], 0 /*offset*/,
          SmallVector<int64_t>{1} /*sizes*/,
          SmallVector<int64_t>{1} /*strides*/);

      rewriter.replaceOp(op, cast);
    }

    return success();
  }
};

class AtomicRMWConverter : public OpConversionPattern<tts::AtomicRMWOp> {
private:
  Value createAtomicBinaryOps(OpBuilder &builder, Location loc,
                              tts::AtomicRMWOp op, Type elementType,
                              Value lhs, Value rhs) const {
    auto rmwOp = op.getAtomicRmwOp();

    // it has been confirmed in AtomicRMWConverter::matchAndRewrite
    // that the ptr of op is of MemRefType
    Value binaryOp;
    if (rmwOp == triton::RMWOp::FADD) {
      binaryOp = builder.create<arith::AddFOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::ADD) {
      binaryOp = builder.create<arith::AddIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::XOR) {
      binaryOp = builder.create<arith::XOrIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::OR) {
      binaryOp = builder.create<arith::OrIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::AND) {
      binaryOp = builder.create<arith::AndIOp>(loc, lhs, rhs);
    } else if (rmwOp == triton::RMWOp::MAX) {
      // Max/Min only support f32/i32 for now
      // Other type is not supported because of semantic.py
      if (isa<FloatType>(elementType)) {
        binaryOp = builder.create<arith::MaxNumFOp>(loc, lhs, rhs);
      } else {
        binaryOp = builder.create<arith::MaxSIOp>(loc, lhs, rhs);
      }
    } else if (rmwOp == triton::RMWOp::MIN) {
      if (isa<FloatType>(elementType)) {
        binaryOp = builder.create<arith::MinNumFOp>(loc, lhs, rhs);
      } else {
        binaryOp = builder.create<arith::MinSIOp>(loc, lhs, rhs);
      }
    } else if (rmwOp == triton::RMWOp::XCHG) {
      binaryOp = rhs;
    } else {
      op.emitOpError("unsupported atomic RMW operation: ");
      llvm_unreachable(
          "Not implemented. Support fadd, add, max, min for now !");
    }
    return binaryOp;
  }

  // used when handling scalar
  // to verify whether we need to handle this scalar
  bool isConstantMaskTrue(Value mask) const {
    if (auto denseAttr =
            mask.getDefiningOp()->getAttrOfType<DenseElementsAttr>("value")) {
      auto eleType = denseAttr.getType().getElementType();
      if (isa<IntegerType>(eleType) &&
          cast<IntegerType>(eleType).getWidth() == 1) {
        auto values = denseAttr.getValues<bool>();
        return values[0];
      }
    }
    return false;
  }

  DenseSet<triton::RMWOp> softwareAtomicKinds = {
      triton::RMWOp::AND, triton::RMWOp::OR, triton::RMWOp::XOR};

public:
  explicit AtomicRMWConverter(MLIRContext *context)
    : OpConversionPattern<tts::AtomicRMWOp>(context) {}
  using OpConversionPattern<tts::AtomicRMWOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tts::AtomicRMWOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override {
      // If the result of AtomicRMWOp is not used, we don't need to load the old
      // data stored at the ptr
      auto ptr = adaptor.getPtr();
      auto val = op.getValue();
      auto loc = op.getLoc();
    
      auto resType = dyn_cast<TensorType>(op.getResult().getType());
      if (!resType) {
        return rewriter.notifyMatchFailure(
            op, "atomicRMWConverter: scalar will be handled by "
                "ScalarAtomicRMWCanonicalizer");
      }
    
      auto rmwOp = op.getAtomicRmwOp();
      if (rmwOp == triton::RMWOp::UMAX || rmwOp == triton::RMWOp::UMIN) {
        return rewriter.notifyMatchFailure(
            op, "AtomicRMWConverter: unsupported atomic kind for now");
      }
    
      // 1. Simple case where no mask is used.
      auto type = dyn_cast<MemRefType>(ptr.getType());
      if (!type) {
        // Seen when implicit broadcasting is done late in a chain of
        // operations. The workaround is to broadcast the pointers early in the
        // address calculation. A proper fix is complicated, but at least we can
        // provide a better error message.
        return rewriter.notifyMatchFailure(
            op, "AtomicRMWOp expects a memref, not a memref of pointers");
      }
    
      auto dstMemref = ptr;
      // Well, linalg structure op wouldn't support mixed tensor/buffer semantics
      // any more in latest LLVM(triton LLVM dependency has involed this), so we
      // need to convert tensor to buffer early.
      auto dstType = dstMemref.getType();
      Value inputMemref =
          rewriter.create<bufferization::ToMemrefOp>(loc, dstType, val);
    
      // 2. handle the mask for the atomic op
    
      SmallVector<int64_t> dimMode;
      auto rank = cast<RankedTensorType>(val.getType()).getRank();
    
      if (op.hasMask()) {
        auto mixedDims = op.getMixedMaskDims();
    
        inputMemref = getSubview(rank, mixedDims, dimMode, inputMemref, loc, rewriter);
        dstMemref = getSubview(rank, mixedDims, dimMode, ptr, loc, rewriter);
    
        // auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        //     loc, srcSlice, dstSubview);
        // storeOp.setWritable(true);
      } else {
        // fixme, kaixin
        // auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        //     loc, storeValue, ptr);
        // storeOp.setWritable(true);
      }
    

    
    
      // 3. If needed, handle the return value of atomic op
      //
      // tt.atomicRMW op has two part of feature
      // 1. load the old data at the ptr
      // 2. atomically store the data on ub to the ptr
      //    at the same time it perform the action it has been assigned
      // So we lower this op to load + atomically store
      //
      // The first part is not necessary when the returned value of atomic op
      // is not used, it will be deleted cause it's meaningless
      // Here, we preemptively determine whether it will be used
      // and decide whether it is necessary to create the load process based on
      // this assessment.
      //
      // logic of handling is copied
      // TODO: decoupling the logic of load, put it in the Utils
      if (!op.getResult().use_empty()) {
        auto tensorType =
            RankedTensorType::get(type.getShape(), type.getElementType());
        auto alloc = rewriter.create<memref::AllocOp>(
            loc, MemRefType::get(type.getShape(), type.getElementType()));
    
        // For the return value, don't need to care about mask for now
        // this op don't support other, so we best not fill it
        rewriter.create<memref::CopyOp>(loc, ptr, alloc);
        Value tensor = rewriter.create<bufferization::ToTensorOp>(
            loc, tensorType, alloc, true /* restrict */, true /* writable */);
        rewriter.replaceOp(op, tensor);
      }
    
      // create element-wise map
      //int64_t rank = type.getRank();
      SmallVector<AffineExpr> inputDims;
      auto context = rewriter.getContext();
    
      for (int i = 0; i < rank; i++) {
        inputDims.push_back(getAffineDimExpr(i, context));
      }
    
      SmallVector<AffineMap> indexingMaps;
      // As mask has been erased for now
      // the number of input must be 2
      // the input memref is also the output memref
      // Thus, there are a total of three inputs and outputs.
      // so here we have 3 map to create
      for (int i = 0; i < 3; i++) {
        indexingMaps.push_back(AffineMap::get(rank, 0, inputDims, context));
      }
    
      auto linalgOp = rewriter.create<linalg::GenericOp>(
          loc, /* operands */ ValueRange{dstMemref, inputMemref},
          ValueRange{dstMemref}, indexingMaps,
          mlir::ConverterUtils::getNParallelLoopsAttrs(rank),
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
            Value opResult = createAtomicBinaryOps(nestedBuilder, nestedLoc, op,
                                                   type.getElementType(),
                                                   blockArgs[0], blockArgs[1]);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, opResult);
          });
    
      // "library_call"
      // indicating the actual semantic of this op
      // TODO: If the hardware support the MemSemantic/MemSyncScope
      //       We pass them down
      //       otherwise they need to be deleted
      const StringRef genericAtomicRMW = "GenericAtomicRMW";
      const StringRef memSemantic = "MemSemantic";
      const StringRef memSyncScope = "MemSyncScope";
      linalgOp->setAttr(genericAtomicRMW,
                        rewriter.getStringAttr(stringifyEnum(op.getAtomicRmwOp())));
      linalgOp->setAttr(memSemantic,
                        rewriter.getStringAttr(stringifyEnum(op.getSem())));
      linalgOp->setAttr(memSyncScope,
                        rewriter.getStringAttr(stringifyEnum(op.getScope())));
    
      // Mark atomic_and/or/xor specially which need software simulation in terms
      // of backend restriction
      if (softwareAtomicKinds.contains(op.getAtomicRmwOp()))
        linalgOp->setAttr("Software", rewriter.getUnitAttr());
    
      // if the result hasn't been replace by load
      // we need to erase it here
      if (op.getResult().use_empty()) {
        rewriter.eraseOp(op);
      }
      return success();
  }

};

class AtomicCASConverter : public OpConversionPattern<tts::AtomicCASOp> {
public:
  explicit AtomicCASConverter(MLIRContext *context) :
    OpConversionPattern<tts::AtomicCASOp>(context) {}
  using OpConversionPattern<tts::AtomicCASOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(tts::AtomicCASOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override {
      // If the result of AtomicCASOp is not used, we don't need to load the old
      // data stored at the ptr
      auto ptr = adaptor.getPtr();
      auto cmp = op.getCmp();
      auto val = op.getVal();
      auto loc = op.getLoc();
    
      auto resType = dyn_cast<TensorType>(op.getResult().getType());
      if (!resType) {
        return rewriter.notifyMatchFailure(
            op, "atomicCASConverter: scalar will be handled by "
                "ScalarAtomicCASCanonicalizer");
      }
    
      // 1. Simple case where no mask is used.
      auto type = dyn_cast<MemRefType>(ptr.getType());
      if (!type) {
        return rewriter.notifyMatchFailure(
            op, "AtomicCASOp expects a memref, not a memref of pointers");
      }
    
      auto dstMemref = ptr;
      // Well, linalg structure op wouldn't support mixed tensor/buffer semantics
      // any more in latest LLVM(triton LLVM dependency has involed this), so we
      // need to convert tensor to buffer early.
      auto dstType = dstMemref.getType();
      Value inputMemref =
          rewriter.create<bufferization::ToMemrefOp>(loc, dstType, val);
    
      Value cmpMemref =
          rewriter.create<bufferization::ToMemrefOp>(loc, dstType, cmp);
    
      // 3. If needed, handle the return value of atomic op
      //
      // tt.atomicRMW op has two part of feature
      // 1. load the old data at the ptr
      // 2. atomically store the data on ub to the ptr
      //    at the same time it perform the action it has been assigned
      // So we lower this op to load + atomically store
      //
      // The first part is not necessary when the returned value of atomic op
      // is not used, it will be deleted cause it's meaningless
      // Here, we preemptively determine whether it will be used
      // and decide whether it is necessary to create the load process based on
      // this assessment.
      //
      // logic of handling is copied
      if (!op.getResult().use_empty()) {
        auto tensorType =
            RankedTensorType::get(type.getShape(), type.getElementType());
        auto alloc = rewriter.create<memref::AllocOp>(
            loc, MemRefType::get(type.getShape(), type.getElementType()));
    
        // For the return value, don't need to care about mask for now
        // this op don't support other, so we best not fill it
        rewriter.create<memref::CopyOp>(loc, ptr, alloc);
        Value tensor = rewriter.create<bufferization::ToTensorOp>(
            loc, tensorType, alloc, true /* restrict */, true /* writable */);
        rewriter.replaceOp(op, tensor);
      }
    
      // create element-wise map
      int64_t rank = type.getRank();
      SmallVector<AffineExpr> inputDims;
      auto context = rewriter.getContext();
    
      for (int i = 0; i < rank; i++) {
        inputDims.push_back(getAffineDimExpr(i, context));
      }
    
      SmallVector<AffineMap> indexingMaps;
      // As mask has been erased for now
      // the number of input must be 2
      // the input memref is also the output memref
      // Thus, there are a total of four inputs and outputs.
      // so here we have 4 map to create
      for (int i = 0; i < 4; i++) {   // 4: 3 input and 1 output
        indexingMaps.push_back(AffineMap::get(rank, 0, inputDims, context));
      }
    
      auto linalgOp = rewriter.create<linalg::GenericOp>(
          loc, ValueRange{dstMemref, cmpMemref, inputMemref},
          mlir::ValueRange{dstMemref}, indexingMaps,
          mlir::ConverterUtils::getNParallelLoopsAttrs(rank),
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
            Value lhs = blockArgs[0];
            Value rhs = blockArgs[1];
            Value setValue = blockArgs[2];
            Value cond;
            if (mlir::isa<mlir::FloatType>(lhs.getType())) {
              cond = nestedBuilder.create<arith::CmpFOp>(nestedLoc,
                                                         arith::CmpFPredicate::UEQ,
                                                         lhs, rhs);
            } else {
              cond = nestedBuilder.create<arith::CmpIOp>(nestedLoc,
                                                         arith::CmpIPredicate::eq,
                                                         lhs, rhs);
            }
            auto ifOp = nestedBuilder.create<scf::IfOp>(nestedLoc, TypeRange{setValue.getType()}, cond, true);
            {
              OpBuilder::InsertionGuard guard(nestedBuilder);
              nestedBuilder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
              nestedBuilder.create<scf::YieldOp>(nestedLoc, setValue);
            }
            {
              OpBuilder::InsertionGuard guard(nestedBuilder);
              nestedBuilder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
              nestedBuilder.create<scf::YieldOp>(nestedLoc, lhs);
            }
            nestedBuilder.setInsertionPointToEnd(nestedBuilder.getBlock());
            nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, ifOp.getResult(0));
          });
    
      const StringRef genericAtomicRMW = "GenericAtomicRMW";
      const StringRef memSemantic = "MemSemantic";
      const StringRef memSyncScope = "MemSyncScope";
      auto attr = mlir::StringAttr::get(context, "cas");
    
      linalgOp->setAttr(genericAtomicRMW, attr);
      linalgOp->setAttr(memSemantic,
                        rewriter.getStringAttr(stringifyEnum(op.getSem())));
      linalgOp->setAttr(memSyncScope,
                        rewriter.getStringAttr(stringifyEnum(op.getScope())));
    
      linalgOp->setAttr("Software", rewriter.getUnitAttr());
    
      // if the result hasn't been replace by load
      // we need to erase it here
      if (op.getResult().use_empty()) {
        rewriter.eraseOp(op);
      }
      return success();
  }
};


} // namespace

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<UnrealizedCastConverter>(typeConverter, patterns.getContext());
  patterns.add<MakeTensorPtrConverter, LoadConverter, StoreConverter,
               ScalarLoadConverter, ScalarStoreConverter,AtomicRMWConverter, 
               AtomicCASConverter>(
      patterns.getContext());
}
