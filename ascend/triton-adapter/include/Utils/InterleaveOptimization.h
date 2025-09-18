#pragma once

#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToLinalg/UseAnalysis.h"
#include "Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"


#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <type_traits>

namespace mlir {
namespace triton {

enum class IndexMode : int { EVEN_MODE = 0, ODD_MODE = 1 };

MemRefType expandInterleaveMemRefType(MemRefType originType);

std::pair<OpFoldResult, IndexMode>
recountReinterpretCastOffset(OpFoldResult originOffset, Builder &builder);

// LogicalResult
// DeinterleaveStatusOptimization(triton::LoadOp op,
//                                triton::LoadOp::Adaptor adaptor,
//                                ConversionPatternRewriter &rewriter);
// template<typename T>
// LogicalResult DeinterleaveStatusWithMaskOptimization(
//     T op, triton::LoadOp::Adaptor adaptor,
//     ConversionPatternRewriter &rewriter, triton_adapter::MaskState &mstate,
//     memref::AllocOp originAllocOp);


template<typename OP_TYPE >
LogicalResult DeinterleaveStatusOptimization(OP_TYPE op, Value ptr,
                               ConversionPatternRewriter &rewriter) {
  //auto ptr = adaptor.getPtr();
  if (auto reinterpretCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto loc = op.getLoc();

    // 1. Get new source memref type
    auto srcType = expandInterleaveMemRefType(reinterpretCast.getType());

    // 2. Create new ReinterpretCastOp
    auto originCastOffset = reinterpretCast.getConstifiedMixedOffset();
    auto castSize = reinterpretCast.getConstifiedMixedSizes();
    auto castStride = reinterpretCast.getConstifiedMixedStrides();
    // Actually, `castSize` is always constant value as `MemRefType` result
    if (auto lastDimSize = getConstantIntValue(castSize.back())) {
      castSize.back() = rewriter.getIndexAttr(lastDimSize.value() * 2);
    } else {
      return failure();
    }
    // Last element of castStride is also constant value as prerequisite
    // is that last dimension stride of casted memref type is always 2.
    castStride.back() = rewriter.getIndexAttr(1);
    auto [castOffset, indexMode] =
        recountReinterpretCastOffset(originCastOffset, rewriter);
    auto newCastOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, srcType, reinterpretCast.getViewSource(), castOffset, castSize,
        castStride);

    // 3. Create new memref allocOp
    auto newAllocOp = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(srcType.getShape(), srcType.getElementType()));

    // 4. Implement memref copy and bufferization back to tensor
    rewriter.create<memref::CopyOp>(loc, newCastOp.getResult(), newAllocOp);
    Value newTensor = rewriter.create<bufferization::ToTensorOp>(
        loc,
        RankedTensorType::get(srcType.getShape(), srcType.getElementType()),
        newAllocOp, true /* restrict */, true /* writable */);

    // 5. Implement tensor extract_slice to represent deinterleave
    // Here use `castOffset` to determine whether even index deinterleave or
    // odd index.
    SmallVector<OpFoldResult> extractOffsets(srcType.getRank(),
                                             rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> extractStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> extractSizes = llvm::to_vector(
        llvm::map_range(srcType.getShape(), [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    // Adjust extract_slice shape
    switch (indexMode) {
    case IndexMode::EVEN_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(0);
      break;
    case IndexMode::ODD_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(1);
      break;
    }
    extractStrides.back() = rewriter.getIndexAttr(2);
    extractSizes.back() = rewriter.getIndexAttr(srcType.getShape().back() / 2);

    Value deinterleaveSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newTensor, extractOffsets, extractSizes, extractStrides);

    rewriter.replaceOp(op, deinterleaveSlice);
    return success();
  }

  return failure();
}

template<typename OP_TYPE >
LogicalResult DeinterleaveStatusWithMaskOptimization(
    OP_TYPE op, Value ptr,
    ConversionPatternRewriter &rewriter, 
    SmallVector<OpFoldResult> &subviewOffsets, SmallVector<OpFoldResult> &subviewSizes,
    memref::AllocOp originAllocOp) {
  //auto ptr = adaptor.getPtr();
  if (auto reinterpretCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto loc = op.getLoc();

    // 1. Get new source memref type
    auto srcType = expandInterleaveMemRefType(reinterpretCast.getType());

    // 2. Create new ReinterpretCastOp
    auto originCastOffset = reinterpretCast.getConstifiedMixedOffset();
    auto castSize = reinterpretCast.getConstifiedMixedSizes();
    auto castStride = reinterpretCast.getConstifiedMixedStrides();

    if (auto lastDimSize = getConstantIntValue(castSize.back())) {
      castSize.back() = rewriter.getIndexAttr(lastDimSize.value() * 2);
    } else {
      return failure();
    }
    castStride.back() = rewriter.getIndexAttr(1);
    auto [castOffset, indexMode] =
        recountReinterpretCastOffset(originCastOffset, rewriter);

    auto newCastOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, srcType, reinterpretCast.getViewSource(), castOffset, castSize,
        castStride);

    // 3. Create new memref allocOp
    // To reuse existing linalg::fill, here need to change insertion point
    auto savedInsertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(originAllocOp);
    auto newAllocOp = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(srcType.getShape(), srcType.getElementType()));
    rewriter.restoreInsertionPoint(savedInsertPoint);

    // 4. Broadcast other value by linalg.fill if necessary
    auto other = op.getOther();
    // While deinterleave optimization will just adjust last dimension info
    // and origin mask state wouldn't involve last dimension. Therefore in
    // current `scf.if + linalg.fill` combination, condition of `if` could be
    // kept and just replace linalg.fill'
    if (other) {
      assert(originAllocOp->hasOneUse() &&
             llvm::isa<linalg::FillOp>(*(originAllocOp->getUsers().begin())));
      auto originFillOp =
          llvm::dyn_cast<linalg::FillOp>(*(originAllocOp->getUsers().begin()));

      assert(llvm::isa<scf::IfOp>(originFillOp->getParentOp()));
      auto ifOp = llvm::dyn_cast<scf::IfOp>(originFillOp->getParentOp());

      auto newFillOp = ifOp.getThenBodyBuilder().create<linalg::FillOp>(
          originFillOp.getLoc(), originFillOp.getInputs(),
          ValueRange{newAllocOp});
      rewriter.eraseOp(originFillOp);
    }

    // 5. Implement new subview, memref copy and bufferization back to tensor
    SmallVector<OpFoldResult> subviewStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    //SmallVector<OpFoldResult> subviewOffsets = mstate.offsets;
    //SmallVector<OpFoldResult> subviewSizes = mstate.dims;
    // Just adjust last dimension size to double
    std::optional<int64_t> originSubviewLastDim =
        getConstantIntValue(subviewSizes.back());
    assert(originSubviewLastDim.has_value());
    subviewSizes.back() =
        rewriter.getIndexAttr(originSubviewLastDim.value() * 2);

    auto argSubviewType = memref::SubViewOp::inferResultType(
        srcType, subviewOffsets, subviewSizes, subviewStrides);
    // alloca subview type doesn't carry layout attribute
    auto allocSubviewType = memref::SubViewOp::inferResultType(
        newAllocOp.getType(), subviewOffsets, subviewSizes, subviewStrides);

    memref::SubViewOp srcSubview = rewriter.create<memref::SubViewOp>(
        loc, llvm::cast<MemRefType>(argSubviewType), newCastOp, subviewOffsets,
        subviewSizes, subviewStrides);
    memref::SubViewOp dstSubview = rewriter.create<memref::SubViewOp>(
        loc, llvm::cast<MemRefType>(allocSubviewType), newAllocOp,
        subviewOffsets, subviewSizes, subviewStrides);
    rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    Value newTensor = rewriter.create<bufferization::ToTensorOp>(
        loc,
        RankedTensorType::get(srcType.getShape(), srcType.getElementType()),
        newAllocOp, true /* restrict */, true /* writable */);

    // 6. Implement tensor extract_slice to represent deinterleave
    // Here use `castOffset` to determine whether even index deinterleave or
    // odd index.
    SmallVector<OpFoldResult> extractOffsets(srcType.getRank(),
                                             rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> extractStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> extractSizes = llvm::to_vector(
        llvm::map_range(srcType.getShape(), [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    switch (indexMode) {
    case IndexMode::EVEN_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(0);
      break;
    case IndexMode::ODD_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(1);
      break;
    }
    extractStrides.back() = rewriter.getIndexAttr(2);
    extractSizes.back() = rewriter.getIndexAttr(srcType.getShape().back() / 2);

    Value deinterleaveSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newTensor, extractOffsets, extractSizes, extractStrides);

    rewriter.replaceOp(op, deinterleaveSlice);
    return success();
  }
  return failure();
};

LogicalResult
InterleaveStatusOptimization(SmallVector<Operation *> materializeVec);

LogicalResult
InterleaveStatusWithMaskOptimization(SmallVector<Operation *> materializeVec);

} // namespace triton
} // namespace mlir