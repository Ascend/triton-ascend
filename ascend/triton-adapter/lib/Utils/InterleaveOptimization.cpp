//===- InterleaveOptimization.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <utility>

namespace mlir {
namespace triton {
// For origin MemRefType of ReinterpretCastOp under interleave state, here wanna
// adjust its shape info by expanding last dimension double.
MemRefType expandInterleaveMemRefType(MemRefType originType) {
  // Double the last dimension shape
  SmallVector<int64_t> shape(originType.getShape());
  shape.back() = shape.back() * 2;

  // Adjuest layout attribute
  StridedLayoutAttr originLayout =
      llvm::dyn_cast<StridedLayoutAttr>(originType.getLayout());
  // If offset is static, just reset it to 0
  auto offset = originLayout.getOffset() == ShapedType::kDynamic
                    ? originLayout.getOffset()
                    : 0;
  // Set last dimension stride to 1
  SmallVector<int64_t> stride(originLayout.getStrides());
  stride.back() = 1;

  return MemRefType::get(
      shape, originType.getElementType(),
      StridedLayoutAttr::get(originType.getContext(), offset, stride));
}

// *********************
// **      NOTE       **
// *********************
// How to determine new offset is a little tricky and specific
// Here just consider this state in triton language:
//
// dim_range = tl.arange(0, BLOCK // 2)
// last_dim_even_range = dim_range * 2
// last_dim_odd_range = dim_range * 2 + 1
//
// Here `multiply two` represents that last dimension stride is 2, and
// `add constant one` represents whether it's odd index part of
// deinterleave result.
//
// Therefore, how to distinguish interleave/deinterleave on even index or odd
// index is whether last dimension range explicitly `add constant one` without
// any other operation. In IR it's shown that whether defining op of
// `castOffset` is an arith::addOp, as this arith::addOp would contain above
// `add constant one` opeartion after LegacyAddPtrConverter.
//
// Well, index mode should be passed to interleave/deinterleave, in other words,
// `add constant one` should work on offset of next insert_slice/extract_slic.
// The new reinterpretcast just wanna describe whole tensor, so new castOffset
// is just from non-last diemsnion accumulation and remove `add constant one`
std::pair<OpFoldResult, IndexMode>
recountReinterpretCastOffset(OpFoldResult originOffset, Builder &builder) {
  // To trace value type offset
  std::function<bool(Operation *)> traceOffset = [&](Operation *op) -> bool {
    // Consider constant one in `add constant one` operation
    if (llvm::isa<arith::ConstantOp>(op))
      return false;

    if (llvm::isa<arith::AddIOp>(op)) {
      auto addOp = llvm::cast<arith::AddIOp>(op);
      if (auto constLHS = addOp.getLhs().getDefiningOp<arith::ConstantOp>()) {
        assert(dyn_cast<IntegerAttr>(constLHS.getValueAttr()).getInt() == 1 &&
               "Arith::constant value of addi's operand must be 1 when "
               "calculate deinterleave offset");
        return false;
      }
      if (auto constRHS = addOp.getRhs().getDefiningOp<arith::ConstantOp>()) {
        assert(dyn_cast<IntegerAttr>(constRHS.getValueAttr()).getInt() == 1 &&
               "Arith::constant value of addi's operand must be 1 when "
               "calculate deinterleave offset");
        return false;
      }
    }
    return true;
  };

  IndexMode evenOrOdd = IndexMode::EVEN_MODE;
  // Reuse origin offset if there's no 'add constant one'
  OpFoldResult newOffset = originOffset;
  if (llvm::isa<Attribute>(originOffset)) {
    // If offset is constant int(IndexAttr),
    // the int value could only be 0 or 1
    int64_t intOffset = getConstantIntValue(originOffset).value();
    assert((intOffset == 0 || intOffset == 1));
    if (intOffset == 1) {
      evenOrOdd = IndexMode::ODD_MODE;
      newOffset = builder.getIndexAttr(0);
    }
  } else if (llvm::isa<Value>(originOffset)) {
    if (!traceOffset(originOffset.get<Value>().getDefiningOp())) {
      evenOrOdd = IndexMode::ODD_MODE;
      Operation *traceResult = findFirstMatchingOperandDef(
          originOffset.get<Value>().getDefiningOp(), traceOffset);
      assert(traceResult->getNumResults() == 1 &&
             "Offset defining operation must have one result");
      newOffset = traceResult->getResult(0);
    }
  }

  return {newOffset, evenOrOdd};
}


LogicalResult
InterleaveStatusOptimization(SmallVector<Operation *> materializeVec) {
  OpBuilder builder(materializeVec[1]);
  auto loc = materializeVec[1]->getLoc();

  auto firstReinterpretCastOp =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[0])
          .getDest()
          .getDefiningOp<memref::ReinterpretCastOp>();
  auto secondReinterpretCastOp =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[1])
          .getDest()
          .getDefiningOp<memref::ReinterpretCastOp>();

  assert(firstReinterpretCastOp && secondReinterpretCastOp);

  // Judge whether two `ReinterpretCastOp` shape satisfy interleave state
  // a. both size are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedSizes(),
          secondReinterpretCastOp.getConstifiedMixedSizes())) {
    return failure();
  }
  // b. both strides are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedStrides(),
          secondReinterpretCastOp.getConstifiedMixedStrides())) {
    return failure();
  }
  // c. both offsets should satisfy tricky rule
  auto firstOriginCastOffset =
      firstReinterpretCastOp.getConstifiedMixedOffset();
  auto secondOriginCastOffset =
      secondReinterpretCastOp.getConstifiedMixedOffset();
  std::pair<IndexMode, IndexMode> indexModeRecord;
  OpFoldResult newCastOffset;
  if (llvm::isa<Attribute>(firstOriginCastOffset) &&
      llvm::isa<Attribute>(secondOriginCastOffset)) {
    auto [firstCastOffset, firstIndexMode] =
        recountReinterpretCastOffset(firstOriginCastOffset, builder);
    auto [secondCastOffset, secondIndexMode] =
        recountReinterpretCastOffset(secondOriginCastOffset, builder);

    if (!(static_cast<int>(firstIndexMode) ^ static_cast<int>(secondIndexMode)))
      return failure();
    newCastOffset = builder.getIndexAttr(0);
    indexModeRecord = {firstIndexMode, secondIndexMode};

  } else if (llvm::isa<Value>(firstOriginCastOffset) &&
             llvm::isa<Value>(secondOriginCastOffset)) {
    auto [firstCastOffset, firstIndexMode] =
        recountReinterpretCastOffset(firstOriginCastOffset, builder);
    auto [secondCastOffset, secondIndexMode] =
        recountReinterpretCastOffset(secondOriginCastOffset, builder);

    if (!(static_cast<int>(firstIndexMode) ^
          static_cast<int>(secondIndexMode)) ||
        (llvm::dyn_cast<Value>(firstCastOffset) !=
         llvm::dyn_cast<Value>(secondCastOffset)))
      return failure();

    if (firstIndexMode == IndexMode::EVEN_MODE) {
      newCastOffset = llvm::dyn_cast<Value>(firstCastOffset);
    }
    if (secondIndexMode == IndexMode::EVEN_MODE) {
      newCastOffset = llvm::dyn_cast<Value>(secondCastOffset);
    }
    indexModeRecord = {firstIndexMode, secondIndexMode};

  } else {
    return failure();
  }

  // Create new op
  // 1. Get new destination memref type
  auto dstType = expandInterleaveMemRefType(firstReinterpretCastOp.getType());

  // 2. New tensor::EmptyOp
  auto emptyTensor = builder.create<tensor::EmptyOp>(loc, dstType.getShape(),
                                                     dstType.getElementType());

  // 3. New insert_slice from materialization source into new empty tensor
  SmallVector<OpFoldResult> insertOffsets(dstType.getRank(),
                                          builder.getIndexAttr(0));
  SmallVector<OpFoldResult> insertStrides(dstType.getRank(),
                                          builder.getIndexAttr(1));
  SmallVector<OpFoldResult> insertSizes = llvm::to_vector(
      llvm::map_range(dstType.getShape(), [&](int64_t dim) -> OpFoldResult {
        return builder.getIndexAttr(dim);
      }));
  insertStrides.back() = builder.getIndexAttr(2);
  insertSizes.back() = builder.getIndexAttr(dstType.getShape().back() / 2);
  if (indexModeRecord.first == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertFirst = builder.create<tensor::InsertSliceOp>(
      loc,
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[0])
          .getSource(),
      emptyTensor.getResult(), insertOffsets, insertSizes, insertStrides);

  if (indexModeRecord.second == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertSecond = builder.create<tensor::InsertSliceOp>(
      loc,
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[1])
          .getSource(),
      insertFirst.getResult(), insertOffsets, insertSizes, insertStrides);

  // 4. Reinterpret_cast block arg
  auto newCastSize = firstReinterpretCastOp.getConstifiedMixedSizes();
  auto newCastStride = firstReinterpretCastOp.getConstifiedMixedStrides();
  newCastSize.back() = builder.getIndexAttr(dstType.getShape().back());
  newCastStride.back() = builder.getIndexAttr(1);
  auto newCastOp = builder.create<memref::ReinterpretCastOp>(
      loc, dstType, firstReinterpretCastOp.getViewSource(), newCastOffset,
      newCastSize, newCastStride);

  // 5. Create new bufferization::MaterializeInDestinationOp
  auto newStoreOp = builder.create<bufferization::MaterializeInDestinationOp>(
      loc, insertSecond.getResult(), newCastOp.getResult());
  // Setting writable is necessary as dst is memref type
  newStoreOp.setWritable(true);

  // 6. Erase origin materialization
  materializeVec[0]->erase();
  materializeVec[1]->erase();

  return success();
}

LogicalResult
InterleaveStatusWithMaskOptimization(SmallVector<Operation *> materializeVec) {
  OpBuilder builder(materializeVec[1]);

  auto firstSubviewOpOfReCast =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[0])
          .getDest()
          .getDefiningOp<memref::SubViewOp>();
  auto firstSrcExtractSlice =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[0])
          .getSource()
          .getDefiningOp<tensor::ExtractSliceOp>();
  auto firstReinterpretCastOp = firstSubviewOpOfReCast.getSource()
                                    .getDefiningOp<memref::ReinterpretCastOp>();

  auto secondSubviewOpOfReCast =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[1])
          .getDest()
          .getDefiningOp<memref::SubViewOp>();
  auto secondSrcExtractSlice =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[1])
          .getSource()
          .getDefiningOp<tensor::ExtractSliceOp>();
  auto secondReinterpretCastOp =
      secondSubviewOpOfReCast.getSource()
          .getDefiningOp<memref::ReinterpretCastOp>();

  // 1. Both source shapes of subview and extract_slice are equal
  if (firstSubviewOpOfReCast.getSourceType().getShape() !=
      firstSrcExtractSlice.getSourceType().getShape())
    return failure();
  if (secondSubviewOpOfReCast.getSourceType().getShape() !=
      secondSrcExtractSlice.getSourceType().getShape())
    return failure();
  if (firstSubviewOpOfReCast.getSourceType().getShape() !=
      secondSubviewOpOfReCast.getSourceType().getShape())
    return failure();

  // 2. both mask state are equal
  std::function<bool(OpFoldResult, OpFoldResult)> cmpFunc =
      mlir::isEqualConstantIntOrValue;
  if (!mlir::detail::sameOffsetsSizesAndStrides(firstSubviewOpOfReCast,
                                                firstSrcExtractSlice, cmpFunc))
    return failure();
  if (!mlir::detail::sameOffsetsSizesAndStrides(secondSubviewOpOfReCast,
                                                secondSrcExtractSlice, cmpFunc))
    return failure();
  if (!mlir::detail::sameOffsetsSizesAndStrides(
          firstSubviewOpOfReCast, secondSubviewOpOfReCast, cmpFunc))
    return failure();

  // 3. Still judge whether two `ReinterpretCastOp` shape satisfy request
  // a. both size are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedSizes(),
          secondReinterpretCastOp.getConstifiedMixedSizes()))
    return failure();
  // b. both strides are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedStrides(),
          secondReinterpretCastOp.getConstifiedMixedStrides()))
    return failure();
  // c. both offsets should satisfy tricky rule
  auto firstOriginCastOffset =
      firstReinterpretCastOp.getConstifiedMixedOffset();
  auto secondOriginCastOffset =
      secondReinterpretCastOp.getConstifiedMixedOffset();
  std::pair<IndexMode, IndexMode> indexModeRecord;
  OpFoldResult newCastOffset;
  if (llvm::isa<Attribute>(firstOriginCastOffset) &&
      llvm::isa<Attribute>(secondOriginCastOffset)) {
    auto [firstCastOffset, firstIndexMode] =
        recountReinterpretCastOffset(firstOriginCastOffset, builder);
    auto [secondCastOffset, secondIndexMode] =
        recountReinterpretCastOffset(secondOriginCastOffset, builder);

    if (!(static_cast<int>(firstIndexMode) ^ static_cast<int>(secondIndexMode)))
      return failure();
    newCastOffset = builder.getIndexAttr(0);
    indexModeRecord = {firstIndexMode, secondIndexMode};

  } else if (llvm::isa<Value>(firstOriginCastOffset) &&
             llvm::isa<Value>(secondOriginCastOffset)) {
    auto [firstCastOffset, firstIndexMode] =
        recountReinterpretCastOffset(firstOriginCastOffset, builder);
    auto [secondCastOffset, secondIndexMode] =
        recountReinterpretCastOffset(secondOriginCastOffset, builder);

    auto offsetsAreEqual = [](Value firstCastOffset, Value secondCastOffset) {
      auto equal = (llvm::dyn_cast<Value>(firstCastOffset) == llvm::dyn_cast<Value>(secondCastOffset));
      if (equal)
          return equal; 
      auto firstOp = firstCastOffset.getDefiningOp();
      auto secondOp = secondCastOffset.getDefiningOp();
      // equal = (firstOp->getLhs() == secondOp->getLhs()) && (firstOp->getRhs() == secondOp->getRhs()) ;
      if (equal)
          return equal;
      if(llvm::isa<arith::AddIOp>(firstOp) && llvm::isa<arith::AddIOp>(secondOp)){
        auto addOp1 = llvm::cast<arith::AddIOp>(firstOp);
        auto addOp2 = llvm::cast<arith::AddIOp>(secondOp);

        equal = (addOp1.getLhs() == addOp2.getLhs() && addOp1.getRhs() == addOp2.getRhs())  || \
            (addOp1.getLhs() == addOp2.getRhs() && addOp1.getRhs() == addOp2.getLhs()) ;
      }
      return equal ;
    };

    if (!(static_cast<int>(firstIndexMode) ^ static_cast<int>(secondIndexMode)) || 
          !offsetsAreEqual(llvm::dyn_cast<Value>(firstCastOffset), llvm::dyn_cast<Value>(secondCastOffset)))
    {
      llvm::dbgs() << "firstOriginCastOffset:" << firstOriginCastOffset << " firstCastOffset:" <<  firstCastOffset << "\n";
      llvm::dbgs() << "secondOriginCastOffset:" << secondOriginCastOffset << " secondCastOffset:" <<  secondCastOffset << "\n";
      return failure();
    }

    if (firstIndexMode == IndexMode::EVEN_MODE) {
      newCastOffset = llvm::dyn_cast<Value>(firstCastOffset);
    }
    if (secondIndexMode == IndexMode::EVEN_MODE) {
      newCastOffset = llvm::dyn_cast<Value>(secondCastOffset);
    }
    indexModeRecord = {firstIndexMode, secondIndexMode};

  } else {
    return failure();
  }
  auto loc = materializeVec[1]->getLoc();

  // Create new op
  // 1. Get new destination memref type
  auto dstType = expandInterleaveMemRefType(firstReinterpretCastOp.getType());

  // 2. New tensor::EmptyOp
  auto emptyTensor = builder.create<tensor::EmptyOp>(loc, dstType.getShape(),
                                                     dstType.getElementType());

  // 3. New insert_slice from extract_slice source into new empty tensor
  SmallVector<OpFoldResult> insertOffsets(dstType.getRank(),
                                          builder.getIndexAttr(0));
  SmallVector<OpFoldResult> insertStrides(dstType.getRank(),
                                          builder.getIndexAttr(1));
  SmallVector<OpFoldResult> insertSizes = llvm::to_vector(
      llvm::map_range(dstType.getShape(), [&](int64_t dim) -> OpFoldResult {
        return builder.getIndexAttr(dim);
      }));
  insertStrides.back() = builder.getIndexAttr(2);
  insertSizes.back() = builder.getIndexAttr(dstType.getShape().back() / 2);
  if (indexModeRecord.first == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertFirst = builder.create<tensor::InsertSliceOp>(
      loc, firstSrcExtractSlice.getSource(), emptyTensor.getResult(),
      insertOffsets, insertSizes, insertStrides);

  if (indexModeRecord.second == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertSecond = builder.create<tensor::InsertSliceOp>(
      loc, secondSrcExtractSlice.getSource(), insertFirst.getResult(),
      insertOffsets, insertSizes, insertStrides);

  // 4. To enable store with mask, create new extract_slice
  SmallVector<OpFoldResult> extractOffsets =
      firstSrcExtractSlice.getMixedOffsets();
  SmallVector<OpFoldResult> extractStrides =
      firstSrcExtractSlice.getMixedStrides();
  SmallVector<OpFoldResult> extractSizes = firstSrcExtractSlice.getMixedSizes();
  assert(llvm::isa<Attribute>(extractSizes.back()));
  extractSizes.back() = builder.getIndexAttr(
      getConstantIntValue(extractSizes.back()).value() * 2);
  auto newSrcExtractSlice = builder.create<tensor::ExtractSliceOp>(
      loc, insertSecond.getResult(), extractOffsets, extractSizes,
      extractStrides);

  // 5. Reinterpret_cast block arg
  auto newCastSize = firstReinterpretCastOp.getConstifiedMixedSizes();
  auto newCastStride = firstReinterpretCastOp.getConstifiedMixedStrides();
  newCastSize.back() = builder.getIndexAttr(dstType.getShape().back());
  newCastStride.back() = builder.getIndexAttr(1);
  auto newCastOp = builder.create<memref::ReinterpretCastOp>(
      loc, dstType, firstReinterpretCastOp.getViewSource(), newCastOffset,
      newCastSize, newCastStride);

  // 6. Create new memref::SubViewOp of above new reinterpret_cast
  // Here could reuse shape info of new extract_slice
  auto dstSubviewType = memref::SubViewOp::inferResultType(
      dstType, extractOffsets, extractSizes, extractStrides);
  auto newSubviewOpOfReCast = builder.create<memref::SubViewOp>(
      loc, llvm::cast<MemRefType>(dstSubviewType), newCastOp, extractOffsets,
      extractSizes, extractStrides);

  // 7. Create new bufferization::MaterializeInDestinationOp
  auto newStoreOp = builder.create<bufferization::MaterializeInDestinationOp>(
      loc, newSrcExtractSlice.getResult(), newSubviewOpOfReCast.getResult());
  // Setting writable is necessary as dst is memref type
  newStoreOp.setWritable(true);

  // 8. Erase origin operation
  materializeVec[0]->erase();
  materializeVec[1]->erase();
  firstSubviewOpOfReCast->erase();
  firstSrcExtractSlice->erase();
  secondSubviewOpOfReCast->erase();
  secondSrcExtractSlice->erase();

  return success();
}

} // namespace triton
} // namespace mlir