//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "Analysis/MaskAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "Analysis/OpFoldResultUtils.h"

#include "Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <cstddef>

#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#define DEBUG_TYPE "triton-arith-to-linalg"

namespace mlir {

namespace triton {

LogicalResult MaskState::parse(Value operand, const Location loc,
                               OpBuilder &builder) {
  if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return this->parseConstant(op, loc, builder);
  } else if (isa<IntegerType>(operand.getType())) {
    return this->parseIntScalar(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return this->parseAdd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    return this->parseAnd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
    return this->parseCmp(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return this->parseMakeRange(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return this->parseBroadcast(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return this->parseSplat(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return this->parseExpandDims(op, loc, builder);
  } else if (!operand.getDefiningOp()) {
    return this->parseLoopIterArg(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return this->parseExtSI(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return this->parseRemsi(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    return this->parseDivsi(op, loc, builder);
  } else {
    llvm::dbgs() << "\033[31m" << "MaskAnalysis: compare operand produced by an "
                    "unsupported operation\n" << "\033[0m";

    return failure();
  }
}

tensor::ExtractSliceOp
MaskState::getExtractSlice(Value source, const Location loc,
                           OpBuilder &builder) const {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));

  auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                         dims, strides);

  return builder.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets,
                                                 dims, strides);
}

memref::SubViewOp
MaskState::getSubview(Value source, const Location loc,
                      OpBuilder &builder) const {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return builder.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType),
                                            source, offsets, dims, strides);
}

static memref::SubViewOp createSubview(Value src, Location loc, OpBuilder &b,
                                       ArrayRef<OpFoldResult> offsets,
                                       ArrayRef<OpFoldResult> sizes,
                                       ArrayRef<OpFoldResult> strides) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), src,
                                     offsets, sizes, strides);
}

// Assume block1 wraps around and the remainder is block2.
//
// |----------------------|
// |         |            |
// | block2  |  block1    |
// |         |            |
// |----------------------|
//
// Once we copy the chunks in order, the end result is block1 followed by
// block2.
//
//   buffer_tmp:
//
// |----------------------|
// |             |        |
// | block1      | block2 |
// |             |        |
// |----------------------|
//
// Assume we have the following subview:
//
// +++++++++++++++++-------
// +               +      |
// + subview       +      |
// +               +      |
// +++++++++++++++++-------
//
// If we simply take the subview of `buffer_tmp`, this requires an extra buffer
// to just hold the temporary result.
//
// So we can subview into block1 and block2 directly. There are 2 cases:
//   + subview only spans block1
//   + subview spans both block1 and block2, creating sv1 and sv2 (illustrated
//     below for case when we wrap around side-by-side)
//
// |----------------------------------------|
// |                                        |
// |    col2                       col1     |
// |++++++--------|          |+++++++++++++++
// | sv2 + block2 |          | block1 & sv1 +
// |++++++--------|          |+++++++++++++++
// |                                        |
// |----------------------------------------|
//
// For simplicity, assume we only wrap around side-by-side.
//
// Let (row, col1) and (row, col2) be the dimensions of block1 and block2,
// respectively.
//
// Let (rowFull, colFull), (rowView1, colView1) and (rowView2, colView2) be the
// dimensions of the full subview, sv1, and sv2, respectively.
//
// + colView1 = min(colFull, col1)
// + colView2 = colFull - colView1
// + rowView1 = rowView2 = row = rowFull
std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getSideBySideSubviews(Value block1, Value block2, const Location loc,
                                 OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult col1 = builder.create<memref::DimOp>(loc, block1, 1).getResult();
  OpFoldResult subviewCol1 = minOFRs(col1, subviewColFull, loc, builder);
  OpFoldResult subviewCol2 = subOFRs(subviewColFull, subviewCol1, loc, builder);

  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sv1 = createSubview(block1, loc, builder, offsets,
                           {subviewRowFull, subviewCol1}, strides);
  auto sv2 = createSubview(block2, loc, builder, offsets,
                           {subviewRowFull, subviewCol2}, strides);

  return {sv1, sv2};
}

std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getStackedSubviews(Value block1, Value block2, const Location loc,
                              OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult row1 = builder.create<memref::DimOp>(loc, block1, 0).getResult();
  OpFoldResult subviewRow1 = minOFRs(row1, subviewRowFull, loc, builder);
  OpFoldResult subviewRow2 = subOFRs(subviewRowFull, subviewRow1, loc, builder);

  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sv1 = createSubview(block1, loc, builder, offsets,
                           {subviewRow1, subviewColFull}, strides);
  auto sv2 = createSubview(block2, loc, builder, offsets,
                           {subviewRow2, subviewColFull}, strides);
  return {sv1, sv2};
}

LogicalResult MaskState::addStateScalar(const MaskState &state,
                                        const OpFoldResult scalar, Location loc,
                                        OpBuilder &builder) {
  start = addOFRs(state.start, scalar, loc, builder);
  end = addOFRs(state.end, scalar, loc, builder);
  dims = state.dims;
  stateInfo = state.stateInfo;
  return success();
}

LogicalResult MaskState::addStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc) << "Unexpected case where both lhs and rhs are scalars";
    return failure();
  }

  if (!lhsState.scalar && !rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unsupported scenario where neither lhs nor rhs is a scalar";
    return failure();
  }

  if (lhsState.scalar)
    return addStateScalar(rhsState, lhsState.scalar, loc, builder);
  else
    return addStateScalar(lhsState, rhsState.scalar, loc, builder);
}

LogicalResult MaskState::minStates(MaskState &lhsState,
                                   MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  // if (lhsState.getRank() != rhsState.getRank()) {
  //   InFlightDiagnostic diag =
  //       emitError(loc)
  //       << "Unexpected case where lhs and rhs have different ranks";
  //   return failure();
  // }
  // TODO 测试用
  // llvm::dbgs() << "\033[34m" << "lhsState.rank = " << lhsState.getRank() << "\n\033[0m";
  // llvm::dbgs() << "\033[34m" << "rhsState.rank = " << rhsState.getRank() << "\n\033[0m";
  // llvm::dbgs() << "\033[34m" << "lhsState.stateInfo.size() = " << lhsState.stateInfo.size() << "\n\033[0m";
  // llvm::dbgs() << "\033[34m" << "rhsState.stateInfo.size() = " << rhsState.stateInfo.size() << "\n\033[0m";
  for (uint32_t i = 0; i < lhsState.stateInfo.size(); i++) {
    for(uint32_t j = 0; j < rhsState.stateInfo.size(); j++){
      if(lhsState.stateInfo[i] == rhsState.stateInfo[j]){
        auto lhsDim = lhsState.dims[i];
        auto rhsDim = rhsState.dims[j];
        lhsState.dims[i] = minOFRs(lhsDim, rhsDim, loc, builder);
        rhsState.stateInfo[j].isRealDim = false;
      }
    }
  }
  for(size_t i = 0; i < lhsState.stateInfo.size(); i++){
    if(lhsState.stateInfo[i].isRealDim){
      this->dims.push_back(lhsState.dims[i]);
      this->stateInfo.push_back(lhsState.stateInfo[i]);
    }
  }
  for(size_t i = 0; i < rhsState.stateInfo.size(); i++){
    if(rhsState.stateInfo[i].isRealDim){
      this->dims.push_back(rhsState.dims[i]);
      this->stateInfo.push_back(rhsState.stateInfo[i]);
    }
  }
  return success();
}

LogicalResult MaskState::parseConstant(arith::ConstantOp constOp,
                                       const Location loc, OpBuilder &builder) {
  assert(this->isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    assert(attr.isSplat() && isa<IntegerType>(elementType) &&
           "All elements must share a single integer constant value");
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto constAttr = builder.getIndexAttr(value.getSExtValue());
    auto op = arith::ConstantOp::materialize(builder, constAttr,
                                             builder.getIndexType(), loc);
    this->scalar = op.getValue();
  } else {
    auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
    this->scalar = builder.getIndexAttr(value);
  }

  return success();
}

LogicalResult MaskState::parseIntScalar(Value scalar, const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  auto castOp =
      builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), scalar);
  this->scalar = castOp.getResult();
  return success();
}

LogicalResult MaskState::parseAdd(arith::AddIOp addOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, builder)))
    return failure();

  return this->addStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseAnd(arith::AndIOp andOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(andOp.getLhs(), loc, builder)))
    return failure();
  MaskState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, builder)))
    return failure();
  return this->minStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseExtSI(arith::ExtSIOp op, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());
  return parse(op.getIn(), loc, builder);
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  if (cmpOp.getPredicate() != arith::CmpIPredicate::slt &&
      cmpOp.getPredicate() != arith::CmpIPredicate::ult &&
      cmpOp.getPredicate() != arith::CmpIPredicate::sge &&
      cmpOp.getPredicate() != arith::CmpIPredicate::uge) {
    InFlightDiagnostic diag = emitError(loc) << "Unsupported cmpi";
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(cmpOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(cmpOp.getRhs(), loc, builder)))
    return failure();

  assert((!lhsState.scalar && rhsState.scalar) && "Unsupported cmpi scenario");

  int32_t cmpDim = -1;
  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    auto dimIntAttr = getIntAttr(lhsState.dims[i]);
    if (!dimIntAttr || dimIntAttr.value() != 1) {
      if (cmpDim != -1) {
        InFlightDiagnostic diag = emitError(loc)
                                  << "Unsupported cmpi with more than one "
                                     "dimension with size larger than 1";
        return failure();
      }
      cmpDim = i;
    }
  }
  assert(cmpDim != -1 &&
         "Unexpected case where no dimension has size larger than 1");

// Important:
  // In the case where the values we are loading are entirely masked off like
  // the following:
  //
  // ---|-------|-----------|
  //    ^       ^           ^
  //   scalar  start       end
  //
  // newEnd = min(end, scalar) = scalar
  // Now scalar < start, so simply doing dim = newEnd - start is incorrect.
  //
  // The correct formula is to optionally move `newDim` back to `start` using
  // max(newEnd, start).
  OpFoldResult newDim;
  bool isSlt;
  if (cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
     cmpOp.getPredicate() == arith::CmpIPredicate::ult){
    auto newEnd = minOFRs(lhsState.end, rhsState.scalar, loc, builder);
    newEnd = maxOFRs(newEnd, lhsState.start, loc, builder);
    newDim = subOFRs(newEnd, lhsState.start, loc, builder);
    isSlt = true;
  } else {
    auto newStart = maxOFRs(lhsState.start, rhsState.scalar, loc, builder);
    newStart = minOFRs(newStart, lhsState.end, loc, builder);
    newDim = subOFRs(lhsState.end, newStart, loc, builder);
    isSlt = false;
  }

  this->stateInfo = lhsState.stateInfo;

  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    if (i == cmpDim){
      this->dims.push_back(newDim);
      this->stateInfo[i].isSlt = isSlt;
    }
    else
      this->dims.push_back(lhsState.dims[i]);
  }
  return success();
}

LogicalResult MaskState::parseLoopIterArg(Value v, const Location loc,
                                          OpBuilder &builder) {
  assert(!v.getDefiningOp());

  auto forOp = llvm::dyn_cast<scf::ForOp>(v.getParentRegion()->getParentOp());

  if (!forOp) {
    return failure();
  }

  // TODO: This implementation does not work with nested loops
  if (forOp->getParentOfType<scf::ForOp>()) {
    return failure();
  }

  auto it = llvm::find(forOp.getRegionIterArgs(), v);
  if (it == forOp.getRegionIterArgs().end()) {
    return failure();
  }

  auto argIndex = std::distance(forOp.getRegionIterArgs().begin(), it);
  auto initArg = forOp.getInitArgs()[argIndex];
  if (auto getStateOp = initArg.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto tritonValue = getStateOp->getOperand(0);
    MaskState lhsState;
    if (failed(lhsState.parse(tritonValue, loc, builder))) {
      return failure();
    }

    // This is a bit of a hack!!
    //
    // The offsets and dimensions of a MaskState can now depend on a loop's
    // iter-arg.
    //
    // Because the PtrAnalysis's pre-pass already sets up the offsets,
    // we can create a new MaskState for each loop iteration by adding the
    // original MaskState with the current iter-arg, which is at `argIndex +
    // 1`.
    //
    // This will not work for nested loop scenarios, which would need a
    // more robust implementation.
    if (failed(this->addStateScalar(
            lhsState, forOp.getRegionIterArgs()[argIndex + 1], loc, builder))) {
      return failure();
    }

    return success();
  }

  return failure();
}

LogicalResult MaskState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "stride must be 1 for make_range whose result is used "
           "as load or store masks";
    return failure();
  }

  this->start = builder.getIndexAttr(start);
  this->end = builder.getIndexAttr(end);
  this->dims.push_back(builder.getIndexAttr(shape[0]));
  this->stateInfo.push_back(dimInfo(builder.getIndexAttr(0),
                                    builder.getIndexAttr(0)));
  this->stateInfo[0].isRealDim = true;

  return success();
}

LogicalResult MaskState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (failed(parse(src, loc, builder)))
    return failure();

  // for (size_t i = 0; i < srcShape.size(); i++) {
  //   if (srcShape[i] == dstShape[i])
  //     continue;
  //   else if (srcShape[i] < dstShape[i])
  //     this->dims[i] = builder.getIndexAttr(dstShape[i]);
  //   else
  //     llvm_unreachable("unexpected dimensions used in broadcast");
  // }

  return success();
}

LogicalResult MaskState::parseSplat(triton::SplatOp splatOp, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());
  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (!isa<IntegerType>(src.getType())) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "splat source must be an integer scalar for load/store masks";
    return failure();
  }

  if (failed(this->parse(src, loc, builder)))
    return failure();

  for (auto s : dstShape)
    this->dims.push_back(builder.getIndexAttr(s));
  
  return success();
}

LogicalResult MaskState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);

  if (failed(this->parse(expandDimsOp.getSrc(), loc, builder)))
    return failure();

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();
  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");
  // this->dims.insert(this->dims.begin() + axis, builder.getIndexAttr(1));
  for(auto &info: stateInfo)  if(info.dim >= axis)  ++info.dim;

  return success();
}

LogicalResult MaskState::parseRemsi(arith::RemSIOp remsiOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);

  MaskState lhsState;
  if (failed(lhsState.parse(remsiOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(remsiOp.getRhs(), loc, builder)))
    return failure();

  if(lhsState.scalar || !rhsState.scalar){
    remsiOp->emitRemark("Unsupported remsi scenario");
    return failure();
  }

  int64_t staticShape;

  if(auto value = rhsState.scalar.dyn_cast<Value>()){
    auto constop = value.getDefiningOp<arith::ConstantOp>();
    staticShape = cast<IntegerAttr>(constop.getValue()).getInt();
  }else if(auto rhsIntAttr = getIntAttr(rhsState.scalar)){
    assert(rhsIntAttr.has_value());
    staticShape = rhsIntAttr.value();
  }else{
    remsiOp->emitError("MaskAnalysis: Static compilation cannot determine the value of this parameter");
    return failure();
  }

  start = lhsState.start;
  dims = lhsState.dims;
  end = minOFRs(lhsState.end, rhsState.scalar, loc, builder);
  stateInfo = lhsState.stateInfo;
  for(auto &info: stateInfo){
    if(info.isRealDim){
      auto staticDim = getIntAttr(rhsState.scalar);
      if(!staticDim.has_value() || (staticDim.value() % staticShape != 0 && staticShape % staticDim.value() != 0)){
        remsiOp->emitError("MaskAnalysis: The shape of the mask is not divisible by the shape of the block");
        return failure();
      }
      // if(getIntAttr(info.div).has_value() && getIntAttr(info.div).value() != 0){
      //   remsiOp->emitError("MaskAnalysis: do not support remsi after div");
      //   return failure();
      // }
      info.shape = builder.getIndexAttr(staticShape);
    }
  }

  return success();
}

LogicalResult MaskState::parseDivsi(arith::DivSIOp divsiOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  auto defaultAttr = builder.getIndexAttr(0);

  MaskState lhsState;
  if (failed(lhsState.parse(divsiOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(divsiOp.getRhs(), loc, builder)))
    return failure();

  if(lhsState.scalar || !rhsState.scalar){
    divsiOp->emitRemark("Unsupported divsi scenario");
    return failure();
  }

  int64_t staticDiv;

  if(auto value = rhsState.scalar.dyn_cast<Value>()){
    auto constop = value.getDefiningOp<arith::ConstantOp>();
    staticDiv = cast<IntegerAttr>(constop.getValue()).getInt();
  }else if(auto rhsIntAttr = getIntAttr(rhsState.scalar)){
    assert(rhsIntAttr.has_value());
    staticDiv = rhsIntAttr.value();
  }else{
    divsiOp->emitError("MaskAnalysis: Static compilation cannot determine the value of this parameter");
    return failure();
  }

  start = divOFRs(lhsState.start, rhsState.scalar, loc, builder);
  dims = lhsState.dims;
  auto minEnd = addOFRs(start, builder.getIndexAttr(1), loc, builder);
  end = subOFRs(lhsState.end, builder.getIndexAttr(1), loc, builder);
  end = divOFRs(end, rhsState.scalar, loc, builder);
  end = addOFRs(end, builder.getIndexAttr(1), loc, builder);
  end = maxOFRs(end, minEnd, loc, builder);
  stateInfo = lhsState.stateInfo;
  for(auto &info: stateInfo){
    if(info.isRealDim){
      auto staticDim = getIntAttr(rhsState.scalar);
      if(!staticDim.has_value() || (staticDim.value() % staticDiv != 0 && staticDiv % staticDim.value() != 0)){
        divsiOp->emitError("MaskAnalysis: The shape of the mask is not divisible by the shape of the block");
        return failure();
      }
      if(getIntAttr(info.shape).has_value() && getIntAttr(info.shape).value() != 0){
        divsiOp->emitError("MaskAnalysis: do not support div after remsi");
        return failure();
      }
      info.div = builder.getIndexAttr(staticDiv);
    }
  }

  return success();
}

void MaskState::eraseInsertedOps(Operation *rawOp, PatternRewriter &rewriter) {
  auto moduleOp = rawOp->getParentOfType<ModuleOp>();
  SmallVector<Operation *> worklist;
  moduleOp->walk([&](Operation *op) {
    if (isOpTriviallyDead(op))
      worklist.push_back(op);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    for (Value value : op->getOperands()) {
      if (auto defOp = value.getDefiningOp())
        worklist.push_back(defOp);
    }
    LLVM_DEBUG({
      llvm::dbgs() << "[MaskState]==> inserted op: \n"
                   << *op << "\n[MaskState]<== is removed\n";
    });
    rewriter.eraseOp(op);
  }
}

tensor::InsertSliceOp MaskState::getInsertSlice(Value source, Value dest,
                                                const Location &loc,
                                                OpBuilder &builder) const {
  auto sourceType = cast<RankedTensorType>(source.getType());
  //kaixin, fixme, offsets are class member originally 
  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  return builder.create<tensor::InsertSliceOp>(loc, source, dest, offsets, dims,
                                               strides);
}

} // namespace triton
} // namespace mlir
