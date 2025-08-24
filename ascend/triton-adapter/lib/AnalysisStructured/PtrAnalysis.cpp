//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "AnalysisStructured/PtrAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "Analysis/MaskAnalysis.h"
#include "Analysis/OpFoldResultUtils.h"
#include "Utils/Utils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <queue>
#include <string>

#define DEBUG_TYPE "triton-ptr-analysis"

namespace mlir {

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
static Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return builder.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return builder.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            builder, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

namespace tts {

int32_t PtrState::getRank() const {
  return stateInfo.size();
}

bool PtrState::isLegal() const {
  return !stateInfo.empty() || scalar || source;
}

bool PtrState::isSameSizeAs(const PtrState& x) const {
  if(sizes.size() != x.sizes.size())
    return false;

  for(size_t i = 0; i < sizes.size(); ++i){
    if(sizes[i] != x.sizes[i])
      return false;
  }
  return true;
}

MemAccType PtrState::getMemAccType() const { return this->memAccTy; };
MemAccType &PtrState::getMemAccTypeRef() { return this->memAccTy; };

bool PtrState::shouldRemove(const StateInfo& x) const {
    auto staticMask = getIntAttr(x.mask);
    auto staticStride = getIntAttr(x.stride);
    auto staticShape = getIntAttr(x.shape);
    auto staticSize = getIntAttr(sizes[x.dim]);

    // Constant Dimension: For example, 'xindex % 1024 + 4096', the number 4096
    // is a fixed value. It serves as a static offset or component in the calculatio
    if(staticMask.has_value() && staticStride.has_value() && staticShape.has_value() &&
       staticMask.value() == 0 && staticStride.value() == 0 && staticShape.value() == 0){

        return true;

    // When a dimension is divided by a number that is a positive integer multiple of
    // each read, it effectively acts as an offset. For example, in the expression
    // '(xindex + id * Xblock) / 8192', if Xblock is 512, then all values of
    // 'xindex / 8192' in this tensor will be the same. In this case, it is equivalent
    // to an offset.
    }
    // else if(staticMask.has_value() && staticSize.has_value() &&
    //          staticMask.value() % staticSize.value() == 0 &&
    //          staticMask.value() != 0
    //          ){
    //       return true;
    // }

    return false;
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

bool PtrState::hasModulo() const {
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimHasModulo(i)) {
      return true;
    }
  }
  return false;
}

bool PtrState::hasBroadcast() const {
  for(auto x : stateInfo){
    auto staticStride = getIntAttr(x.stride);
    // assert(staticStride.has_value() && "do not support dynamic stride");
    // if(staticStride == 0) return true;
    if(staticStride.has_value() && staticStride == 0) return true;
  }
  return false;
}

bool PtrState::hasDivision() const {
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimHasDivision(i)) {
      return true;
    }
  }
  return false;
}

bool PtrState::dimHasDivision(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check division if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(stateInfo[dim].mask);
  if (!intAttr.has_value()) {
    return true;
  }

  return intAttr.value() != 0;

}

bool PtrState::dimHasModulo(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check modulo if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(stateInfo[dim].shape);
  if (!intAttr.has_value()) {
    return true;
  }

  return intAttr.value() != 0;
}

bool PtrState::isBlockPtr() const { return !order.empty(); }

LogicalResult PtrState::broadcastIfNeeded(SmallVector<StateInfo> &infoPerDim,
                                          Operation *op, OpBuilder &builder) {
    auto loc = op->getLoc();
    auto defaultAttr = builder.getIndexAttr(0);
    auto staticSize = getIntAttr(this->sizes[infoPerDim[0].dim]);
    assert(staticSize.has_value() && "do not support dynamic size");
    int64_t readSize = 1;

    for (StateInfo info : infoPerDim) {
      auto staticShape =  getIntAttr(info.shape);
      auto staticMask = getIntAttr(info.mask);

      assert(staticShape.has_value() && staticMask.has_value() && "do not support dynamic shape/mask");

      int64_t divDim = staticMask.value() ? (staticSize.value() - 1) / staticMask.value() + 1 : staticSize.value();
      int64_t remsiDim = staticShape.value() ? staticShape.value() : staticSize.value();

      readSize *= std::min(divDim, remsiDim);
    }

    if (readSize < staticSize.value()) {
      // 补维
      int64_t brcDim = (staticSize.value() - 1) / readSize + 1;
      auto shape = builder.getIndexAttr(brcDim);
      auto mask = builder.getIndexAttr(readSize);
      StateInfo broadcasrInfo(defaultAttr, defaultAttr, shape, mask);
      infoPerDim.push_back(broadcasrInfo);
    }

    return success();
}

bool PtrState::countDims() {
  dimLenth = SmallVector<size_t>(sizes.size() + 1, 0);

  for(auto info : stateInfo){

    assert(info.dim <= stateInfo.back().dim);
    ++dimLenth[info.dim];
  }
  return true;
}

LogicalResult PtrState::removeConstDim(SmallVector<StateInfo> &infoPerDim,
                                        Operation *op, OpBuilder &builder) {
    auto loc = op->getLoc();
    auto defaultAttr = builder.getIndexAttr(0);
    OpFoldResult offsetDim = defaultAttr;

    for (auto x : infoPerDim) {
      offsetDim = addOFRs(offsetDim, x.offset, loc, builder);
      if(shouldRemove(x)){
        assert(dimLenth[x.dim] > 0);
        --dimLenth[x.dim];
      }
    }

    infoPerDim.erase(
        std::remove_if(infoPerDim.begin(), infoPerDim.end(),
                       [this](const StateInfo& x) { return shouldRemove(x); }),
        infoPerDim.end()
    );

    if(infoPerDim.empty()){
      StateInfo placeHolder(offsetDim, defaultAttr, defaultAttr, defaultAttr);
      infoPerDim.push_back(placeHolder);
    }

    for(size_t i = 0;  i < infoPerDim.size(); ++i){
      if(i == 0)  infoPerDim[i].offset = offsetDim;
      else        infoPerDim[i].offset = defaultAttr;
    }


    return success();
}

LogicalResult PtrState::ExpandInfo(SmallVector<StateInfo> &infoPerDim,
                                      Operation *op, OpBuilder &builder) {
  if(infoPerDim.size() == 0)  return success();
  SmallVector<StateInfo> insertInfo;
  SmallVector<size_t> insertPos;
  auto defaultAttr = builder.getIndexAttr(0);
  auto staticPreMask = getIntAttr(defaultAttr);
  auto staticPreShape = getIntAttr(builder.getIndexAttr(1));

  for(size_t i = 0; i < infoPerDim.size(); i++){
    auto staticMask = getIntAttr(infoPerDim[i].mask);
    assert(staticMask.has_value() && "PtrAnalysis: do not support dymic mask/size");
    auto staticShape = getIntAttr(infoPerDim[i].shape);
    auto staticSize = getIntAttr(sizes[infoPerDim[i].dim]);
    assert(staticPreMask.has_value() && staticPreShape.has_value()  &&
           staticSize.has_value() && "PtrAnalysis: do not support dymic mask/shape");

    if(staticMask.value() % staticSize.value() == 0 && staticMask != 0) continue;

    int64_t prevDimSize = staticPreShape.value() * (staticPreMask.value() == 0 ? 1 : staticPreMask.value());
    if(staticMask.value() == 0 && i != 0){
       op->emitRemark(
            "PtrAnalysis: do not support index % a + index % b in same dim without div");
        return failure();
    }
    if(staticMask.value() % prevDimSize != 0){
      op->emitError(
            "Unstructured memory access cannot be transformed into an equivalent structured memory access pattern.");
        return failure();
    }

    if(staticMask.value() / prevDimSize != 1 && staticMask.value() != 0){
      auto mask = builder.getIndexAttr(prevDimSize);
      auto shape = builder.getIndexAttr(staticMask.value() / prevDimSize);
      StateInfo preInfo(defaultAttr, defaultAttr, shape, mask);
      insertInfo.push_back(preInfo);
      insertPos.push_back(i);
    }
    staticPreMask = staticMask;
    staticPreShape = staticShape;
  }


  assert(insertInfo.size() == insertPos.size());
  for(size_t i = 0; i < insertInfo.size(); i++){
    infoPerDim.insert(infoPerDim.begin() + insertPos[i] + i, insertInfo[i]);
  }

  if(this->broadcastIfNeeded(infoPerDim, op, builder).failed()){
    return failure();
  }
  return success();
}

LogicalResult PtrState::addPtrState(const PtrState &lhsState,
                                    const PtrState &rhsState, Operation *op,
                                    OpBuilder &builder) {
  assert(isEmpty());
  auto loc = op->getLoc();

  if (lhsState.source && rhsState.source) {
    op->emitRemark(
        "PtrAnalysis: do not support adding two pointer states that both "
        "have base pointers");
    return failure();
  }

  source = lhsState.source ? lhsState.source : rhsState.source;

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;
  if (!(lhs->source && !rhs->source)) {
    std::swap(lhs, rhs);
  }

  assert(lhs->source && "Addptr must contain one pointer!");

  stateInfo = rhs->stateInfo;
  sizes = rhs->sizes;

  if(rhs->scalar){
    sizes.push_back(builder.getIndexAttr(1));
    auto defaultAttr = builder.getIndexAttr(0);
    StateInfo placeHolder(rhsState.scalar, builder.getIndexAttr(1),
                           defaultAttr, defaultAttr);
    stateInfo.push_back(placeHolder);
    dimLenth.push_back(1);
    scalar = rhs->scalar;
    if(!isa<mlir::RankedTensorType>(dyn_cast<triton::AddPtrOp>(op).getOffset().getType())){
      ptrIsTensor = false;
    }
    return success();
  }


  std::sort(stateInfo.begin(), stateInfo.end(), [](const StateInfo& a, const StateInfo& b) {
    auto staticL = getIntAttr(a.mask);
    auto staticR = getIntAttr(b.mask);
    assert(staticL.has_value() && staticR.has_value() && "PtrAnalysis: do not support dymic mask");
    return a.dim < b.dim || (a.dim == b.dim && staticL.value() < staticR.value());
  });
  this->countDims();

  assert(stateInfo.size() && "No information could be analyzed in the state");
  SmallVector<SmallVector<StateInfo>> infoInDifDim;
  size_t startIndex = 0;
  for(auto lenth : dimLenth){
    if(lenth == 0)  continue;
    infoInDifDim.push_back(SmallVector<StateInfo>(stateInfo.begin() + startIndex,
                                                  stateInfo.begin() + startIndex + lenth));
    startIndex += lenth;
  }

  for(auto &infoPerDim : infoInDifDim){
    if(this->removeConstDim(infoPerDim, op, builder).failed()){
      return failure();
    }
    if(this->ExpandInfo(infoPerDim, op, builder).failed())
      return failure();
  }

  this->stateInfo.clear();

  for(auto infoPerDim : infoInDifDim){
    std::reverse(infoPerDim.begin(), infoPerDim.end());
    for(auto info : infoPerDim){
      this->stateInfo.push_back(info);
    }
  }
  this->countDims();

  if(lhs->scalar){
    assert(getIntAttr(stateInfo.back().stride).has_value() &&
      getIntAttr(stateInfo.back().stride).value() == 1 && "lastDim stride must be 1");
    stateInfo.back().offset = addOFRs(stateInfo.back().offset, lhsState.scalar, loc, builder);
  }

  auto leftState = const_cast<PtrState&>(lhsState) ;
  auto rightState = const_cast<PtrState&>(rhsState) ;
  this->getMemAccTypeRef().merge(leftState.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rightState.getMemAccTypeRef());

  return success();
}

LogicalResult PtrState::addState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty());
  auto loc = op->getLoc();

  if(!lhsState.isLegal() || !rhsState.isLegal()){
    op->emitRemark(
        "PtrAnalysis: Pointer analysis is not supported for input parameters");
    return failure();
  }


  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) { // both lhs and rhs are scalars
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  assert(lhsState.isSameSizeAs(rhsState) && "The original size of the addition should be the same");

  if(lhsState.scalar || rhsState.scalar){
    auto scalarState = lhsState.scalar ? lhsState : rhsState;
    auto normalState = lhsState.scalar ? rhsState : lhsState;
    auto offset = normalState.stateInfo[0].offset;

    normalState.stateInfo[0].offset = addOFRs(offset, scalarState.scalar, loc, builder);
    stateInfo = normalState.stateInfo;
    sizes = normalState.sizes;

  }else if (!lhsState.hasDivision() && !lhsState.hasModulo() &&
      !rhsState.hasDivision() && !rhsState.hasModulo() &&
      lhsState.stateInfo[0].dim == rhsState.stateInfo[0].dim) {

    // assert(lhsState.getRank() == rhsState.getRank());
    if (lhsState.getRank() != rhsState.getRank()) {
      llvm::dbgs() << "\033[34m" << "lhstate中存储的维度为:"<<  lhsState.getRank() << "\n\033[0m";
      llvm::dbgs() << "\033[34m" << "rhstate中存储的维度为:"<<  rhsState.getRank() << "\n\033[0m";
      op->emitRemark(
          "PtrAnalysis: only support multiplying pointer states when one of "
          "them represent a scalar");
      return failure();
    }
    for (uint64_t i = 0; i < lhsState.getRank(); i++) {
      auto newOffset = addOFRs(lhsState.stateInfo[i].offset, rhsState.stateInfo[i].offset,
                                                  loc, builder);
      auto newStride = addOFRs(lhsState.stateInfo[i].stride, rhsState.stateInfo[i].stride,
                                                  loc, builder);
      StateInfo newStateInfo(newOffset, newStride, lhsState.stateInfo[i].shape,
                               lhsState.stateInfo[i].mask, lhsState.stateInfo[i].dim);
      this->stateInfo.push_back(newStateInfo);
    }
    this->sizes = lhsState.sizes;
  }else{
    for (uint64_t i = 0; i < lhsState.getRank(); this->stateInfo.push_back(lhsState.stateInfo[i++]));
    for (uint64_t i = 0; i < rhsState.getRank(); this->stateInfo.push_back(rhsState.stateInfo[i++]));
    this->sizes = rhsState.sizes;
  }
  auto leftState = const_cast<PtrState&>(lhsState) ;
  auto rightState = const_cast<PtrState&>(rhsState) ;
  this->getMemAccTypeRef().merge(leftState.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rightState.getMemAccTypeRef());

  return success();
}

LogicalResult PtrState::mulState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.isSameSizeAs(rhsState));

  auto loc = op->getLoc();
  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense

  if (lhsState.hasSource() && rhsState.hasSource()) {
    op->emitRemark("PtrAnalysis: do not support both sides have base inters in multiplying");
    return failure();
  }

  if(lhsState.scalar && rhsState.scalar){
    auto mulOp =
        builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    this->scalar = mulOp.getResult();
  }

  // currently do not support both tensors are effectively non-scalar
  if (!lhsState.scalar && !rhsState.scalar) {
    op->emitRemark(
        "PtrAnalysis: only support multiplying pointer states when one of "
        "them represent a scalar");
    return failure();
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->scalar && lhs->scalar) {
    std::swap(lhs, rhs);
  }

  for(auto info : lhs->stateInfo){
    // StateInfo newStateInfo;
    OpFoldResult newOffset = mulOFRValue(info.offset, rhs->scalar, loc, builder);
    OpFoldResult newStride = mulOFRValue(info.stride, rhs->scalar, loc, builder);

    StateInfo newStateInfo(newOffset, newStride, info.shape, info.mask, info.dim);

    stateInfo.push_back(newStateInfo);
  }

  sizes = lhs->sizes;

  if (rhs->hasModulo()) {
    op->emitRemark(
        "PtrAnalysis: do not support multiplying pointer states that has "
        "modulos");
    return failure();
  }

  auto leftState = const_cast<PtrState&>(lhsState) ;
  auto rightState = const_cast<PtrState&>(rhsState) ;
  this->getMemAccTypeRef().merge(leftState.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rightState.getMemAccTypeRef());
  return success();
}

tts::MakeTensorPtrOp PtrState::createTTSMakeTensorPtrOp(OpBuilder &builder,
                                                        Location loc) {
  SmallVector<int64_t> tensorSizes;
  SmallVector<OpFoldResult> tensorStrides;
  SmallVector<OpFoldResult> tensorOffsets;
  SmallVector<OpFoldResult> tensorShape;
  // [BEG] changed
  for(auto info : stateInfo){
    auto staticMask = getIntAttr(info.mask); // div rhs
    auto staticShape = getIntAttr(info.shape); // mod rhs
    auto staticSize = getIntAttr(sizes[info.dim]);
    auto staticStride = getIntAttr(info.stride);

    assert(staticSize.has_value() && "PtrAnalysis: do not support dynamic size");

    if(staticMask.has_value() && staticShape.has_value() ){
      if(staticStride.has_value() && staticStride == 0) continue;

      int64_t divDim = staticMask.value() ? (staticSize.value() - 1) / staticMask.value() + 1 : staticSize.value();
      int64_t remsiDim = staticShape.value() ? staticShape.value() : staticSize.value();
      int64_t trueDim = std::min(divDim, remsiDim);
      tensorSizes.push_back(trueDim);
    }else {
      tensorSizes.push_back(staticSize.value());
    }
    tensorShape.push_back(builder.getIndexAttr(0)); // after this, tensorShape express maskmode <|>
    tensorStrides.push_back(info.stride);
    tensorOffsets.push_back(info.offset);
  }

  assert(tensorSizes.size() && tensorStrides.size() && tensorOffsets.size() && tensorShape.size());
  assert(tensorSizes.size() == tensorStrides.size() && tensorOffsets.size() == tensorShape.size());
  assert(tensorStrides.size() == tensorOffsets.size());

  auto op = builder.create<mlir::tts::MakeTensorPtrOp>(
      // loc, source, tensorSizes, tensorStrides, tensorOffsets, tensorShape, SmallVector<int32_t>()); // changed
      loc, source, tensorSizes, tensorStrides, tensorOffsets, tensorShape, order); // origin
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_tensor_ptr:\n";
    op->dump();
  });
  return op;
}

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed())
    return failure();


  return state.addState(lhsState, rhsState, addOp, builder);
}

LogicalResult PtrAnalysis::visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(mulOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(mulOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  return state.mulState(lhsState, rhsState, mulOp, builder);
}

LogicalResult PtrAnalysis::visitOperandDiv(arith::DivSIOp divOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());
  PtrState rhsState;
  if (visitOperand(divOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    divOp->emitRemark(
        "PtrAnalysis: do not support division on non-scalar operands");
    return failure();
  }

  if (visitOperand(divOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  if(state.scalar){
    auto divOp =
        builder.create<arith::DivSIOp>(loc, state.scalar, rhsState.scalar);
    state.scalar = divOp.getResult();
    return success();
  }

  auto maskop = rhsState.scalar.getDefiningOp<arith::ConstantOp>();
  if(!maskop){
    divOp->emitError("Static compilation cannot determine the value of this parameter");
    return failure();
  }

  auto staticMask = cast<IntegerAttr>(maskop.getValue()).getInt();
  for(auto &info : state.stateInfo){
    auto staticSize = getIntAttr(state.sizes[info.dim]);
    auto staticStride = getIntAttr(info.stride);
    auto staticShape = getIntAttr(info.shape);
    auto preMask = getIntAttr(info.mask);

    assert(staticShape.has_value() && preMask.has_value());

    if(staticStride.has_value() && staticStride.value() % staticMask == 0){
      info.stride = builder.getIndexAttr(staticStride.value() / staticMask);
      auto newOffset = divOFRs(info.offset, rhsState.scalar, loc, builder);
      info.offset = newOffset;
      return success();
    }

    if(preMask.value() != 0){
      if(staticStride.has_value() && staticMask % staticStride.value() == 0 && staticShape.value() % staticSize.value() == 0){
        info.mask = builder.getIndexAttr(staticMask / staticStride.value());
      }else{
        divOp->emitError(
        "PtrAnalysis: do not support division after div.");
        return failure();
      }
    }else{
      info.mask = builder.getIndexAttr(staticMask);
    }
    auto newOffset = divOFRs(info.offset, rhsState.scalar, loc, builder);

    info.offset = newOffset;
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState rhsState;
  if (visitOperand(remOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    remOp->emitRemark("PtrAnalysis: only support cases when rhs of remainder "
                      "contains scalar");
    return failure();
  }

  if (visitOperand(remOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  if(state.scalar){
    auto RemOp =
        builder.create<arith::RemSIOp>(loc, state.scalar, rhsState.scalar);
    state.scalar = remOp.getResult();
    return success();
  }

  auto remsiop = rhsState.scalar.getDefiningOp<arith::ConstantOp>();
  if(!remsiop){
    remOp->emitError("Static compilation cannot determine the value of this parameter");
    return failure();
  }

  auto staticShape = cast<IntegerAttr>(remsiop.getValue()).getInt();
  for(auto &info : state.stateInfo){
    auto staticSize = getIntAttr(state.sizes[info.dim]);
    auto staticStride = getIntAttr(info.stride);
    auto preShape = getIntAttr(info.shape);
    auto staticMask = getIntAttr(info.mask);

    assert(preShape.has_value() && staticMask.has_value());

    if(!staticStride.has_value()){
      remOp->emitError(
        "PtrAnalysis: do not support dynamimx stride before remsi.");
        return failure();
    }else if(staticStride.value() % staticShape == 0) {
      info.stride = builder.getIndexAttr(0);
    }else if(staticShape % staticStride.value() != 0){
      remOp->emitError(
        "PtrAnalysis: do not support remsi after mul.");
        return failure();
    }

    if (preShape.value() != 0) {
      if (staticStride.has_value() && preShape.value() % staticShape == 0 && staticShape % staticStride.value() == 0) {
        info.shape = builder.getIndexAttr(staticShape / staticStride.value());
      } else if ((staticShape % preShape.value() == 0 || preShape.value() % staticShape == 0)
                  && staticStride.has_value()&& staticStride.value() == 1) {
        info.shape = builder.getIndexAttr(std::min(staticShape, preShape.value()));
      } else{
        remOp->emitError(
        "PtrAnalysis: do not support remsi after remsi.");
        return failure();
      }
    } else if (staticStride.has_value() && staticShape % staticStride.value() == 0) {
      info.shape = builder.getIndexAttr(staticShape / staticStride.value());
    } else {
      info.shape = builder.getIndexAttr(staticShape);
    }
    auto newOffset = remOFRs(info.offset, rhsState.scalar, loc, builder);

    info.offset = newOffset;
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());

  auto srcType = extOp.getIn().getType();
  if(visitOperand(extOp.getIn(), state, loc, builder).failed()){
    return failure();
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                                 PtrState &state, Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  auto offset = builder.getIndexAttr(start);
  auto infoStride = builder.getIndexAttr(stride);
  StateInfo newStateInfo(offset, infoStride, defaultAttr, defaultAttr);

  state.stateInfo.push_back(newStateInfo);
  state.sizes.push_back(builder.getIndexAttr(shape[0]));

  return success();
}

LogicalResult
PtrAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder) {
  assert(state.isEmpty());

  if (visitOperand(expandDimsOp.getSrc(), state, loc, builder).failed()) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");


  for (auto& info : state.stateInfo){
    if(info.dim >= axis)  ++info.dim;
  }

  state.sizes.insert(state.sizes.begin() + axis, builder.getIndexAttr(1));

  return success();
}

LogicalResult
PtrAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                   PtrState &state, const Location loc,
                                   OpBuilder &builder) {
  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);
  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();

  if (!isa<ShapedType>(src.getType())) {
    broadcastOp->emitRemark("PtrAnalysis: Unsupported broadcast source type");
    return failure();
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (srcShape.size() == 1 && srcShape[0] == 1) {
    StateInfo newStateInfo(defaultAttr, defaultAttr, defaultAttr, defaultAttr, 0);
    state.stateInfo.push_back(newStateInfo);
    state.sizes.push_back(builder.getIndexAttr(1));
  }
  if(state.sizes.empty()){
    for (size_t i = 0; i < dstShape.size(); i++) {
      state.sizes.push_back(builder.getIndexAttr(dstShape[i]));
    }
  }
  for (size_t i = 0; i < dstShape.size(); i++) {
    if (srcShape[i] == dstShape[i]) {
      continue;
    } else if (srcShape[i] < dstShape[i] && srcShape[i] == 1) {
      state.sizes[i] = builder.getIndexAttr(dstShape[i]);
    } else {
      llvm_unreachable("unexpected dimensions used in broadcast");
    }
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if(isa<IntegerType, IndexType, triton::PointerType>(src.getType())){
    for (size_t i = 0; i < dstShape.size(); ++i) {
      if(dstShape[i] != 1){
        StateInfo newStateInfo(defaultAttr, defaultAttr, defaultAttr, defaultAttr, i);
        state.stateInfo.push_back(newStateInfo);
      }
      state.sizes.push_back(builder.getIndexAttr(dstShape[i]));
    }
  } else {
    splatOp->emitRemark("PtrAnalysis: unsupported splat pattern");
    return failure();
  }

  // If we splat a integer value, scalar should become the offset of the outer
  // most dimension
  if (state.scalar)
    state.stateInfo[0].offset = state.scalar;

  if (state.hasModulo() && state.getRank() > 2) {
    LLVM_DEBUG({
      llvm::dbgs() << "visitOperandSplat failed\n";
      splatOp->dump();
    });
    splatOp->emitRemark("PtrAnalysis: unsupported scenario where splat result "
                        "has modulo and rank > 2");
    return failure();
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                              PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState ptrState;
  if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder)
          .failed()) {
    return failure();
  }

  PtrState offsetState;
  if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(),
                   builder)
          .failed()) {
    return failure();
  }

  assert(ptrState.source && "ptr field should provide source / base pointer");

  // offset has source means offset is from tl.load and other ops(TODO)
  if (offsetState.hasSource()) {
    ptrState.setMemAccTy(offsetState.getMemAccType());
    offsetState.removeSource();
  }

  // handle for loop & scalar
  if (ptrState.getRank() == 1 && offsetState.getRank() == 0) {
    auto size  = builder.getIndexAttr(1) ;
    auto offset = offsetState.scalar ;
    auto stride = builder.getIndexAttr(0);
    StateInfo newStateInfo(offset, stride, builder.getIndexAttr(0), builder.getIndexAttr(0));
    offsetState.stateInfo.push_back(newStateInfo);
    offsetState.sizes.push_back(size);
  }

  return state.addPtrState(ptrState, offsetState, addptrOp, builder);
}

LogicalResult PtrAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);
  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = builder.getIndexAttr(value.getSExtValue());
  auto constOp = arith::ConstantOp::materialize(builder, constAttr,
                                                builder.getIndexType(), loc);

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (size_t i = 0; i < resultType.getShape().size(); i++) {
    OpFoldResult offset;
    if (i == 0) {
      offset = constOp.getResult();
    } else {
      offset = defaultAttr;
    }
    StateInfo newStateInfo(offset, defaultAttr, defaultAttr, defaultAttr);

    state.stateInfo.push_back(newStateInfo);
    state.sizes.push_back(builder.getIndexAttr(resultType.getShape()[i]));
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandMakeTPtr(tts::MakeTensorPtrOp makeTPtrOp,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {

  assert(state.isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);
  state.source = makeTPtrOp.getBase();
  SmallVector<OpFoldResult> newOffsets = makeTPtrOp.getMixedOffsets();
  SmallVector<OpFoldResult> newStrides = makeTPtrOp.getMixedStrides();
  SmallVector<OpFoldResult> newShape = makeTPtrOp.getMixedShape();
  for(size_t i = 0; i < newOffsets.size(); i++){
    StateInfo newStateInfo(newOffsets[i], newStrides[i], newShape[i], defaultAttr);
    state.stateInfo.push_back(newStateInfo);
  }

  state.sizes = makeTPtrOp.getMixedSizes();

  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());

  return success();
}

SmallVector<int32_t> computeOrder(ArrayRef<int64_t> shape)
{
    SmallVector<int32_t> order;
    int rank = shape.size();
    order.reserve(rank);
    // 默认采用逆序 [dims - 1, ..., 0]
    for (int i = rank - 1; i >= 0; --i) {
        order.push_back(i);
    }
    return order;
}

LogicalResult PtrAnalysis::visitOperandDescriptorLoad(triton::DescriptorLoadOp descLoadOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) {
    assert(state.isEmpty());

    auto desc = descLoadOp.getDesc() ;
    auto descType = desc.getType();
    const auto blockShape = descType.getBlockType().getShape();
    //auto signlessBlockType = descType.getSignlessBlockType();
                                      
    auto makeDescOp = desc.getDefiningOp<triton::MakeTensorDescOp>();
    assert(makeDescOp && "Descriptor must be defined by MakeTensorDescOp");
    state.source = makeDescOp.getBase();  
    //Descriptor res;
    int rank = descType.getBlockType().getRank();

    if (rank != descLoadOp.getIndices().size() || makeDescOp.getShape().size() != rank 
        || makeDescOp.getStrides().size() != rank) {
        descLoadOp->emitRemark(
            "PtrAnalysis: expect sizes are aligned between desciptor_load and make_descriptor OP");
        return failure();
    }  


    SmallVector<int32_t> tensorShapeValues;
    for (auto dim : blockShape) {
        tensorShapeValues.push_back(static_cast<int32_t>(dim));
    }
    
    state.order = computeOrder(blockShape);
    auto defaultAttr = builder.getIndexAttr(0);
    // 直接回溯处理的 tt.make_tensor_descriptor
    for (int64_t i = 0; i < rank; i++) {
        auto offset = descLoadOp.getIndices()[i] ;
        auto shape =  makeDescOp.getShape()[i];
        auto stride = makeDescOp.getStrides()[i] ;
        StateInfo newStateInfo(offset, stride, shape, defaultAttr, i);
        state.stateInfo.push_back(newStateInfo);
        state.sizes.push_back(builder.getIndexAttr(tensorShapeValues[i]));
        state.dimLenth.push_back(1); // default dimLength in make_tensor_ptr, because no linear cases
    }
  
    return success() ;
}

LogicalResult
PtrAnalysis::visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) {
  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();

  if (makeTPtrOp.getOrder().empty()) {
    makeTPtrOp->emitRemark(
        "PtrAnalysis: expect tt.make_tensor_ptr to have order field set");
    return failure();
  }

  auto resType = cast<triton::PointerType>(makeTPtrOp.getResult().getType());
  auto pointeeType = cast<ShapedType>(resType.getPointeeType());
  auto shape = pointeeType.getShape();
  auto defaultAttr = builder.getIndexAttr(0);

  for (int64_t i = 0; i < pointeeType.getRank(); i++) {
    auto strideCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getStrides()[i]);
    auto offsetCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getOffsets()[i]);
    auto scaledOffset = builder.create<arith::MulIOp>(
        loc, offsetCst.getResult(), strideCst.getResult());
    auto shapeCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getShape()[i]);

    auto offset = scaledOffset.getResult();
    auto stride = strideCst.getResult();
    auto infoShape = shapeCst.getResult();

    StateInfo newStateInfo(offset, stride, infoShape, defaultAttr, i);
    state.stateInfo.push_back(newStateInfo);
    state.sizes.push_back(builder.getIndexAttr(shape[i]));
    state.dimLenth.push_back(1); // default dimLength in make_tensor_ptr, because no linear cases
  }
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());
  assert(state.isBlockPtr() &&
         "tt.make_tensor_ptr pointer state should describe a block pointer");

  return success();
}

LogicalResult PtrAnalysis::visitOperandForOp(scf::ForOp forOp, Value operand,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {

  auto it = llvm::find(forOp->getResults(), operand);
  auto index = std::distance(forOp->getResults().begin(), it);

  auto newState = getLoopResultPtrState(forOp, index);
  if (failed(newState)) {
    forOp.emitError(
        "Rewrite for-op failed. Could not find PtrState returned by "
        "the loop.");
    return failure();
  }

  state = newState.value();
  return success();
}

template <typename OpTy>
LogicalResult PtrAnalysis::visitOperandIndirectLoad(OpTy op,
                                      PtrState &state,
                                      const Location &loc,
                                      OpBuilder &builder) {
  // FIXME: assume single result of operation
  auto opRes = op->getResult(0);
  auto opResTy = opRes.getType();
  std::vector<int64_t> resShape;
  if (auto shapedResTy = dyn_cast<ShapedType>(opResTy)) {
    // For now, we consider this is UnstrucMemAcc because we have no other info.
    // Visiting other ops may change the type due to more info.
    state.setMemAccVal( MemAccVal::UnstrucMemAcc);
    resShape = shapedResTy.getShape().vec();
  } else {
    // scalar load means this is used as offset. It is StrucMemAcc.
    state.setMemAccVal(MemAccVal::StrucMemAcc);
    resShape.push_back(1);
  }

  auto defaultAttr = builder.getIndexAttr(0);
  auto count = 0 ;
  for (auto &s : resShape) {
     auto offset = builder.getIndexAttr(0);
     auto stride = builder.getIndexAttr(1);
     auto shape = builder.getIndexAttr(s);
     StateInfo newStateInfo(offset, stride,  builder.getIndexAttr(0), defaultAttr, count++);
     state.stateInfo.push_back(newStateInfo);
     state.sizes.push_back(shape);
  }
  // set the source in BlockData so that we know an indirect-load op exists in
  // the chain.
  state.source = opRes ;
  return success();

}

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {
  // to fix UT gather_flip. 当pointertype, 而且来自AddPtr
  // 将source 指向自己
  if (isa<triton::PointerType>(operand.getType()))
  {
    if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
      state.source = operand ;
      return success();
    }
  }

  if (knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return success();
  }

  if (isa<IntegerType>(operand.getType())) {
    OpBuilder::InsertionGuard guard(builder);
    if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
      builder.setInsertionPointAfter(operand.getDefiningOp());
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return success();
  } else if (isa<IndexType>(operand.getType())) {
    state.scalar = operand;
    return success();
  } else if(isa<RankedTensorType>(operand.getType()) && cast<ShapedType>(operand.getType()).getShape().size() == 1 && cast<ShapedType>(operand.getType()).getShape()[0] == 1){
    state.scalar = operand;
    return success();
  }

  if (isa<triton::PointerType>(operand.getType())) {
    // A scalar pointer can either be produced by AddPtrOp or a block
    // argument
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc,
                                  builder);
      } else if (auto bitCastOp = dyn_cast<triton::BitcastOp>(op)){
        state.source = operand;
        return success();
      } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        llvm_unreachable("Unexpected operand defining operation tts.make_tptr");
      } else {
        llvm_unreachable("Unexpected operand");
      }
    } else {
      state.source = operand;
      return success();
    }
  }

  auto tensorType = dyn_cast<mlir::RankedTensorType>(operand.getType());
  bool isScalar = true;
  for(size_t i = 0; i < tensorType.getRank() && isScalar; ++i){
    isScalar = tensorType.getDimSize(i) == 1;
  }
  if(isScalar){
    auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    SmallVector<mlir::Value> indices;
    for(size_t i = 0; i < tensorType.getRank() && isScalar; ++i){
      indices.push_back(index);
    }
    auto extractedElement = builder.create<mlir::tensor::ExtractOp>(loc, operand, indices);
    state.scalar = extractedElement.getResult();
    return success();
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return visitOperandAdd(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    return visitOperandMul(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return visitOperandMakeRange(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return visitOperandBroadcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return visitOperandSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return visitOperandExpandDims(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return visitOperandAddptr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return visitOperandConstSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return visitOperandRem(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    return visitOperandDiv(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return visitOperandExtSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<scf::ForOp>()) {
    return visitOperandForOp(op, operand, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<tts::LoadOp>()) {
    return visitOperandIndirectLoad(op, state, loc, builder) ;
    // op->emitError("PtrAnalysis: Invalid dynamic offset"
    //               "The load operation's offset cannot be derived from another load result.");
    // operand.dump();
    // return failure();
  } else if (auto op = operand.getDefiningOp<arith::FPToSIOp>()) {
    return visitOperandIndirectLoad(op, state, loc, builder) ;
    // op->emitError("IllegalTypeConversionInAddressCalculation"
    //               "float-to-int precision conversion is not supported during address computation.");
    // operand.dump();
    // return failure();
  } else if (!operand.getDefiningOp()) {
    if (!knownPtrs.contains(operand)) {
      llvm::dbgs() << "PtrAnalysis: Pointer analysis is not supported for input parameters\n";
      return failure();
    }

    // This operand must be an iter-arg of an inner-loop in a multiple-level
    // nested loop, which means its PtrState must have already been populated
    // during rewriteForOp of the parent loop.
    state = knownPtrs[operand];
    return success();
  } else {
    auto op = operand.getDefiningOp();
    op->emitError("PtrAnalysis: encountered addptr operand produced by an unsupported operation");
    operand.dump();
    return failure();
  }
}


LogicalResult PtrAnalysis::rewriteAddptrOp(triton::AddPtrOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  PtrState state;
  if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  knownPtrs[op.getResult()] = state;

  if(state.sizes.empty()){
    op->emitRemark("state is empty");
    return failure();
  }

  if (state.memAccTy.isUnstructured()) {
    // TODO: Based on more info, try to create a performant IR
    auto ret = rewriteAddPtrToUnstrucMemAcc(op, state);
    LLVM_DEBUG({ llvm::dbgs() << *getModuleOpFromOperation(op) << "\n"; });
    return ret ;
  }

  auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
  ptrMap.map(op.getResult(), maketptrOp.getResult());

  return success();
}

LogicalResult rewriteTensorDescTypeOp(triton::TensorDescType op,  PtrState &state) {
  return success();
}

LogicalResult PtrAnalysis::rewriteMakeTensorPtrOp(triton::MakeTensorPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandMakeTensorPtr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), maketptrOp.getResult());
  return success();
}

LogicalResult PtrAnalysis::rewriteAdvanceOp(triton::AdvanceOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  PtrState state;
  if (visitOperand(op->getOperand(0), state, loc, builder).failed()) {
    op->emitRemark("PtrAnalysis: Failed to analyze ptr of tt.advance");
    return failure();
  }
  assert(state.isBlockPtr() &&
         "tt.advance pointer state should describe a block pointer");

  auto incrementOffsets = op.getOffsets();

  SmallVector<OpFoldResult> newOffsets;
  for (auto [increment, tempStateInfo] :
       llvm::zip(incrementOffsets, state.stateInfo)) {
    OpFoldResult offset = tempStateInfo.offset;
    OpFoldResult stride = tempStateInfo.stride;
    Value offsetValue;
    if (auto offsetIntAttr = getIntAttr(offset)) {
      auto constOp = builder.create<arith::ConstantOp>(
          loc, builder.getIndexAttr(offsetIntAttr.value()));
      offsetValue = constOp.getResult();
    } else {
      offsetValue = offset.get<Value>();
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), increment);
    auto mulOp = builder.create<arith::MulIOp>(loc, castOp.getResult(),
                                               stride.get<Value>());
    auto addOp =
        builder.create<arith::AddIOp>(loc, mulOp.getResult(), offsetValue);
    newOffsets.push_back(addOp.getResult());
  }

  for(size_t i = 0; i < newOffsets.size(); i++){
    state.stateInfo[i].offset = newOffsets[i];
  }

  auto newOp = state.createTTSMakeTensorPtrOp(builder, loc);
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), newOp.getResult());
  return success();
}

static bool isPointerType(Type t) {
  if (auto tensor = llvm::dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensor.getElementType());
  }
  return isa<triton::PointerType>(t);
}

FailureOr<PtrState> PtrAnalysis::getLoopInitArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto ptr = forOp.getInitArgs()[index];

  // If the pointer into the scf.for was defined by tts.get_structured_state,
  // we can get the pointer state from the original pointer (the op's input):
  //
  // %ptr, %offset_1, %offset_2,..., %stride_1, %stride_2,... =
  // tts.get_structured_state %original
  // scf.for ... (%ptr) {...}
  if (auto getStateOp = ptr.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto originalPtr = getStateOp->getOperand(0);
    if (knownPtrs.count(originalPtr)) {
      return knownPtrs[originalPtr];
    }
  }

  // For nested loops scenarios, a pointer in init-args can be returned from
  // another loop of the same level:
  // e.g.:
  // clang-format off
  //  %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //    %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
  //    }
  //    %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
  //      %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      ...
  //    }
  //    ...
  //  }
  // clang-format on
  // Notice %arg8 = %23 comes from the return value of the first loop.
  if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    return getLoopResultPtrState(forOp, index);
  }

  // If the pointer isn't defined by tts.get_structured_state nor another loop,
  // it means the current pointer is an iterarg of the outer loop.
  // In such cases, the outer loops would have already set up the PtrState for
  // us already.
  //
  // scf.for iterargs(%ptr = %init_arg) {
  //    scf.for iterargs(%ptr1 = %ptr) {  <--- we're dealing with `%ptr1` here.
  //          ...
  //    }
  // }
  if (knownPtrs.count(ptr)) {
    assert(!ptr.getDefiningOp() && "Expect the ptr to be an iterarg");
    return knownPtrs[ptr];
  }

  return failure();
}

PtrState PtrAnalysis::reconcileLoopPtrState(
    scf::ForOp forOp, size_t iterArgIndex, const PtrState &state,
    llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal) {
  PtrState newState = state;
  int cnt = iterArgIndex + 1;
  if (newState.getRank() == 0) {
    assert(newState.scalar);
    // for scalar pointers, the scalar contains the offset and is the only
    // relevant newState that could be updated by the loop.
    newState.scalar = getReplacementVal(forOp, cnt);
  } else {
    for (auto &info : newState.stateInfo) {
      info.offset = getReplacementVal(forOp, cnt++);
    }

    for (auto &info : newState.stateInfo) {
      info.stride = getReplacementVal(forOp, cnt++);
    }
  }

  return newState;
}

FailureOr<PtrState> PtrAnalysis::getLoopIterArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op.getRegionIterArg(index); });
}

FailureOr<PtrState> PtrAnalysis::getLoopResultPtrState(scf::ForOp forOp,
                                                       size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
}

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op->getResult(index); });
}


// Update for-loop transformation to the latest triton-shared version
LogicalResult PtrAnalysis::rewriteForOp(scf::ForOp op) {
  for (auto [i, arg] : llvm::enumerate(op.getRegionIterArgs())) {
    if (!maybeStructuredArgs.contains(arg)) {
      continue;
    }

    auto state = getLoopIterArgPtrState(op, i);
    if (failed(state)) {
      // Because the maybeStructuredArgs may contain values that are not
      // considered structured by PtrAnalysis, failing to retrieve the PtrState
      // should not fail the rewrite process.
      // We emit an error for diagnostics and debugging purposes.
      op->emitWarning(
          "Rewrite for-op failed. Could not find PtrState for iter-arg index " +
          std::to_string(i));
      continue;
    }
    // Skip when no structured dimension exists
    // if (state->noStructuredDimExists())
    //   continue;

    // Save the current init arg's PtrState
    knownPtrs[arg] = state.value();

    // For tensors of pointers, create a tts.make_tptr at the beginning of the
    // loop body that correspond to this region iter arg. In case it is used
    // by tt.load/tt.store in the loop body before pointer updates, this will
    // make sure rewriteLoadOp/rewriteStoreOp can use the analysis result.
    // E.g., given the following input (%tensor_of_ptr is a block arg):
    // scf.for (%tensor_of_ptr) {
    //   %data = tt.load %tensor_of_ptr
    //   // more operations to update %tensor_of_ptr
    // }
    // We may produce the following output:
    // scf.for (%base_ptr, %stride, %offset) {
    //   %tensor_of_ptr = tts.make_tptr(%base_ptr, %stride, %offset)
    //   %data = tts.load %tensor_of_ptr
    //   // more operations to update %offset
    // }
    // If %tensor_of_ptr is not used (i.e., %tensor_of_ptr is updated before
    // used in the original IR), it will simply be removed by
    // canonicalization.

    // For scalar pointers, there is no need to create a tts.addptr at the
    // beginning of the loop body. We don't lower tt.load and tt.store on
    // scalars in this pass; pointer arithmetics can also just use the
    // original pointer.
    // Note that there can be tensor of indices in iter-arg, so we only create
    // the make_tensor_ptr op when the arg is of pointer type.
    if (isPointerType(arg.getType())) {
      if (state->getRank() != 0) {
        OpBuilder builder(op.getRegion());
        auto maketptrOp = state->createTTSMakeTensorPtrOp(builder, op.getLoc());
        ptrMap.map(arg, maketptrOp.getResult());
      }
    }
  }

  // Recursively rewrite the inner ops
  if (rewriteOp(op).failed()) {
    op->emitRemark(
        "PtrAnalysis: update loop body failed when rewriting for op");
    return failure();
  }

  return success();
}

LogicalResult
PtrAnalysis::rewriteGetStructuredStateOp(tts::GetStructuredStateOp op) {
  auto tritonValue = op->getOperand(0);

  // If this triton value isn't known, it means PtrAnalysis has failed to
  // analyze this pointer. In such cases, simply remap all uses of the
  // structured value back to its original triton value.
  if (!knownPtrs.contains(tritonValue)) {
    op.emitRemark(
        "Rewrite GetStructuredStateOp failed. Could not find PtrState.");
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }

  tts::PtrState state = knownPtrs[tritonValue];
  Value remappedValue =
      ptrMap.contains(tritonValue) ? ptrMap.lookup(tritonValue) : tritonValue;

  SmallVector<Value> replacements{remappedValue};
  OpBuilder builder(op);

  if (state.getRank() == 0) {
    // For scalar pointers, the scalar contains the offset and is the only
    // relevant state that could be updated by the loop.
    if (state.scalar) {
      replacements.push_back(state.scalar);
    } else {
      // This operand is a pointer directly from the kernel arguments.
      // Use offset 0.
      assert(!tritonValue.getDefiningOp());
      replacements.push_back(builder.create<arith::ConstantOp>(
          op.getLoc(), builder.getIndexAttr(0)));
    }
  } else {
    for (auto info : state.stateInfo) {
      auto s = info.offset;
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(s.get<Value>());
      }
    }

    for (auto info : state.stateInfo) {
      auto s = info.stride;
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(s.get<Value>());
      }
    }
  }

  op->replaceAllUsesWith(replacements);
  op->erase();
  return success();
}

LogicalResult
PtrAnalysis::rewriteYieldOp(scf::YieldOp op,
                            llvm::SmallDenseMap<int, PtrState> &knownPtrsFor) {
  if (levelToBlockArgIndex.find(level) == levelToBlockArgIndex.end()) {
    // no need to rewrite this op
    return success();
  }

  OpBuilder builder(op);

  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // PtrState for those values.
  SmallVector<PtrState, 5> initArgState;
  for (auto [i, v] : llvm::enumerate(op->getOperands())) {
    // If this operand is not rewritten by forOp, skip
    auto thisSet = levelToBlockArgIndex.find(level)->second;
    if (thisSet.find(i) == thisSet.end())
      continue;

    auto mappedV = ptrMap.lookupOrNull(v);
    if (!mappedV) {
      op->emitRemark("Prior rewrite failure lead to yield rewrite failure");
      return failure();
    }

    PtrState state;
    LogicalResult ret = failure();
    if (auto makeTPtrOp = mappedV.getDefiningOp<tts::MakeTensorPtrOp>()) {
      ret = visitOperandMakeTPtr(makeTPtrOp, state, op.getLoc(), builder);
    } else if (auto addptrOp = mappedV.getDefiningOp<triton::AddPtrOp>()) {
      ret = visitOperandAddptr(addptrOp, state, op.getLoc(), builder);
    }
    if (ret.failed()) {
      op->emitRemark("Failed to rewrite yield op");
      return failure();
    }
    initArgState.push_back(state);

    // Verify that shape is not updated during the for loop
    auto forState = knownPtrsFor[i];
    for (auto i = 0; i < forState.getRank(); ++i) {
      if(forState.stateInfo[i].shape != state.stateInfo[i].shape){
        // Special case, see comments in addState in dealing with shape/modulo
        if (i == 0 && forState.getRank() == 2) {
          if (forState.stateInfo[1].shape == state.stateInfo[0].shape &&
              forState.stateInfo[0].shape == state.stateInfo[1].shape) {
            break;
          }
        }
        assert(0);
        op->emitRemark("PtrAnalysis: operand's shape/modulo state changed "
                       "within loop body");
        return failure();
      }
    }
  }

  SmallVector<Value> operands;
  for (auto opnd : op->getOperands()) {
    auto mappedV = ptrMap.lookupOrNull(opnd);
    if (mappedV) {
      operands.push_back(mappedV);
    } else {
      operands.push_back(opnd);
    }
  }

  // For each of the PtrState recorded in the last step, extract value
  // that correspond to offset and stride for each dimension and append
  // them to yield operands.
  for (auto state : initArgState) {
    for(size_t i = 0; i < state.stateInfo.size(); i++){
      auto s = state.stateInfo[i].offset;
      if (auto sIntAttr = getIntAttr(s)) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(s.get<Value>());
      }
    }

    for (size_t i = 0; i < state.stateInfo.size(); i++) {
      auto s = state.stateInfo[i].stride;
      assert(!getIntAttr(s) && "PtrState strides for yield within for "
                               "loop not expected to be attribute.");
      operands.push_back(s.get<Value>());
    }

    if (state.getRank() == 0) {
      operands.push_back(state.scalar);
    }
  }

  auto newOp = builder.create<scf::YieldOp>(op->getLoc(), operands);

  LLVM_DEBUG({
    llvm::dbgs() << "new yield:";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  op->erase();
  return success();
}

LogicalResult PtrAnalysis::analysisSplat(Operation *op, OpBuilder &builder, Value &ptr, PtrState &ptrState){
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    ptr = loadOp.getPtr();
  } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
    ptr = storeOp.getPtr();
  } else if (auto atomicRmwOp = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
    ptr = atomicRmwOp.getPtr();
  } else if (auto atomicCasOp = llvm::dyn_cast<triton::AtomicCASOp>(op)) {
    ptr = atomicCasOp.getPtr();
  }
  else {
      op->emitError("Unsupported operation type for mask generate mask");
      return failure();
  }
  PtrState tempState;
  if(!isa<mlir::RankedTensorType>(ptr.getType())){
    tempState.ptrIsTensor = false;
  }
  if (auto splatOp = ptr.getDefiningOp<triton::SplatOp>()) {
    auto splatPtr = splatOp.getSrc();
    // auto dst = splatOp.getResult();
    // auto dstShape = cast<ShapedType>(dst.getType()).getShape();
    // if((dstShape.size() != 1 || dstShape[0] != 1) && dstShape){
    //   op->emitRemark("Non-scalar pointers are not supported from splat");
    //   return failure();
    // }
    // auto tensorPtr = ptrMap.lookupOrNull(splatPtr);
    tempState.source = splatPtr;
  } else if(auto bitcastOp = ptr.getDefiningOp<triton::BitcastOp>()){
    auto bitcastPtr = bitcastOp.getSrc();
    tempState.source = bitcastPtr;
    if(auto addptrOp = bitcastPtr.getDefiningOp<triton::AddPtrOp>()){
      auto sourceType = ptr.getType();
      if(auto rankedType = dyn_cast<mlir::RankedTensorType>(sourceType)){
        sourceType = rankedType.getElementType();
      }
      tempState = knownPtrs[bitcastPtr];
      auto bitCastOp = builder.create<triton::BitcastOp>(op->getLoc(), sourceType, tempState.source);
      tempState.source = bitCastOp.getResult();

      auto maketptrOp = tempState.createTTSMakeTensorPtrOp(builder, op->getLoc());
      knownPtrs[ptr] = tempState;
      ptrMap.map(ptr, maketptrOp);
      ptr = maketptrOp;
      ptrState = tempState;
      return success();
    }
  } else {
    op->emitError("PtrAnalysis: pointer is not replace with tts.make_tptr so "
                  "loadOp cannot be rewritten");
    return failure();
  }
  auto tensorType = dyn_cast<mlir::RankedTensorType>(ptr.getType());
  auto defaultAttr = builder.getIndexAttr(0);
  if(tensorType){
    for(size_t i = 0; i < tensorType.getRank(); ++i){
      tempState.sizes.push_back(builder.getIndexAttr(1));
      StateInfo placeHolder(defaultAttr, builder.getIndexAttr(1),
                          defaultAttr, defaultAttr, i);
      tempState.stateInfo.push_back(placeHolder);
      tempState.dimLenth.push_back(1);
    }
  }else{
    tempState.source = ptr;
    tempState.sizes.push_back(builder.getIndexAttr(1));
    StateInfo placeHolder(defaultAttr, builder.getIndexAttr(1),
                        defaultAttr, defaultAttr, 0);
    tempState.stateInfo.push_back(placeHolder);
    tempState.dimLenth.push_back(1);
  }
  if(tempState.sizes.empty()){
    op->emitRemark("state is empty");
    return failure();
  }
  auto maketptrOp = tempState.createTTSMakeTensorPtrOp(builder, op->getLoc());
  knownPtrs[ptr] = tempState;
  ptrMap.map(ptr, maketptrOp.getResult());
  ptr = maketptrOp.getResult();
  ptrState = tempState;
  return success();
}

LogicalResult PtrAnalysis::rewriteScalarLoadOp(triton::LoadOp op, OpBuilder &builder,
                                                Value &loadResult, const Location loc) {
  if(ptrMap.lookupOrNull(op.getPtr())){
    auto tensorType = dyn_cast<mlir::RankedTensorType>(loadResult.getType());
    auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    SmallVector<mlir::Value> indices;
    for(size_t i = 0; i < tensorType.getRank(); ++i){
      assert(tensorType.getDimSize(i) == 1 && "Input tensor should be of shape tensor<1xanytype>");
      indices.push_back(index);
    }
    auto extractedElement = builder.create<mlir::tensor::ExtractOp>(loc, loadResult, indices);
    loadResult = extractedElement.getResult();
  }
  op.replaceAllUsesWith(loadResult);
  op->erase();
  return success();
}


LogicalResult PtrAnalysis::createBroadcast(Operation *op, SmallVector<int64_t> &loadShape,
                                            Value &loadResult){
  PtrState ptrState;
  OpBuilder builder(op);
  auto loc = op->getLoc();
  SmallVector<int64_t> dimensions;

  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    ptrState = knownPtrs[loadOp.getPtr()];
  } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
    ptrState = knownPtrs[storeOp.getPtr()];
  }else if (auto atomicRmwOp = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
    ptrState = knownPtrs[atomicRmwOp.getPtr()];
  }else if (auto atomicCasOp = llvm::dyn_cast<triton::AtomicCASOp>(op)) {
    ptrState = knownPtrs[atomicCasOp.getPtr()];
  }else {
      op->emitError("Unsupported operation type for mask generate mask");
      return failure();
  }

  for(size_t i = 0; i < ptrState.stateInfo.size(); ++i){
    auto x = ptrState.stateInfo[i];
    auto staticStride = getIntAttr(x.stride);
    auto staticShape = getIntAttr(x.shape);
    auto staticMask = getIntAttr(x.mask);
    auto staticSize = getIntAttr(ptrState.sizes[x.dim]);
    if (!staticShape.has_value()) {
      op->emitError("do not support dynamic shape");
      return failure();
    }
    if(staticStride.has_value() && staticStride == 0){
      int64_t divDim = staticMask.value() ? (staticSize.value() - 1) / staticMask.value() + 1 : staticSize.value();
      int64_t remsiDim = staticShape.value() ? staticShape.value() : staticSize.value();
      int64_t trueDim = std::min(divDim, remsiDim);
      loadShape.insert(loadShape.begin() + i, trueDim);
      dimensions.push_back(i);
    }
  }
  auto targetShapeType = RankedTensorType::get(loadShape, cast<ShapedType>(loadResult.getType()).getElementType());

  auto init = builder.create<tensor::EmptyOp>(loc, loadShape, cast<ShapedType>(loadResult.getType()).getElementType());
  auto broadcastShapeAttr = builder.getI64VectorAttr(loadShape);
  auto broadcastOp = builder.create<linalg::BroadcastOp>(
      loc,
      loadResult,
      init,
      dimensions
  );
  loadResult = broadcastOp->getResult(0);
  return success();
}


LogicalResult PtrAnalysis::createReshape(Operation *op, Value &srcResult, SmallVector<int64_t> &srcShape) {
  Value ptr;
  PtrState ptrState;
  OpBuilder builder(op);
  auto loc = op->getLoc();
  SmallVector<int64_t> validSizes;
  SmallVector<int64_t> flatSizes;
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    ptrState = knownPtrs[loadOp.getPtr()];
  } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
    ptrState = knownPtrs[storeOp.getPtr()];
  } else if (auto atomicRmwOp = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
    ptrState = knownPtrs[atomicRmwOp.getPtr()];
  } else if (auto atomicCasOp = llvm::dyn_cast<triton::AtomicCASOp>(op)) {
    ptrState = knownPtrs[atomicCasOp.getPtr()];
  } else {
      op->emitError("Unsupported operation type for mask generate mask");
      return failure();
  }

  for(auto x : ptrState.sizes){
    auto staticSize =  getIntAttr(x);
    if (!staticSize.has_value()) {
      op->emitError("do not support dynamic size");
      return failure();
    }
    validSizes.push_back(staticSize.value());

    flatSizes.push_back(1);
  }
  size_t startPos = 0;
  for(size_t i = 0; i < flatSizes.size(); ++i){
    size_t endPos = startPos + ptrState.dimLenth[i];
    for(size_t j = startPos; j < endPos; ++j){
      assert(j < srcShape.size());
      flatSizes[i] *= srcShape[j];
    }
    startPos = endPos;
  }
  if(srcShape.size() != flatSizes.size()){
    auto targetShapeType = RankedTensorType::get(flatSizes, cast<ShapedType>(srcResult.getType()).getElementType());
    auto targetShapeAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(flatSizes.size())}, builder.getI64Type()), flatSizes);
    auto targetShape = builder.create<arith::ConstantOp>(loc, targetShapeAttr);
    auto reshapeOp = builder.create<tensor::ReshapeOp>(loc, targetShapeType, srcResult, targetShape);

    srcResult = reshapeOp.getResult();
  }

  if(!std::equal(flatSizes.begin(), flatSizes.end(), validSizes.begin())){
    auto sourceTy = cast<mlir::RankedTensorType>(srcResult.getType());
    mlir::Type elementTy = sourceTy.getElementType();
    auto resultTy = mlir::RankedTensorType::get(validSizes, elementTy);
    SmallVector<int64_t> staticOffsets(sourceTy.getRank(), 0);
    SmallVector<int64_t> staticStrides(sourceTy.getRank(), 1);

    auto extractSliceOp = builder.create<mlir::tensor::ExtractSliceOp>(
      loc,
      resultTy,
      srcResult,
      /*offsets=*/ValueRange{},
      /*sizes=*/ValueRange{},
      /*strides=*/ValueRange{},
      staticOffsets,
      validSizes,
      staticStrides
  );

    srcResult = extractSliceOp.getResult();
  }
  return success();
}

LogicalResult PtrAnalysis::generateMask(Operation * op, PtrState &ptrState, SmallVector<OpFoldResult> &dims, SmallVector<int64_t> &dimMode) {
  Value mask;
  OpBuilder builder(op);
  mlir::triton::MaskState mstate;
  auto defaultAttr = builder.getIndexAttr(0);
  auto loc = op->getLoc();
  assert(dimMode.empty());

  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    mask = loadOp.getMask();
  } else if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
    mask = storeOp.getMask();
  } else if (auto atomicRmwOp = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
    mask = atomicRmwOp.getMask();
  }else {
      op->emitError("Unsupported operation type for mask generate mask");
      return failure();
  }
  if (mstate.parse(mask, loc, builder).failed()) {
    op->emitRemark("MaskAnalysis failed");
    return failure();
  }
  dims = mstate.dims;
  size_t remainMask = dims.size();
  if(mstate.stateInfo.empty())  mstate.stateInfo.push_back(triton::dimInfo(defaultAttr, defaultAttr));
  SmallVector<OpFoldResult> tempDim;
  for(auto info : ptrState.stateInfo){
    auto staticStride = getIntAttr(info.stride);
    auto staticShape = getIntAttr(info.shape);
    auto staticMask = getIntAttr(info.mask);
    auto staticSize = getIntAttr(ptrState.sizes[info.dim]);

    assert(staticShape.has_value() && staticMask.has_value());
    if(staticStride.has_value() && staticStride.value() == 0) continue;

    bool findMask = false;
    for(int i = 0; i < mstate.stateInfo.size(); ++i){
      auto msShape = getIntAttr(mstate.stateInfo[i].shape);
      auto msDiv = getIntAttr(mstate.stateInfo[i].div);
      auto msDim = mstate.stateInfo[i].dim;
      assert(msShape.has_value() && msDiv.has_value());
      if(msDim == info.dim && ((msShape.value() == staticShape.value() &&
          msDiv.value() == staticMask.value()))){
        findMask = true;
      }else if(msDim == info.dim && (staticMask.value() == 0 || staticMask.value() % staticSize.value() != 0) &&
               msShape.value() == 0 && msDiv.value() == 0){
        size_t trueLenth = 0;
        for(auto y : ptrState.stateInfo){
          if(y.dim != info.dim) continue;
          auto mask = getIntAttr(y.mask);
          auto size = getIntAttr(ptrState.sizes[y.dim]);
          if(mask.value() == 0 || mask.value() % size.value() != 0) ++trueLenth;
        }
        if(trueLenth == 1)  findMask = true;
        if(findMask && msShape.value() + msDiv.value() != 0){
           // op->emitWarning("Mask-Pointer inconsistency detected");
        }
      }
      if(findMask){
        tempDim.emplace_back(dims[i]);
        int64_t mode = mstate.stateInfo[i].isSlt ? 0 : 1;
        dimMode.emplace_back(mode);
        --remainMask;
        break;
      }
    }
    if(!findMask) {
      int64_t divDim = staticMask.value() ? (staticSize.value() - 1) / staticMask.value() + 1 : staticSize.value();
      int64_t remsiDim = staticShape.value() ? staticShape.value() : staticSize.value();
      int64_t trueDim = std::min(divDim, remsiDim);
      tempDim.emplace_back(builder.getIndexAttr(trueDim));
      dimMode.emplace_back(0);
    }
  }
  if(remainMask != 0){
    //op->emitWarning("Mask-Pointer inconsistency detected");
  }
  dims = tempDim;
  return success();
}

LogicalResult PtrAnalysis::extractScalarFromLoadedTensor(Operation* op, OpBuilder &builder,
                                                Value &loadResult, const Location loc) {
  Value ptr;
  if(auto loadOp = dyn_cast<triton::LoadOp>(op)){
    ptr = loadOp.getPtr();
  } else if(auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)){
    ptr = atomicRMWOp.getPtr();
  } else if (auto atomicCasOp = dyn_cast<triton::AtomicCASOp>(op)) {
    ptr = atomicCasOp.getPtr();
  } else{
    op->emitError("Unsupported operation type for mov data from GM to UB");
    return failure();
  }

  if(ptrMap.lookupOrNull(ptr)){
    auto tensorType = dyn_cast<mlir::RankedTensorType>(loadResult.getType());
    auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    SmallVector<mlir::Value> indices;
    for(size_t i = 0; i < tensorType.getRank(); ++i){
      assert(tensorType.getDimSize(i) == 1 && "Input tensor should be of shape tensor<1xanytype>");
      indices.push_back(index);
    }
    auto extractedElement = builder.create<mlir::tensor::ExtractOp>(loc, loadResult, indices);
    loadResult = extractedElement.getResult();
  }

  return success();
}

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute> filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs)
{
    LLVM_DEBUG({
      llvm::dbgs() << "original attrs:";
      for (auto it = attrs.begin(), end = attrs.end(); it != end; ++it) {
        auto attr = *it;
        auto attrName = attr.getName().getValue();
        
        llvm::dbgs() << attrName << " " ;
      }
       llvm::dbgs() << "\n " ;

    });
    mlir::SmallVector<NamedAttribute> ret;
    llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
        auto attrName = attr.getName().getValue();
        return attrName != "operandSegmentSizes";
    });

    LLVM_DEBUG({
      llvm::dbgs() << "after fileter:";
      for (auto it = ret.begin(), end = ret.end(); it != end; ++it) {
        auto attr = *it;
        auto attrName = attr.getName().getValue();
        llvm::dbgs() << attrName << " " ;
      }
       llvm::dbgs() << "\n " ;

    });
    return ret;
}

struct Descriptor {
    Value base;
    SmallVector<Value> shape;
    SmallVector<Value> strides;
};

Descriptor unpackDescriptor(TensorDescType type, Value desc, OpBuilder &rewriter)
{
    auto makeDescOp = desc.getDefiningOp<triton::MakeTensorDescOp>();
    assert(makeDescOp && "Descriptor must be defined by MakeTensorDescOp");

    Descriptor res;
    int rank = type.getBlockType().getRank();

    // 直接回溯处理的 tt.make_tensor_descriptor
    res.base = makeDescOp.getBase();
    for (auto s : makeDescOp.getShape()) {
        res.shape.push_back(rewriter.createOrFold<arith::ExtSIOp>(makeDescOp.getLoc(), rewriter.getI64Type(), s));
    }
    for (auto st : makeDescOp.getStrides()) {
        res.strides.push_back(rewriter.createOrFold<arith::ExtSIOp>(makeDescOp.getLoc(), rewriter.getI64Type(), st));
    }

    return res;
}

LogicalResult PtrAnalysis::rewriteDescriptorLoadOp(triton::DescriptorLoadOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();
  const auto blockShape = op.getDesc().getType().getBlockType().getShape();
  auto descTy = op.getDesc().getType();
  auto resultTy = descTy.getSignlessBlockType();
  auto indices = op.getIndices();

  // 1. 解包 descriptor
  auto desc = unpackDescriptor(descTy, op.getDesc(), builder);

  // 2. 新增 make_tensor_ptr
  SmallVector<int32_t> tensorShapeValues;
  for (auto dim : blockShape) {
      tensorShapeValues.push_back(static_cast<int32_t>(dim));
  }

  //auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, loc);
   Value maketptrOp = builder.create<triton::MakeTensorPtrOp>(loc,
                                                               desc.base,               // 基址
                                                               desc.shape,              // 形状
                                                               desc.strides,            // 步长
                                                               indices,                 // 偏移
                                                               tensorShapeValues,       // tensorShape
                                                               computeOrder(blockShape) // 使用动态计算的 order
    );
    LLVM_DEBUG({
       llvm::dbgs() << "creating tt::maketptr:\n";
        maketptrOp.dump();
    });
    // 3. 替换 tt.load 操作
  auto attrs =  filterSegmentSizes(op->getAttrs()) ;
  auto loadOp = builder.create<triton::LoadOp>(loc, descTy.getSignlessBlockType(), maketptrOp, attrs);
  loadOp.getProperties().setOperandSegmentSizes({1, 0});
  // newLoad->setAttrs(nullptr);
  LLVM_DEBUG({
    llvm::dbgs() << "creating tt::load:\n";
    loadOp->dump();
  });

  op.replaceAllUsesWith(loadOp.getResult());
  op->erase();
  
  return success();
}

LogicalResult PtrAnalysis::rewriteDescriptorStoreOp(triton::DescriptorStoreOp op) {
    OpBuilder rewriter(op);
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto indices = op.getIndices();

    // 1. 解包 descriptor
    auto desc = unpackDescriptor(descTy, op.getDesc(), rewriter);

    // 2. 新增 make_tensor_ptr
    SmallVector<int32_t> tensorShapeValues;
    for (auto dim : blockShape) {
        tensorShapeValues.push_back(static_cast<int32_t>(dim));
    }
    Value tensorPtr = rewriter.create<triton::MakeTensorPtrOp>(loc,
                                                               desc.base,               // 基址
                                                               desc.shape,              // 形状
                                                               desc.strides,            // 步长
                                                               indices,                 // 偏移
                                                               tensorShapeValues,       // tensorShape
                                                               computeOrder(blockShape) // 使用动态计算的 order
    );

    // 3. 替换 tt.store 操作
    Value valueToStore = op.getSrc();

    auto maskType = RankedTensorType::get(blockShape, rewriter.getI1Type());
    Value mask = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(maskType, true));

    // 创建属性
    auto boundaryCheck = rewriter.getDenseI32ArrayAttr({}); // 空的边界检查
    auto cacheModifier = triton::CacheModifierAttr::get(rewriter.getContext(), triton::CacheModifier::NONE);
    auto evictionPolicy = triton::EvictionPolicyAttr::get(rewriter.getContext(), triton::EvictionPolicy::NORMAL);
    auto isVolatile = rewriter.getBoolAttr(false);

    // 创建 store 操作并替换原始操作
    auto newStore = rewriter.create<triton::StoreOp>(loc,            // 要替换的操作
                                                                 tensorPtr,     // 指针
                                                                 valueToStore,  // 要存储的值
                                                                 nullptr,       // 掩码
                                                                 boundaryCheck, // 边界检查
                                                                 cacheModifier, // 缓存修饰符
                                                                 evictionPolicy // 驱逐策略
    );
    // newStore.getProperties().setOperandSegmentSizes({1, 0});
    

    // 保留原始操作的其他属性
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));
    op->replaceAllUsesWith(newStore);
    op->erase();

    return success();
}

LogicalResult PtrAnalysis::rewriteLoadOp(triton::LoadOp op) {

  // Check if tt.load is modified by AddPtrConverter to a specified state.
  if (checkModifiedByAddPtrConverter(op).succeeded()) {
    return continueModifyFromAddPtrConverter(op);
  }

  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto mask = op.getMask();
  auto other = op.getOther();
  auto loc = op.getLoc();
  OpBuilder builder(op);
  auto ptrState = knownPtrs[op.getPtr()];
  auto defaultAttr = builder.getIndexAttr(0);

  if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
    op->emitRemark("The offset value for the load operation is neither from addptr nor splat");
    return failure();
  }

  // auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  SmallVector<OpFoldResult> dims;
  SmallVector<int64_t> dimMode;
  mlir::triton::MaskState mstate;
  Value scalarOther;

  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  if (mask && generateMask(op, ptrState, dims, dimMode).failed()) {
    op.emitError("Failed to generate mask");
    return failure();
  }
  if (other) {
    assert(mask && "other value used while no masks are specified");

    scalarOther = getScalarValue(other, loc, builder);
    if (!scalarOther) {
      op->emitRemark("other value used in masked load produced by "
                     "unsupported instruction");
      return failure();
    }
  }

  auto loadOp = builder.create<tts::LoadOp>(loc, ptr, dims, dimMode, scalarOther);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::load:\n";
    loadOp->dump();
  });
  mlir::Value loadResult = loadOp.getResult();


  // auto loadArry = cast<ShapedType>(ptr.getType()).getShape();
  // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer types.
  SmallVector<int64_t> loadShape;
  if (auto ptrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
    loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
    auto loadArry = ptrType.getShape();
    loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  }


  // splat的情况下不会直接访问addptr，此时load后的元素类型正确，无需调整，因此当找不到addptr的时候说明使用了splat
  // if(!ptrState.ptrIsTensor){
  //   return rewriteScalarLoadOp(op, builder, loadResult, loc);
  // }
  if(!ptrState.ptrIsTensor){
    if(extractScalarFromLoadedTensor(op, builder, loadResult, loc).failed())
      return failure();
    op.replaceAllUsesWith(loadResult);
    op->erase();
    return success();
  }

  if(ptrState.hasBroadcast() &&
    createBroadcast(op, loadShape, loadResult).failed()){
    op->emitRemark("Failed to add broadcast");
    return failure();
  }
  if(loadShape.size() != ptrState.stateInfo.size()){
    llvm::dbgs() << "\033[34m" << "ptr::" << ptr << "\n\033[0m";
    llvm::dbgs() << "\033[34m" << "ptrState.stateInfo.size()" << ptrState.stateInfo.size() << "\n\033[0m";
    llvm::dbgs() << "\033[34m" << "state中存储的维度为: " << ptrState.stateInfo.size() << "\n\033[0m";
      llvm::dbgs() << "\033[34m" << "stride\t\tshape\t\tmask\t\tdim\n" << "\033[0m";
      for(auto x : ptrState.stateInfo){
        llvm::dbgs() << "\033[34m" << x.stride << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.shape << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.mask << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.dim << "\n\033[0m";
      }
  }
  assert(loadShape.size() == ptrState.stateInfo.size());
  if(createReshape(op, loadResult, loadShape).failed()){
    op->emitRemark("Failed to reshape load shape");
    return failure();
  }
  // 保留order
  // if(!ptrState.order.empty()){
  //   SmallVector<int64_t> permuteOrder;
  //   for(auto o: ptrState.order)permuteOrder.push_back(o);
  //   std::reverse(permuteOrder.begin(),permuteOrder.end());

  //   bool need_to_permute=false;
  //   for(auto [i, v]: llvm::enumerate(permuteOrder))
  //     if(i!=v){need_to_permute=true;break;}

  //   if(need_to_permute){
  //     SmallVector<int64_t> transposedShape(loadShape.size());
  //     for(int i=0;i<loadShape.size();i++)
  //       transposedShape[i]=loadShape[permuteOrder[i]];

  //     Value transposeInit = builder.create<tensor::EmptyOp>(
  //         loc, transposedShape, cast<RankedTensorType>(loadResult.getType()).getElementType()
  //     );
  //     loadResult = builder.create<linalg::TransposeOp>(
  //         loc, loadResult, transposeInit, permuteOrder).getResults()[0];
  //   }
  // }
  op.replaceAllUsesWith(loadResult);
  op->erase();
  return success();
}

void PtrAnalysis::initializeMaybeStructuredArgs(Operation *op) {
  std::queue<Value> q;
  DenseSet<Value> visited;

  op->walk([&q, &visited](tts::GetStructuredStateOp getStateOp) {
    Value value = getStateOp->getResult(0);
    visited.insert(value);
    q.push(value);
  });

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    for (auto user : v.getUsers()) {
      // scf.for is a special case. We have 2 set of values to consider:
      // - iter-args
      // - loop results
      // for every init arg that originates from a `tts.get_structured_state`
      // op, its corresponding iter-arg and loop result will also be considered
      // "maybeStructured".
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        auto it = llvm::find(forOp.getInitArgs(), v);

        if (it == forOp.getInitArgs().end()) {
          continue;
        }

        auto argIndex = std::distance(forOp.getInitArgs().begin(), it);
        auto iterArg = forOp.getRegionIterArg(argIndex);
        auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);

        SmallVector<Value> neighbors{iterArg, tiedLoopRes};
        for (auto neighbor : neighbors) {
          maybeStructuredArgs.insert(neighbor);
          if (!visited.contains(neighbor)) {
            visited.insert(neighbor);
            q.push(neighbor);
          }
        }

      } else {
        for (auto res : user->getResults()) {
          if (res.getType() != v.getType()) {
            continue;
          }
          maybeStructuredArgs.insert(res);
          if (!visited.contains(res)) {
            visited.insert(res);
            q.push(res);
          }
        }
      }
    }
  }
}

LogicalResult PtrAnalysis::rewriteStoreOp(triton::StoreOp op) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getValue();
  auto mask = op.getMask();
  auto loc = op.getLoc();
  auto ptrState = knownPtrs[op.getPtr()];
  OpBuilder builder(op);

  if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
    op->emitRemark("The offset value for the load operation is neither from addptr nor splat");
    return failure();
  }
  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    auto elementType = val.getType();
    if(isa<RankedTensorType>(elementType)){
      elementType = dyn_cast<RankedTensorType>(elementType).getElementType();
    }
    auto tensorType = mlir::RankedTensorType::get({1}, elementType);
    auto tensor = builder.create<tensor::FromElementsOp>(
        loc, tensorType, val
    );
    // llvm::dbgs()<<tensor<<"\n";
    val = tensor.getResult();
  }

  SmallVector<OpFoldResult> dims;
  SmallVector<int64_t> dimMode;

  // Analyze the mask operand to determine at runtime the size of the data
  // are moving.
  if (mask && generateMask(op, ptrState, dims, dimMode).failed()) {
    op.emitError("Failed to generate mask");
    return failure();
  }

  if (!isa<mlir::RankedTensorType>(val.getType())) {
      assert(val.getType().isIntOrFloat() && "only int or float scalar can be stored!");
      Value initTensor =
        builder.create<tensor::EmptyOp>(loc, SmallVector<int64_t>{1}, val.getType());
      Value filledTensor = builder.create<linalg::FillOp>(loc, ValueRange{val}, ValueRange{initTensor}).result();
      auto storeOp = builder.create<tts::StoreOp>(loc, ptr, filledTensor, dims);
      LLVM_DEBUG({
        llvm::dbgs() << "creating tts::store:\n";
        storeOp->dump();
      });
      op->erase();
      return success();
  }

  auto tensorType = cast<mlir::RankedTensorType>(val.getType());
  auto valDims = tensorType.getShape();
  int64_t valLen = 1;
  int64_t storeLen = 1;

  for (auto dim : valDims) {
      valLen *= dim;
  }

  // auto storeShape = cast<ShapedType>(ptr.getType()).getShape();
  // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer types.
  SmallVector<int64_t> storeShape;
  if (auto ptrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
    storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
    auto loadArry = ptrType.getShape();
    storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  }
  for(auto x : storeShape) storeLen *= x;


  // assert(valLen == storeLen && "Unaligned writes are not currently supported");
  if(valLen != storeLen){
    llvm::dbgs() << "\033[34m" << "valDims.size() = " << valDims.size() << "\033[0m\n";
    for(auto x :valDims){
      llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
    }
    llvm::dbgs() << "\n\033[34m" << "storeShape.size() = " << storeShape.size() << "\033[0m\n";
    for(auto x: storeShape){
      llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
    }
    llvm::dbgs() << "\n" << "\t\033[0m";
    op.emitError("Unaligned writes are not currently supported");
    return failure();
  }
  bool needReshape = false;
  for(size_t i = 0; i < valDims.size(); ++i){
    if(valDims.size() != storeShape.size() || valDims[i] != storeShape[i]){
      needReshape = true;
      break;
    }
  }
  if(needReshape){
    auto targetShapeType = RankedTensorType::get(storeShape, cast<ShapedType>(val.getType()).getElementType());
    auto targetShapeAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(storeShape.size())}, builder.getI64Type()), storeShape);
    auto targetShape = builder.create<arith::ConstantOp>(loc, targetShapeAttr);
    auto reshapeOp = builder.create<tensor::ReshapeOp>(loc, targetShapeType, val, targetShape);
    val = reshapeOp.getResult();
  }

  auto storeOp = builder.create<tts::StoreOp>(loc, ptr, val, dims);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::store:\n";
    storeOp->dump();
  });

  op->erase();
  return success();
}



LogicalResult PtrAnalysis::rewriteAtomicRMWOp(triton::AtomicRMWOp op) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getVal();
  auto mask = op.getMask();

  auto loc = op.getLoc();
  auto ptrState = knownPtrs[op.getPtr()];
  OpBuilder builder(op);

  if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
    op->emitRemark("The offset value for the atomic_rmw operation is neither from addptr nor splat");
    return failure();
  }
  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    auto elementType = val.getType();
    if(isa<RankedTensorType>(elementType)){
      elementType = dyn_cast<RankedTensorType>(elementType).getElementType();
    }
    auto tensorType = mlir::RankedTensorType::get({1}, elementType);
    auto tensor = builder.create<tensor::FromElementsOp>(
        loc, tensorType, val
    );
    // llvm::dbgs()<<tensor<<"\n";
    val = tensor.getResult();
  }

  SmallVector<OpFoldResult> dims;
  SmallVector<int64_t> dimMode;

  // Analyze the mask operand to determine at runtime the size of the data
  // are moving.
  // if (mask && generateMask(op, ptrState, dims, dimMode).failed()) {
  //   op.emitError("Failed to generate mask");
  //   return failure();
  // }
   //fixme, kaixin
  // if (!isa<mlir::RankedTensorType>(val.getType())) {
  //     assert(val.getType().isIntOrFloat() && "only int or float scalar can be stored!");
  //     Value initTensor =
  //       builder.create<tensor::EmptyOp>(loc, SmallVector<int64_t>{1}, val.getType());
  //     Value filledTensor = builder.create<linalg::FillOp>(loc, ValueRange{val}, ValueRange{initTensor}).result();
  //     auto storeOp = builder.create<tts::StoreOp>(loc, ptr, filledTensor, dims);
  //     LLVM_DEBUG({
  //       llvm::dbgs() << "creating tts::store:\n";
  //       storeOp->dump();
  //     });
  //     op->erase();
  //     return success();
  // }

  auto tensorType = cast<mlir::RankedTensorType>(val.getType());
  auto valDims = tensorType.getShape();
  int64_t valLen = 1;
  int64_t storeLen = 1;

  for (auto dim : valDims) {
      valLen *= dim;
  }

  // auto storeShape = cast<ShapedType>(ptr.getType()).getShape();
  // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer types.
  SmallVector<int64_t> storeShape;
  if (auto ptrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
    storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
    auto loadArry = ptrType.getShape();
    storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  }
  for(auto x : storeShape) storeLen *= x;


  // assert(valLen == storeLen && "Unaligned writes are not currently supported");
  if(valLen != storeLen){
    llvm::dbgs() << "\033[34m" << "valDims.size() = " << valDims.size() << "\033[0m\n";
    for(auto x :valDims){
      llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
    }
    llvm::dbgs() << "\n\033[34m" << "storeShape.size() = " << storeShape.size() << "\033[0m\n";
    for(auto x: storeShape){
      llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
    }
    llvm::dbgs() << "\n" << "\t\033[0m";
    op.emitError("Unaligned writes are not currently supported");
    return failure();
  }
  bool needReshape = false;
  for(size_t i = 0; i < valDims.size(); ++i){
    if(valDims.size() != storeShape.size() || valDims[i] != storeShape[i]){
      needReshape = true;
      break;
    }
  }
  if(needReshape){
    auto targetShapeType = RankedTensorType::get(storeShape, cast<ShapedType>(val.getType()).getElementType());
    auto targetShapeAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(storeShape.size())}, builder.getI64Type()), storeShape);
    auto targetShape = builder.create<arith::ConstantOp>(loc, targetShapeAttr);
    auto reshapeOp = builder.create<tensor::ReshapeOp>(loc, targetShapeType, val, targetShape);
    val = reshapeOp.getResult();
  }

  // TODO: need to support load/store mask generated by >|>=
  // auto atomicOp = builder.create<tts::AtomicRMWOp>(loc,
  //   builder.getStringAttr(stringifyEnum(op.getAtomicRmwOp())),
  //   ptr, val, dims,
  //   builder.getStringAttr(stringifyEnum(op.getSem())),
  //   builder.getStringAttr(stringifyEnum(op.getScope()))
  // );

  auto valType = val.getType();

  auto atomicOp = builder.create<tts::AtomicRMWOp>(loc, valType,
    op.getAtomicRmwOpAttr(), ptr, val, dims, op.getSemAttr(), op.getScopeAttr() );

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::atomic_rmw:\n";
    atomicOp->dump();
  });
  mlir::Value loadResult = atomicOp.getResult();


  // auto loadArry = cast<ShapedType>(ptr.getType()).getShape();
  // SmallVector<int64_t> loadShape(loadArry.begin(), loadArry.end());

  // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer types.
  SmallVector<int64_t> loadShape;
  if (auto ptrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
    loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
    auto loadArry = ptrType.getShape();
    loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  }


  // splat的情况下不会直接访问addptr，此时load后的元素类型正确，无需调整，因此当找不到addptr的时候说明使用了splat
  if(!ptrState.ptrIsTensor){
    if(extractScalarFromLoadedTensor(op, builder, loadResult, loc).failed())
      return failure();
    op.replaceAllUsesWith(loadResult);
    op->erase();
    return success();
  }

  if(ptrState.hasBroadcast() &&
    createBroadcast(op, loadShape, loadResult).failed()){
    op->emitRemark("Failed to add broadcast");
    return failure();
  }
  if(loadShape.size() != ptrState.stateInfo.size()){
    llvm::dbgs() << "\033[34m" << "ptr::" << ptr << "\n\033[0m";
    llvm::dbgs() << "\033[34m" << "ptrState.stateInfo.size()" << ptrState.stateInfo.size() << "\n\033[0m";
    llvm::dbgs() << "\033[34m" << "state中存储的维度为: " << ptrState.stateInfo.size() << "\n\033[0m";
      llvm::dbgs() << "\033[34m" << "stride\t\tshape\t\tmask\t\tdim\n" << "\033[0m";
      for(auto x : ptrState.stateInfo){
        llvm::dbgs() << "\033[34m" << x.stride << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.shape << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.mask << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.dim << "\n\033[0m";
      }
  }
  assert(loadShape.size() == ptrState.stateInfo.size());

  if(createReshape(op, loadResult, loadShape).failed()){
    op->emitRemark("Failed to reshape load shape");
    return failure();
  }
  // if(!ptrState.order.empty()){
  //   SmallVector<int64_t> permuteOrder;
  //   for(auto o: ptrState.order)permuteOrder.push_back(o);
  //   std::reverse(permuteOrder.begin(),permuteOrder.end());

  //   bool need_to_permute=false;
  //   for(auto [i, v]: llvm::enumerate(permuteOrder))
  //     if(i!=v){need_to_permute=true;break;}

  //   if(need_to_permute){
  //     SmallVector<int64_t> transposedShape(loadShape.size());
  //     for(int i=0;i<loadShape.size();i++)
  //       transposedShape[i]=loadShape[permuteOrder[i]];

  //     Value transposeInit = builder.create<tensor::EmptyOp>(
  //         loc, transposedShape, cast<RankedTensorType>(loadResult.getType()).getElementType()
  //     );
  //     loadResult = builder.create<linalg::TransposeOp>(
  //         loc, loadResult, transposeInit, permuteOrder).getResults()[0];
  //   }
  // }
  op.replaceAllUsesWith(loadResult);
  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteAtomicCASOp(triton::AtomicCASOp op) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getVal();
  // auto mask = op.getMask();
  auto cmd = op.getCmp();

  auto loc = op.getLoc();
  auto ptrState = knownPtrs[op.getPtr()];
  OpBuilder builder(op);

  if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
    op->emitRemark("The offset value for the atomic_rmw operation is neither from addptr nor splat");
    return failure();
  }
  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    auto elementType = val.getType();
    if(isa<RankedTensorType>(elementType)){
      elementType = dyn_cast<RankedTensorType>(elementType).getElementType();
    }
    auto tensorType = mlir::RankedTensorType::get({1}, elementType);
    auto tensor = builder.create<tensor::FromElementsOp>(
        loc, tensorType, val
    );
    val = tensor.getResult();
  }

  auto tensorType = cast<mlir::RankedTensorType>(val.getType());
  auto valDims = tensorType.getShape();
  int64_t valLen = 1;
  int64_t storeLen = 1;

  for (auto dim : valDims) {
      valLen *= dim;
  }

  // auto storeShape = cast<ShapedType>(ptr.getType()).getShape();
  // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer types.
  SmallVector<int64_t> storeShape;
  if (auto ptrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
    storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
    auto loadArry = ptrType.getShape();
    storeShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  }
  for(auto x : storeShape) storeLen *= x;


  // assert(valLen == storeLen && "Unaligned writes are not currently supported");
  if(valLen != storeLen){
    llvm::dbgs() << "\033[34m" << "valDims.size() = " << valDims.size() << "\033[0m\n";
    for(auto x :valDims){
      llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
    }
    llvm::dbgs() << "\n\033[34m" << "storeShape.size() = " << storeShape.size() << "\033[0m\n";
    for(auto x: storeShape){
      llvm::dbgs() << "\033[34m" << x << "\t\033[0m";
    }
    llvm::dbgs() << "\n" << "\t\033[0m";
    op.emitError("Unaligned writes are not currently supported");
    return failure();
  }
  bool needReshape = false;
  for(size_t i = 0; i < valDims.size(); ++i){
    if(valDims.size() != storeShape.size() || valDims[i] != storeShape[i]){
      needReshape = true;
      break;
    }
  }
  if(needReshape){
    auto targetShapeType = RankedTensorType::get(storeShape, cast<ShapedType>(val.getType()).getElementType());
    auto targetShapeAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(storeShape.size())}, builder.getI64Type()), storeShape);
    auto targetShape = builder.create<arith::ConstantOp>(loc, targetShapeAttr);
    auto reshapeOp = builder.create<tensor::ReshapeOp>(loc, targetShapeType, val, targetShape);
    val = reshapeOp.getResult();
  }

  auto valType = val.getType();
  auto atomicOp = builder.create<tts::AtomicCASOp>(loc, valType,
          ptr, cmd, val, op.getSemAttr(), op.getScopeAttr());

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::atomic_czs:\n";
    atomicOp->dump();
  });
  mlir::Value loadResult = atomicOp.getResult();

  // Support shape reading for both !ptr<tensor> and tensor<!ptr> pointer types.
  SmallVector<int64_t> loadShape;
  if (auto ptrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    auto loadArry = cast<ShapedType>(ptrType.getPointeeType()).getShape();
    loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  } else if (auto ptrType = dyn_cast<ShapedType>(ptr.getType())) {
    auto loadArry = ptrType.getShape();
    loadShape =  SmallVector<int64_t>(loadArry.begin(), loadArry.end());
  }


  // splat的情况下不会直接访问addptr，此时load后的元素类型正确，无需调整，因此当找不到addptr的时候说明使用了splat
  if(!ptrState.ptrIsTensor){
    if(extractScalarFromLoadedTensor(op, builder, loadResult, loc).failed())
      return failure();
    op.replaceAllUsesWith(loadResult);
    op->erase();
    return success();
  }

  if(ptrState.hasBroadcast() &&
    createBroadcast(op, loadShape, loadResult).failed()){
    op->emitRemark("Failed to add broadcast");
    return failure();
  }
  if(loadShape.size() != ptrState.stateInfo.size()){
    llvm::dbgs() << "\033[34m" << "ptr::" << ptr << "\n\033[0m";
    llvm::dbgs() << "\033[34m" << "ptrState.stateInfo.size()" << ptrState.stateInfo.size() << "\n\033[0m";
    llvm::dbgs() << "\033[34m" << "state中存储的维度为: " << ptrState.stateInfo.size() << "\n\033[0m";
      llvm::dbgs() << "\033[34m" << "stride\t\tshape\t\tmask\t\tdim\n" << "\033[0m";
      for(auto x : ptrState.stateInfo){
        llvm::dbgs() << "\033[34m" << x.stride << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.shape << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.mask << "\t\033[0m";
        llvm::dbgs() << "\033[34m" << x.dim << "\n\033[0m";
      }
  }
  assert(loadShape.size() == ptrState.stateInfo.size());

  if(createReshape(op, loadResult, loadShape).failed()){
    op->emitRemark("Failed to reshape load shape");
    return failure();
  }

  op.replaceAllUsesWith(loadResult);
  op->erase();
  return success();
}


/// @brief Rewrite the triton::AddPtrOp to handle unstructured memory access.
/// @param op The triton::AddPtrOp to be rewritten.
/// @param adaptor The adaptor of the triton::AddPtrOp, used to get operands.
/// @param rewriter The pattern rewriter used to modify the IR.
/// @param data The BlockData containing information about the memory access.

LogicalResult PtrAnalysis::rewriteAddPtrToUnstrucMemAcc(
    triton::AddPtrOp op,  PtrState &state) {
  OpBuilder builder(op);
  auto loc = op.getLoc();
  // auto &offsets = data.getOffsetsRef();
  auto &blockSizes = state.sizes;
  // auto &strides = data.getStridesRef();
  Value ptrOffset = op.getOffset();
  Value zeroIdx =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
  Value oneIdx =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1));
  auto addptrRes = op.getResult();
  assert(addptrRes.hasOneUse() && "Invalid: tt.addptr has multiple users");
  auto loadOp = *(addptrRes.user_begin());
  // Prepare empty tensor for loop based scalar load
  // FIXME: We use cast here because addptr must return tensor<?x!tt.ptr<f32>>.
  // True?
  auto resTy = cast<ShapedType>(addptrRes.getType());
  auto resEPtrTy = resTy.getElementType();
  auto resETy = cast<triton::PointerType>(resEPtrTy).getPointeeType();
  Value loaded = builder.create<tensor::EmptyOp>(loc, blockSizes, resETy);
  SmallVector<Value> initArgs;
  initArgs.push_back(loaded);

  SmallVector<Value> forLBs;
  SmallVector<Value> forUBs;
  SmallVector<Value> forSteps;
  for (auto &s : state.stateInfo) {
    forLBs.push_back(zeroIdx);
  }
  for (auto &s : state.sizes) {
    forUBs.push_back(getValueOrCreateConstantIndexOp(builder, loc, s));
  }
  for (auto &s : state.stateInfo) {
    forSteps.push_back(oneIdx);
  }
  SmallVector<Value> ivs;
  auto loop = createNestedLoops(
      builder, loc, 0, blockSizes.size(), forLBs, forUBs, forSteps, ivs,
      initArgs,
      [&](OpBuilder &bB, Location bLoc, SmallVector<Value> &allIVs,
          ValueRange iterArgs) {
        OpBuilder::InsertionGuard g(bB);
        bB.setInsertionPointToStart(bB.getBlock());

        Value scalarOffsetRaw =
            bB.create<tensor::ExtractOp>(bLoc, ptrOffset, allIVs);
        Value scalarOffset = bB.create<arith::IndexCastOp>(
            bLoc, bB.getIndexType(), scalarOffsetRaw);
        auto defaultAttr = builder.getIndexAttr(0);

        auto stride = builder.getIndexAttr(1);
        auto shape = builder.getIndexAttr(1);
        PtrState state_tts;
        StateInfo newStateInfo(scalarOffset, stride, shape, defaultAttr);
        state_tts.stateInfo.push_back(newStateInfo);
        state_tts.sizes.push_back(shape);
        state_tts.source = state.source ;
        auto maketptrOp = state_tts.createTTSMakeTensorPtrOp(builder, op.getLoc());
        ptrMap.map(op.getResult(), maketptrOp.getResult());
        // loadOp->moveAfter(maketptrOp);
        loadOp->setAttr("IndirectLoad", UnitAttr::get(op.getContext()));
        bB.create<scf::YieldOp>(bLoc, iterArgs);
      });

    return success();
}

/// @brief Check whether the triton::LoadOp has been modified to the specified
/// state by the AddPtrConverter.
/// @param op The triton::LoadOp operation to be checked.
/// @return Return success if the operation conforms to the specified state;
/// otherwise, return failure.
LogicalResult
PtrAnalysis::checkModifiedByAddPtrConverter(triton::LoadOp &op) const {
  // if (!isa<scf::ForOp>(op->getParentOp())) {
  //   return failure();
  // }
  if (!op->hasAttr("IndirectLoad")) {
    return failure();
  }
  // auto ptrOp = op.getPtr().getDefiningOp();
  // auto ptrBlock = ptrOp->getBlock();
  // auto opBlock = op->getBlock();
  // if (ptrBlock == opBlock) {
  //   return failure();
  // }

  return success();
}

/// @brief Continue to modify the triton::LoadOp from the state modified by the
/// AddPtrConverter.
/// @param op The triton::LoadOp operation to be processed.
/// @param adaptor The adaptor for the operation, used to obtain operands.
/// @param rewriter The pattern rewriter used to rewrite the operation.
/// @return Return success if the operation is successful; otherwise, return
/// failure.
LogicalResult PtrAnalysis::continueModifyFromAddPtrConverter(triton::LoadOp &op)  {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto mask = op.getMask();
  auto other = op.getOther();
  //auto loc = op.getLoc();

  auto ptrState = knownPtrs[op.getPtr()];
  auto defaultAttr = builder.getIndexAttr(0);

  if (!ptr && analysisSplat(op, builder, ptr, ptrState).failed()) {
    op->emitRemark("The offset value for the load operation is neither from addptr nor splat");
    return failure();
  }

  // auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  SmallVector<OpFoldResult> dims;
  SmallVector<int64_t> dimMode;
  mlir::triton::MaskState mstate;
  Value scalarOther;

  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  if (mask && generateMask(op, ptrState, dims, dimMode).failed()) {
    op.emitError("Failed to generate mask");
    return failure();
  }
  if (other) {
    assert(mask && "other value used while no masks are specified");

    scalarOther = getScalarValue(other, loc, builder);
    if (!scalarOther) {
      op->emitRemark("other value used in masked load produced by "
                     "unsupported instruction");
      return failure();
    }
  }

  auto ttsPtrOp = ptr.getDefiningOp();
  auto forOp = ttsPtrOp->getParentOfType<scf::ForOp>();
  Operation *firstOp = &forOp.getBody()->front();
  auto extractOp = cast<tensor::ExtractOp>(firstOp);
  auto ivs = extractOp.getIndices();
  auto iterArg = forOp.getRegionIterArg(0);

  builder.setInsertionPointAfter(ttsPtrOp);
  auto loadVal = builder.create<tts::LoadOp>(loc, ptr, dims, dimMode, scalarOther);

  Value idxZero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto loadScalar = builder.create<tensor::ExtractOp>(loc, loadVal, idxZero);
  // Value castVal = ptr.getDefiningOp<memref::ReinterpretCastOp>();
  // Value loadVal =
  //     rewriter.create<memref::LoadOp>(loc, castVal, ValueRange{idxZero});

  Value insertedVal =
      builder.create<tensor::InsertOp>(loc, loadScalar, iterArg, ValueRange{ivs});
  // // a yield op is already created by AddPtrConverter.
  // // so we need to replace it with a new yield op.
  Operation *terminator = forOp.getBody()->getTerminator();
  scf::YieldOp oldYieldOp = cast<scf::YieldOp>(terminator);
  // // auto yieldOp = builder.create<scf::YieldOp>(loc, ValueRange{insertedVal});
  oldYieldOp->setOperand(0, insertedVal);
  // // oldYieldOp->erase();
  // //builder.replaceOp(oldYieldOp, yieldOp);
  // // Now the scf.for is complete, we can replace tt.load with it.
  auto rank = cast<ShapedType>(op.getResult().getType()).getShape().size();
  Operation *rootForOp = oldYieldOp;
  while (rank != 0) {
    rank--;
    rootForOp = rootForOp->getParentOfType<scf::ForOp>();
  }
  op.replaceAllUsesWith(rootForOp->getResult(0));

  return success();
}


LogicalResult PtrAnalysis::walk(Operation *rootOp) {
  LogicalResult ret = success();
  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    LLVM_DEBUG({
      llvm::dbgs() << "walking Op\n";
      op->dump();
    });
    if (op == rootOp) {
      return WalkResult::advance();
    }
    // if (isa<scf::YieldOp>(op)) {
    //   return WalkResult::interrupt();
    // }
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<triton::AddPtrOp>([&](auto addptr) {
          LLVM_DEBUG({
              llvm::dbgs() << "TypeSwitch AddPtrOp\n";
              addptr->dump();
              rootOp->dump();
          });
          if (rewriteAddptrOp(addptr).failed()) {
            addptr->emitRemark("PtrAnalysis: Failed to rewrite AddPtrOp");
          }
          return WalkResult::advance();
        })
        .Case<triton::MakeTensorPtrOp>([&](auto maketptr) {
          if (rewriteMakeTensorPtrOp(maketptr).failed()) {
            maketptr->emitRemark(
                "PtrAnalysis: Failed to rewrite MakeTensorPtrOp");
          }
          return WalkResult::advance();
        })
        .Case<triton::AdvanceOp>([&](auto advance) {
          if (rewriteAdvanceOp(advance).failed()) {
            advance->emitRemark("PtrAnalysis: Failed to rewrite AdvanceOp");
          }
          return WalkResult::advance();
        })
        .Case<triton::LoadOp>([&](auto load) {
          if (rewriteLoadOp(load).failed()) {
            load->emitRemark("PtrAnalysis: Failed to rewrite LoadOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::DescriptorLoadOp>([&](auto descLoad) {
          ret = failure() ;
          if (rewriteDescriptorLoadOp(descLoad).failed()) {
            descLoad->emitRemark("PtrAnalysis: Failed to rewrite DescriptorLoadOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::DescriptorStoreOp>([&](auto descStore) {
          ret = failure() ;
          if (rewriteDescriptorStoreOp(descStore).failed()) {
            descStore->emitRemark("PtrAnalysis: Failed to rewrite DescriptorStoreOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::StoreOp>([&](auto store) {
          if (rewriteStoreOp(store).failed()) {
            store->emitRemark("PtrAnalysis: Failed to rewrite StoreOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::AtomicRMWOp>([&](auto atomic_rmw) {
          if (rewriteAtomicRMWOp(atomic_rmw).failed()) {
            atomic_rmw->emitRemark("PtrAnalysis: Failed to rewrite AtomicRMWOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::AtomicCASOp>([&](auto atomic_cas) {
          if (rewriteAtomicCASOp(atomic_cas).failed()) {
            atomic_cas->emitRemark("PtrAnalysis: Failed to rewrite AtomicCASOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<scf::ForOp>([&](auto forOp) {
          LLVM_DEBUG({
             llvm::dbgs() << "before rewriteForOp\n";
             op->dump();
          });
          if (rewriteForOp(forOp).failed()) {
            forOp->emitRemark("PtrAnalysis: Failed to rewrite ForOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Default([&](auto) { return WalkResult::advance(); });
  });

  return ret ;
  
}

LogicalResult PtrAnalysis::rewriteOp(Operation *rootOp, bool useUnsafeMask) {
  LLVM_DEBUG({
    llvm::dbgs() << "rewriting rootOp\n";
    rootOp->dump();
  });
  
  for (int i = 0; i < 2; i++) {
    if(walk(rootOp).succeeded()) 
        break;
  }

  return success();
}

void PtrState::setMemAccTy(const MemAccType &v) { this->memAccTy = v; }

void PtrState::setMemAccVal(const MemAccVal v) { this->memAccTy.value = v; }



} // namespace tts
} // namespace mlir
