//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//


#include "Utils/Utils.h"
#include "Conversion/ConversionCommon.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "Conversion/TritonToStructured/TritonToStructured.h"

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

namespace mlir{
namespace triton{

void rewriteUserWithNewOrder(mlir::Operation *user, PatternRewriter &rewriter, llvm::SmallVector<int64_t, 8> &blkShapeI64, // 8: container size
                             mlir::Location &loc, llvm::ArrayRef<int32_t> &order, size_t &orderSize)
{
  rewriter.setInsertionPointAfter(user);
  if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
    auto loadResTy = loadOp.getResult().getType();
    auto loadResShapedTy = cast<ShapedType>(loadResTy);
    auto newLoadTy = loadResShapedTy.cloneWith(
        blkShapeI64, loadResShapedTy.getElementType());
    auto newLoadOp = rewriter.create<triton::LoadOp>(
        loc, newLoadTy, loadOp->getOperands(), loadOp->getAttrs());
    rewriter.replaceOp(loadOp, newLoadOp);
    // load contiguous data then permute. thus the permute order is as
    // follows.
    SmallVector<int32_t, 8> permuteOrder; // 8: container size
    for (auto [i, v] : llvm::enumerate(order)) {
      permuteOrder.push_back(orderSize - 1 - order[i]);
    }
    auto permuteOp = rewriter.create<triton::TransOp>(
        loc, newLoadOp.getResult(),
        DenseI32ArrayAttr::get(loadOp.getContext(), permuteOrder));
    newLoadOp.getResult().replaceAllUsesExcept(permuteOp.getResult(), permuteOp);
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
    // permute to contiguous then store. thus the permute order is as follows.
    SmallVector<int32_t, 8> permuteOrder; // 8: container size
    for (auto [i, v] : llvm::enumerate(order)) {
      permuteOrder.push_back(order[orderSize - 1 - i]);
    }
    auto permuteOp = rewriter.create<triton::TransOp>(
        loc, storeOp.getValue(),
        DenseI32ArrayAttr::get(storeOp.getContext(), permuteOrder));
    storeOp.getValue().replaceAllUsesExcept(permuteOp.getResult(), permuteOp);
    auto newStoreOp = rewriter.create<triton::StoreOp>(
        loc, storeOp.getPtr(), storeOp.getValue(), storeOp.getMask(),
        storeOp.getBoundaryCheck(), storeOp.getCache(), storeOp.getEvict());
    rewriter.replaceOp(storeOp, newStoreOp);
  } else {
    auto advanceOp = dyn_cast<triton::AdvanceOp>(user);
    auto advanceResPtrTy =
        cast<triton::PointerType>(advanceOp.getResult().getType());
    auto advanceResShapedTy =
        cast<ShapedType>(advanceResPtrTy.getPointeeType());
    auto newAdvanceResShapedTy = advanceResShapedTy.cloneWith(
        blkShapeI64, advanceResShapedTy.getElementType());
    auto newAdvanceResPtrTy = triton::PointerType::get(
        newAdvanceResShapedTy, advanceResPtrTy.getAddressSpace());
    auto advanceOffsets = advanceOp.getOffsets();
    llvm::SmallVector<Value, 8> newAdvanceOffsets; // 8: container size
    for (int i = orderSize - 1; i >= 0; i--) {
      newAdvanceOffsets.push_back(advanceOffsets[order[i]]);
    }
    auto newAdvanceOp = rewriter.create<triton::AdvanceOp>(
        loc, newAdvanceResPtrTy, advanceOp.getPtr(), newAdvanceOffsets);
    rewriter.replaceOp(advanceOp, newAdvanceOp);
  }
}

void setBlockArgumentAttr(BlockArgument blockArg, triton::FuncOp func, TensorKind tensorKind)
{
    unsigned argIdx = blockArg.getArgNumber();
    auto existingAttr = func.getArgAttrOfType<IntegerAttr>(argIdx, "tt.tensor_kind");
    TensorKind oldVal = existingAttr ? static_cast<TensorKind>(existingAttr.getInt()) : TensorKind::NONE;

    TensorKind finalVal = tensorKind;
    if ((oldVal == TensorKind::INPUT && tensorKind == TensorKind::OUTPUT) ||
        (oldVal == TensorKind::OUTPUT && tensorKind == TensorKind::INPUT)) {
        finalVal = TensorKind::INPUT_OUTPUT;
    } else if (oldVal == TensorKind::INPUT_OUTPUT) {
        finalVal = oldVal;
    }

    func.setArgAttr(argIdx, "tt.tensor_kind",
                    IntegerAttr::get(IntegerType::get(func.getContext(), INT_BIT_WIDTH), static_cast<int>(finalVal)));
}

LogicalResult
MakeTensorPtrCanonicalizer::matchAndRewrite(triton::MakeTensorPtrOp op,
                                            PatternRewriter &rewriter) const {

  auto order = op.getOrder();
  auto orderSize = order.size();
  if (orderSize == 1) {
    return rewriter.notifyMatchFailure(
        op, "make_tensor_ptr's order has single value.");
  }

  bool isPermuted = false;
  for (auto [first, second] : llvm::zip(order.slice(0, orderSize - 1),
                                        order.slice(1, orderSize - 1))) {
    if (first != second + 1) {
      isPermuted = true;
      break;
    }
  }
  if (!isPermuted) {
    return rewriter.notifyMatchFailure(
        op, "make_tensor_ptr's order is contiguous.");
  }

  auto loc = op.getLoc();
  auto base = op.getBase();
  auto shape = op.getShape();
  auto strides = op.getStrides();
  auto offsets = op.getOffsets();
  auto result = op.getResult();
  auto opUsers = result.getUsers();
  for (auto user : opUsers) {
    if (isa<scf::ForOp>(user)) {
      auto forOp = dyn_cast<scf::ForOp>(user);
      auto initArgs = forOp.getInitArgs();
      for (auto [idx, initArg] : llvm::enumerate(initArgs)) {
        if (initArg == result) {
          auto iterArgs = forOp.getRegionIterArgs();
          auto targetIterArg = iterArgs[idx];
          auto argUsers = targetIterArg.getUsers();
          for (auto argUser : argUsers) {
            if (!isa<triton::LoadOp>(argUser) && !isa<triton::StoreOp>(argUser) &&
                !isa<triton::AdvanceOp>(argUser)) {
              return rewriter.notifyMatchFailure(forOp,
                                                 "[MakeTensorPtrCanonicalizer] scf.for's arg is "
                                                 "not used by load/store/advance op");
            }
          }
        }
      }
    } else if (!isa<triton::LoadOp>(user) && !isa<triton::StoreOp>(user) &&
        !isa<triton::AdvanceOp>(user)) {
      return rewriter.notifyMatchFailure(
          op, "[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
              "not used by load/store/advance op");
    };
  }

  llvm::SmallVector<int32_t, 8> blkShapeI32;
  llvm::SmallVector<int64_t, 8> blkShapeI64;
  auto resPtrType = cast<triton::PointerType>(result.getType());
  if (auto resShapedTy = dyn_cast<ShapedType>(resPtrType.getPointeeType())) {
    auto resBlkShape = resShapedTy.getShape();
    for (auto [i, v] : llvm::enumerate(resBlkShape)) {
      auto reverseI = orderSize - 1 - i;
      blkShapeI32.push_back(resBlkShape[order[reverseI]]);
      blkShapeI64.push_back(resBlkShape[order[reverseI]]);
    }
  }

  llvm::SmallVector<Value, 8> newShape;
  llvm::SmallVector<Value, 8> newStrides;
  llvm::SmallVector<Value, 8> newOffsets;
  for (int i = orderSize - 1; i >= 0; i--) {
    newShape.push_back(shape[order[i]]);
    newStrides.push_back(strides[order[i]]);
    newOffsets.push_back(offsets[order[i]]);
  }

  llvm::SmallVector<int, 8> contiguousOrder;
  for (int i = orderSize - 1; i >= 0; i--)
    contiguousOrder.push_back(i);

  rewriter.setInsertionPoint(op);
  auto newMakeTensorPtrOp = rewriter.create<triton::MakeTensorPtrOp>(
      loc, base, ValueRange(newShape), ValueRange(newStrides),
      ValueRange(newOffsets), blkShapeI32, contiguousOrder);
  rewriter.replaceOp(op, newMakeTensorPtrOp);

  for (auto user : opUsers) {
    if (isa<scf::ForOp>(user)) {
      auto forOp = dyn_cast<scf::ForOp>(user);
      auto initArgs = forOp.getInitArgs();
      for (auto [idx, initArg] : llvm::enumerate(initArgs)) {
        if (initArg == newMakeTensorPtrOp.getResult()) {
          auto iterArgs = forOp.getRegionIterArgs();
          auto targetIterArg = iterArgs[idx];
          auto argUsers = targetIterArg.getUsers();
          for (auto argUser : argUsers) {
            rewriteUserWithNewOrder(argUser, rewriter, blkShapeI64, loc, order, orderSize);
          }
        }
      }
    } else {
      rewriteUserWithNewOrder(user, rewriter, blkShapeI64, loc, order, orderSize);
    }
  }

  return success();
}


LogicalResult
ScalarStoreCanonicalizer::matchAndRewrite(triton::StoreOp op,
                                          PatternRewriter &rewriter) const {
  if (!op.getValue().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarStoreCanonicalizer handles scalar store scene!");
  }

  auto ptr = op.getPtr();
  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto valTy = RankedTensorType::get({(int64_t)1}, op.getValue().getType());
  auto valSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), valTy, op.getValue());

  auto newStoreOp = rewriter.create<triton::StoreOp>(
      op.getLoc(), ptrSplat, valSplat, op.getCache(), op.getEvict());
  rewriter.replaceOp(op, newStoreOp);
  return success();
}

LogicalResult
ScalarAtomicRMWCanonicalizer::matchAndRewrite(triton::AtomicRMWOp op,
                                              PatternRewriter &rewriter) const {
  if (!op.getVal().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarAtomicRMWCanonicalizer handles scalar atomic rmw op scene!");
  }

  auto ptr = op.getPtr();
  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto valTy = RankedTensorType::get({(int64_t)1}, op.getVal().getType());
  auto valSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), valTy, op.getVal());
  auto maskTy = RankedTensorType::get({(int64_t)1}, op.getMask().getType());
  auto maskSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), maskTy, op.getMask());

  auto newAtomicOp = rewriter.create<triton::AtomicRMWOp>(
      op.getLoc(), valTy, op.getAtomicRmwOp(), ptrSplat, valSplat, maskSplat,
      op.getSem(), op.getScope());
  rewriter.replaceOp(op, newAtomicOp);
  return success();
}

LogicalResult
ScalarAtomicCASCanonicalizer::matchAndRewrite(triton::AtomicCASOp op,
                                              PatternRewriter &rewriter) const {
  if (!op.getVal().getType().isIntOrIndexOrFloat() && !op.getCmp().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarAtomicCASCanonicalizer handles scalar atomic cas op scene!");
  }

  auto ptr = op.getPtr();
  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto cmpTy = RankedTensorType::get({(int64_t)1}, op.getCmp().getType());
  auto cmpSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), cmpTy, op.getCmp());
  auto valTy = RankedTensorType::get({(int64_t)1}, op.getVal().getType());
  auto valSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), valTy, op.getVal());

  auto newAtomicOp = rewriter.create<triton::AtomicCASOp>(
      op.getLoc(), valTy, ptrSplat, cmpSplat, valSplat,
      op.getSem(), op.getScope());
  rewriter.replaceOp(op, newAtomicOp);
  return success();
}

LogicalResult
AtomicMaxMinCanonicalizer::matchAndRewrite(triton::AtomicRMWOp op,
                                           PatternRewriter &rewriter) const {
  // Revert the op to its original form
  auto ptrBitcastOp = op.getPtr().getDefiningOp<triton::BitcastOp>();
  auto valueBitcastOp = op.getVal().getDefiningOp<triton::BitcastOp>();
  if (!ptrBitcastOp || !valueBitcastOp) {
    return failure();
  }

  // We only need to handle the op when the element type is float
  auto elementType =
      dyn_cast<TensorType>(valueBitcastOp.getSrc().getType()).getElementType();
  if (!isa<FloatType>(elementType)) {
    return failure();
  }

  auto rmwOp = op.getAtomicRmwOp();
  // here we know that atomic UMAX/UMIN
  // is created by special logic of triton right now
  // so we can simply delete it
  if (rmwOp == triton::RMWOp::UMAX || rmwOp == triton::RMWOp::UMIN) {
    // if the return value of op is used, we can't simply erase it
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }

  if (rmwOp != triton::RMWOp::MAX && rmwOp != triton::RMWOp::MIN) {
    return failure();
  }

  // 1. Though semantic interpreter will generate full true tensor as original
  // mask if atomicrmwOp don't have it, above float devision process will also
  // generate positive and negative comparison mask, which will cause to fold
  // true mask.
  // 2. While if atomicrmwOp has original mask, there exists andiop between
  // original mask and positive/negative comparison mask
  //
  // Here wanna extract original mask
  Value originalMask = op.getMask();
  if (auto andOp = originalMask.getDefiningOp<arith::AndIOp>())
    // LHS is convention in semantic interpreter
    originalMask = andOp.getLhs();
  else if (auto cmpOp = originalMask.getDefiningOp<arith::CmpFOp>()) {
    if (cmpOp.getPredicate() != mlir::arith::CmpFPredicate::OGE ||
        !matchPattern(cmpOp.getRhs(),
                      /*positive float zero matcher*/ m_PosZeroFloat()))
      // Here recheck frontend interpreter generation in no manual mask state
      return op->emitError("Illegal mask for atomicrmwOp of float type");
    // Restore original true mask
    originalMask = rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        /*typed attr*/ DenseElementsAttr::get(
            cast<ShapedType>(originalMask.getType()), true));
  } else
    return op->emitError("Illegal mask for atomicrmwOp of float type");

  auto originAtomicOp = rewriter.create<triton::AtomicRMWOp>(
      op.getLoc(), valueBitcastOp.getSrc().getType(), op.getAtomicRmwOp(),
      ptrBitcastOp.getSrc(), valueBitcastOp.getSrc(), originalMask, op.getSem(),
      op.getScope());

  // if the return value of op is used
  // we need to handle its usage
  // In semantic.py, if the atomic Max/Min with float input is used
  // It will use select + bitcast to get float value
  // so here we need to revert it too
  //
  // For example:
  // %0 = tt.atomic_rmw max, acq_rel, gpu, %gm, %input, %mask1 :
  // (tensor<32x!tt.ptr<i32>>... %1 = tt.atomic_rmw umin, acq_rel, gpu, %gm,
  // %input, %mask2 : (tensor<32x!tt.ptr<i32>>... %2 = arith.select
  // %devidedMask, %0, %1 : tensor<32xi1>, tensor<32xi32> %3 = tt.bitcast %2 :
  // tensor<32xi32> -> tensor<32xf32> tt.store %outputMemref, %3 :
  // tensor<32x!tt.ptr<f32>>
  //
  // will be revert to:
  // %0 = tt.atomic_rmw max, acq_rel, gpu, %gm, %input, %mask :
  // (tensor<32x!tt.ptr<f32>>... tt.store %outputMemref, %0 :
  // tensor<32x!tt.ptr<f32>>
  //
  if (!op.getResult().use_empty()) {
    for (OpOperand &use : op->getUses()) {
      auto selectOp = dyn_cast<arith::SelectOp>(use.getOwner());
      if (!selectOp)
        continue;

      for (OpOperand &selectUse : selectOp->getUses()) {
        if (auto bitcastOp =
                dyn_cast<triton::BitcastOp>(selectUse.getOwner())) {
          bitcastOp.getResult().replaceAllUsesWith(originAtomicOp);
        }
      }
    }
    rewriter.replaceOp(op, originAtomicOp);
  } else {
    rewriter.eraseOp(op);
  }

  return success();
}

bool hasATensorDescriptorType(mlir::TypeRange types)
{
    return llvm::any_of(types, [](mlir::Type t) { return llvm::isa<mlir::triton::TensorDescType>(t); });
}

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute> filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs)
{
    mlir::SmallVector<NamedAttribute> ret;
    llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
        auto attrName = attr.getName().getValue();
        return attrName != "operandSegmentSizes";
    });
    return ret;
}

Descriptor unpackDescriptor(TensorDescType type, Value desc, ConversionPatternRewriter &rewriter)
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

LogicalResult DescriptorLoadConverter::matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                                                       ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto resultTy = descTy.getSignlessBlockType();
    auto indices = op.getIndices();

    // 1. 解包 descriptor
    auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

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
    // 3. 替换 tt.load 操作

    //auto newLoad = rewriter.replaceOpWithNewOp<triton::LoadOp>(op, descTy.getSignlessBlockType(), tensorPtr);
    auto attrs = filterSegmentSizes(op->getAttrs()) ;
    auto newLoadOp = rewriter.create<triton::LoadOp>( loc, descTy.getSignlessBlockType(), 
                                     tensorPtr, attrs);

    rewriter.replaceOp(op, newLoadOp) ;
    //op->erase() ;
    return success();
}

LogicalResult DescriptorStoreConverter::matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                                                        ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto indices = op.getIndices();

    // 1. 解包 descriptor
    auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

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
    Value valueToStore = adaptor.getSrc();

    auto maskType = RankedTensorType::get(blockShape, rewriter.getI1Type());
    Value mask = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(maskType, true));

    // 创建属性
    auto boundaryCheck = rewriter.getDenseI32ArrayAttr({}); // 空的边界检查
    auto cacheModifier = triton::CacheModifierAttr::get(rewriter.getContext(), triton::CacheModifier::NONE);
    auto evictionPolicy = triton::EvictionPolicyAttr::get(rewriter.getContext(), triton::EvictionPolicy::NORMAL);
    auto isVolatile = rewriter.getBoolAttr(false);

    // 创建 store 操作并替换原始操作
    auto newStore = rewriter.create<triton::StoreOp>( loc,
                                                                tensorPtr,     // 指针
                                                                 valueToStore,  // 要存储的值
                                                                 nullptr,       // 掩码
                                                                 boundaryCheck, // 边界检查
                                                                 cacheModifier, // 缓存修饰符
                                                                 evictionPolicy // 驱逐策略
    );

    // 保留原始操作的其他属性
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    rewriter.replaceOp(op, newStore) ;
    op->erase();
    return success();
}

}
}
