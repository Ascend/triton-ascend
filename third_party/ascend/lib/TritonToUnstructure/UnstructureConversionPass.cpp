/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"


#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "triton-unstructure-converter"

using namespace mlir;
using namespace triton;

#include "llvm/Support/Debug.h"

bool forceSimtTemplateFlag = false;

template <typename MemAccOpTy>
bool UnstructuredMemAccessConverter<MemAccOpTy>::checkUnstructureAnnotated(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  return llvm::any_of(op->getUsers(), [&rewriter](Operation *user) {    
    auto annotationOp = dyn_cast<annotation::MarkOp>(user);
    if (annotationOp && annotationOp->hasAttr("mayDiscretememaccess")) {
      rewriter.eraseOp(annotationOp);
      return true;
    }
    return false;
  });
}

template <>
bool UnstructuredMemAccessConverter<triton::StoreOp>::checkUnstructureAnnotated(
    triton::StoreOp op, PatternRewriter &rewriter) const {
  return llvm::any_of(op.getValue().getUsers(), [&rewriter](Operation *user) {   
    auto annotationOp = dyn_cast<annotation::MarkOp>(user);
    if (annotationOp && annotationOp->hasAttr("mayDiscretememaccess")) {
      rewriter.eraseOp(annotationOp);
      return true;
    }
    return false;
  });
}

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> iterIdx) const {
  if (!value)
    return value;
  SmallVector<Value> indices;
  for (auto idx : iterIdx) {
    if (auto val = dyn_cast<Value>(idx)) {
      indices.push_back(val);
    } else {
      auto idxVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(*getConstantIntValue(idx)));
      indices.push_back(idxVal);
    }
  }
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, value, indices);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides) const {
  if (!value)
    return value;
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracting\n";
    os << value << "\n";
  });
  auto extractedOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, value, offsets, sizes, strides);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <>
template <typename... Args>
triton::LoadOp UnstructuredMemAccessConverter<triton::LoadOp>::createMemAccOp(
    triton::LoadOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  return rewriter.create<triton::LoadOp>(loc, ptrToAccess, op.getCache(),
                                         op.getEvict(), op.getIsVolatile());
}

template <>
template <typename... Args>
triton::AtomicRMWOp
UnstructuredMemAccessConverter<triton::AtomicRMWOp>::createMemAccOp(
    triton::AtomicRMWOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedValue =
      createExtractOp(loc, op.getVal(), rewriter, std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  Type targetType = ptrToAccess.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    targetType =
        RankedTensorType::get(tensorType.getShape(), ptrType.getPointeeType());
  } else {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
    targetType =
        RankedTensorType::get(scalarLikeShape, resultType.getElementType());
    ptrToAccess = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
        ptrToAccess);
    extractedValue = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
        extractedValue);
    if (extractedMask) {
      extractedMask = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, extractedMask.getType()),
          extractedMask);
    }
  }
  return rewriter.create<triton::AtomicRMWOp>(
      loc, targetType, op.getAtomicRmwOpAttr(), ptrToAccess, extractedValue,
      extractedMask, op.getSemAttr(), op.getScopeAttr());
}

template <>
template <typename... Args>
triton::AtomicCASOp
UnstructuredMemAccessConverter<triton::AtomicCASOp>::createMemAccOp(
    triton::AtomicCASOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedCmp =
      createExtractOp(loc, op.getCmp(), rewriter, std::forward<Args>(args)...);
  auto extractedValue =
      createExtractOp(loc, op.getVal(), rewriter, std::forward<Args>(args)...);
  Type targetType = ptrToAccess.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    targetType =
        RankedTensorType::get(tensorType.getShape(), ptrType.getPointeeType());
  } else {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
    targetType =
        RankedTensorType::get(scalarLikeShape, resultType.getElementType());
    ptrToAccess = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
        ptrToAccess);
    extractedCmp = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedCmp.getType()),
        extractedCmp);
    extractedValue = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
        extractedValue);
  }
  return rewriter.create<triton::AtomicCASOp>(
      loc, targetType, ptrToAccess, extractedCmp, extractedValue,
      op.getSemAttr(), op.getScopeAttr());
}

template <>
template <typename... Args>
triton::StoreOp UnstructuredMemAccessConverter<triton::StoreOp>::createMemAccOp(
    triton::StoreOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedValue = createExtractOp(loc, op.getValue(), rewriter,
                                        std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  return rewriter.create<triton::StoreOp>(loc, ptrToAccess, extractedValue,
                                          extractedMask);
}

template <>
template <>
void UnstructuredMemAccessConverter<triton::LoadOp>::splatAndLoadScenario<
    triton::LoadOp>(triton::LoadOp op, int rank,
                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  SmallVector<OpFoldResult> idx(rank, rewriter.getIndexAttr(0));
  auto extractedPtr = createExtractOp(loc, op.getPtr(), rewriter, idx);
  Value mask = op.getMask();
  Value other = op.getOther();
  Value loadedValue = rewriter.create<triton::LoadOp>(
      loc, extractedPtr, /*mask=*/nullptr, /*other=*/nullptr,
      /*boundaryCheck=*/ArrayRef<int32_t>(),
      /*PaddingOptionAttr=*/nullptr);
  loadedValue = rewriter.create<triton::SplatOp>(loc, op.getResult().getType(),
                                                 loadedValue);
  if (mask)
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, loadedValue, other);
  else
    rewriter.replaceOp(op, loadedValue);
}

template <typename MemAccOpTy>
UnstructuredMemAccessConverter<MemAccOpTy>::UnstructuredMemAccessConverter(
    MLIRContext *context, bool forceScalarizeMode,
    const llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
    const llvm::SmallDenseMap<Value, bool> &fromTensorArg)
    : OpRewritePattern<MemAccOpTy>(context),
      forceScalarizeMode(forceScalarizeMode), offsetMap(offsetMap), fromTensorArg(fromTensorArg) {}

template <typename MemAccOpTy>
LogicalResult UnstructuredMemAccessConverter<MemAccOpTy>::matchAndRewrite(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto ptr = op.getPtr();
  auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());

  if (auto ptrPtrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    if (auto ptrTensorType =
            dyn_cast_or_null<RankedTensorType>(ptrPtrType.getPointeeType()))
      ptrType = ptrTensorType;
  }

  if (!ptrType || op->hasAttr(ConverterUtils::discreteAttrName))
    return failure();
  if (!offsetMap.contains(ptr))
    return op.emitError() << "PtrOffsetInfo should be computed\n" << ptr;

  auto ptrOffsetInfo = offsetMap.at(ptr);

  if (checkUnstructureAnnotated(op, rewriter))
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());

  if (ptrOffsetInfo.isStructured() &&
      (!ptrOffsetInfo.isScalarLike() ||
       llvm::all_of(ptrType.getShape(), [](int64_t dim) { return dim == 1; })))
    return failure();

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Converting " << op->getName() << "\n";
    os << op << "\n";
    os << ptrOffsetInfo.isStructured() << "\n";
    os << ptrOffsetInfo.isScalarLike() << "\n";
  });

  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    if (ptrOffsetInfo.isScalarLike()) {
      splatAndLoadScenario(op, ptrOffsetInfo.getRank(), rewriter);
      return success();
    }
  }

  std::optional<MaskState> mstate = runMaskAnalysis(op, rewriter);

  if (op->hasAttr(ConverterUtils::discreteMaskAttrName)) {
    if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      auto selectOp = op.getValue().template getDefiningOp<arith::SelectOp>();
      op = rewriter.replaceOpWithNewOp<triton::StoreOp>(
          op, op.getPtr(), selectOp.getTrueValue(), selectOp.getCondition(),
          op.getCache(), op.getEvict());
      rewriter.setInsertionPoint(op);
      ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
    } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
      auto selectOp = op.getVal().template getDefiningOp<arith::SelectOp>();
      op = rewriter.replaceOpWithNewOp<triton::AtomicRMWOp>(
          op, op.getType(), op.getAtomicRmwOp(), op.getPtr(),
          selectOp.getTrueValue(), selectOp.getCondition(), op.getSem(),
          op.getScope());
    }
    rewriter.setInsertionPoint(op);
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  }

  if (forceScalarizeMode || ptrOffsetInfo.isScalarLike() ||
      fromTensorArg.at(ptr)) {
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  }

  auto srcPtr = ptrOffsetInfo.getPtr();
  auto ptrOffset = ptrOffsetInfo.getOffset();

  // LoadLike is operation with result
  bool isLoadLike = !op->use_empty();

  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto resultShape = ptrType.getShape();
  auto resultElementType = ptrType.getElementType();
  if (auto pointerType =
          dyn_cast<triton::PointerType>(ptrType.getElementType())) {
    resultElementType = pointerType.getPointeeType();
  }

  int64_t sizeInByte;
  if (auto intType = dyn_cast<IntegerType>(resultElementType)) {
    sizeInByte = intType.getWidth() / 8;
  } else if (auto floatType = dyn_cast<FloatType>(resultElementType)) {
    sizeInByte = floatType.getWidth() / 8;
  } else {
    llvm_unreachable("Unhandled element type of tensor");
  }

  for (int i = ptrOffsetInfo.getRank() - 1; i >= 0; i--) {
    if (!ptrOffsetInfo.isStructured(i))
      break;
    sizeInByte *= resultShape[i];
  }

  // Force scalarize if memory is not aligned
  if (sizeInByte % 32 != 0)
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "UnStructured Flag check:\n";
    os << "ptrOffsetInfo.isStructured: " << ptrOffsetInfo.isStructured() << "\n";
    os << "compileOn91095Flag: " << compileOn91095Flag << "\n";
    os << "forceSimtTemplateFlag: " << forceSimtTemplateFlag << "\n";
  });
  
  // Fast path on A5: rewrite tt.load/store to tt.indirect_load/store directly.
  if (compileOn91095Flag && forceSimtTemplateFlag && !ptrOffsetInfo.isStructured()) {
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
      assert(isa<triton::PointerType>(srcPtr.getType()) && "src must be ptr type");
      Value mask = op.getMask();
      Value other = op.getOther();
      auto resultType = op.getType();
      auto indirect = rewriter.create<triton::ascend::IndirectLoadOp>(
          loc, resultType, srcPtr, ptrOffset, mask, other);
      rewriter.replaceOp(op, indirect.getResult());
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Rewriting tt.load to tt.indirect_load\n";
        os << indirect << "\n";
      });
      return success();
    } else if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      assert(isa<triton::PointerType>(srcPtr.getType()) && "src must be ptr type");
      Value value = op.getValue();
      Value mask = op.getMask();
      auto indirect = rewriter.create<triton::ascend::IndirectStoreOp>(
          loc, srcPtr, ptrOffset, value, mask);
      rewriter.eraseOp(op);
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Rewriting tt.store to tt.indirect_store\n";
        os << indirect << "\n";
      });
      return success();
    }
  }

  Value iterArg = nullptr;

  // Only load case
  if (isLoadLike) {
    iterArg =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, resultElementType);
  }
  Value newOpResult = nullptr;

  auto insertPoint = rewriter.saveInsertionPoint();

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<int64_t> extractedShape;

  for (size_t i = 0; i < resultShape.size(); i++) {
    auto size = resultShape[i];
    auto structured = ptrOffsetInfo.getStructuredRef()[i];
    // handle indirect dimension
    strides.push_back(rewriter.getIndexAttr(1));
    Value sizeVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(size));
    if (structured) {
      offsets.push_back(rewriter.getIndexAttr(0));
      sizes.push_back(rewriter.getIndexAttr(size));
      extractedShape.push_back(size);
    } else {
      scf::ForOp forOp;
      if (auto mtptOp =
              srcPtr.template getDefiningOp<triton::MakeTensorPtrOp>()) {
        auto tptShape = mtptOp.getShape()[i];
        if (tptShape.getType() != rewriter.getIndexType()) {
          tptShape = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), tptShape);
        }
        sizeVal = rewriter.create<arith::MinSIOp>(loc, sizeVal, tptShape);
      } else if (mstate) {
        sizeVal = getValueOrCreateConstantIndexOp(rewriter, loc, mstate->dims[i]);
      }
      if (isLoadLike) {
        forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, sizeVal, oneIdx,
                                            ValueRange({iterArg}));
        if (!newOpResult) {
          newOpResult = forOp->getResult(0);
        } else {
          rewriter.create<scf::YieldOp>(loc, forOp->getResult(0));
        }
        iterArg = forOp.getRegionIterArg(0);
      } else {
        forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, sizeVal, oneIdx);
      }
      sizes.push_back(rewriter.getIndexAttr(1));
      offsets.push_back(forOp.getInductionVar());
      extractedShape.push_back(1);
      forOp->setAttr("ExtractedLoadOrStore",
                     UnitAttr::get(rewriter.getContext()));
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
  }

  bool fullyUnstructured = ptrOffsetInfo.isUnstructured();
  auto extractedType = RankedTensorType::get(extractedShape, resultElementType);

  Value extractedOffset;
  if (fullyUnstructured) {
    if (auto mtptOp =
            srcPtr.template getDefiningOp<triton::MakeTensorPtrOp>()) {
      auto I64Type = rewriter.getIntegerType(64);
      srcPtr = mtptOp.getBase();
      extractedOffset = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      for (auto [indVar, offset, stride] :
           llvm::zip_equal(offsets, ptrOffsetInfo.getOffsets(), mtptOp.getStrides())) {
        Value inductionVar = rewriter.create<arith::IndexCastOp>(
            loc, I64Type, cast<Value>(indVar));
        Value tptOffset = rewriter.create<arith::ExtSIOp>(loc, I64Type, offset);
        Value tptStride = rewriter.create<arith::ExtSIOp>(loc, I64Type, stride);
        tptOffset = rewriter.create<arith::MulIOp>(loc, tptStride, tptOffset);
        tptStride =
            rewriter.create<arith::MulIOp>(loc, tptStride, inductionVar);
        extractedOffset =
            rewriter.create<arith::AddIOp>(loc, extractedOffset, tptOffset);
        extractedOffset =
            rewriter.create<arith::AddIOp>(loc, extractedOffset, tptStride);
      }
    } else {
      extractedOffset = createExtractOp(loc, ptrOffset, rewriter, offsets);
    }
  } else {
    extractedOffset =
        createExtractOp(loc, ptrOffset, rewriter, offsets, sizes, strides);
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracted offset\n";
    os << extractedOffset << "\n";
  });

  assert(isa<triton::PointerType>(srcPtr.getType()) && "src must be ptr type");
  if (!fullyUnstructured) {
    srcPtr = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(extractedShape, srcPtr.getType()), srcPtr);
  }
  Value ptrToAccess = rewriter.create<triton::AddPtrOp>(
      loc, srcPtr.getType(), srcPtr, extractedOffset);

  MemAccOpTy accessedOp;
  if (fullyUnstructured) {
    accessedOp = createMemAccOp(op, ptrToAccess, loc, rewriter, offsets);
  } else {
    accessedOp =
        createMemAccOp(op, ptrToAccess, loc, rewriter, offsets, sizes, strides);
  }

  accessedOp->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));

  if (isLoadLike) {
    assert(iterArg && "Load case must have iterArg in for loop");

    Value value = accessedOp->getResult(0);
    Value result;
    if (!isa<RankedTensorType>(value.getType()) &&
        (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp> ||
        std::is_same_v<MemAccOpTy, triton::AtomicCASOp>)) {
      value =	
          rewriter.create<triton::SplatOp>(loc, extractedType, value);
    }
    if (!isa<RankedTensorType>(value.getType())) {
      SmallVector<Value> indices;
      for (auto idx : offsets) {
        if (auto val = dyn_cast<Value>(idx)) {
          indices.push_back(val);
        } else {
          auto idxVal = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIndexAttr(*getConstantIntValue(idx)));
          indices.push_back(idxVal);
        }
      }
      result = rewriter.create<tensor::InsertOp>(
          loc, value, iterArg, indices);
    } else {
      result = rewriter.create<tensor::InsertSliceOp>(
          loc, value, iterArg, offsets, sizes, strides);
    }
    rewriter.create<scf::YieldOp>(loc, result)
          ->setAttr(ConverterUtils::discreteAttrName,
                    UnitAttr::get(rewriter.getContext()));
    rewriter.restoreInsertionPoint(insertPoint);
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
      if (op.getMask() && op.getOther()) {
        rewriter
            .replaceOpWithNewOp<arith::SelectOp>(op, op.getMask(), newOpResult,
                                                 op.getOther())
            ->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));
      } else {
        rewriter.replaceOp(op, newOpResult);
      }
    } else {
      rewriter.replaceOp(op, newOpResult);
    }
  } else {
    if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
      if (fullyUnstructured && accessedOp.getMask()) {
        auto mask = createExtractOp(loc, accessedOp.getMask(), rewriter,
                                    SmallVector<OpFoldResult>(ptrOffsetInfo.getRank(), rewriter.getIndexAttr(0)));
        rewriter.create<scf::IfOp>(loc, mask, [&](OpBuilder &b, Location loc) {
          b.create<triton::AtomicRMWOp>(
               loc, accessedOp.getType(), accessedOp.getAtomicRmwOp(),
               accessedOp.getPtr(), accessedOp.getVal(), nullptr,
               accessedOp.getSem(), accessedOp.getScope())
              ->setAttr(ConverterUtils::discreteAttrName,
                        UnitAttr::get(rewriter.getContext()));
          b.create<scf::YieldOp>(loc);
        });
        rewriter.eraseOp(accessedOp);
      }
    }
    rewriter.eraseOp(op);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After conversion\n"
       << ptrToAccess.getDefiningOp()
              ->template getParentOfType<triton::FuncOp>()
       << "\n";
  });
  return success();
}

void replaceOperands(MutableArrayRef<OpOperand> oprs, RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  for (auto it = oprs.begin(); it != oprs.end(); ++it) {
    auto &opr = *it;
    auto operand = opr.get();
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      parse(operand, operand.getLoc(), rewriter, offsetMap);
      opr.set(offsetMap.at(operand).getOffset());
    } else if (auto ptrType = dyn_cast<triton::PointerType>(operand.getType())) {
      parse(operand, operand.getLoc(), rewriter, offsetMap);
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        for (auto offset: offsetMap.at(operand).getOffsets()) {
          it->set(offset);
          ++it;
        }
        --it;
      } else {
        opr.set(offsetMap.at(operand).getOffset());
      }
    }
  }
}

void replaceArgs(ValueRange args, RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  for (auto it = args.begin(); it != args.end(); ++it) {
    auto arg = *it;
    if (auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
        tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
      RewriterBase::InsertionGuard guard(rewriter);
      if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
        rewriter.setInsertionPointToStart(blockArg.getOwner());
      } else {
        rewriter.setInsertionPointAfterValue(arg);
      }
      auto tempVar = rewriter
                         .create<UnrealizedConversionCastOp>(
                             arg.getLoc(), arg.getType(), ValueRange({}))
                         ->getResult(0);
      parse(arg, arg.getLoc(), rewriter, offsetMap);
      auto src = offsetMap.at(arg).getPtr();
      rewriter.replaceAllUsesWith(arg, tempVar);
      arg.setType(RankedTensorType::get(tensorType.getShape(),
                                        rewriter.getIntegerType(64)));
      src = rewriter.create<triton::SplatOp>(arg.getLoc(), tempVar.getType(),
                                             src);
      rewriter.replaceOpWithNewOp<triton::AddPtrOp>(
          tempVar.getDefiningOp(), tempVar.getType(), src, arg);
    } else if (auto ptrType = dyn_cast<triton::PointerType>(arg.getType())) {
      RewriterBase::InsertionGuard guard(rewriter);
      if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
        rewriter.setInsertionPointToStart(blockArg.getOwner());
      } else {
        rewriter.setInsertionPointAfterValue(arg);
      }
      auto tempVar = rewriter
                        .create<UnrealizedConversionCastOp>(
                            arg.getLoc(), arg.getType(), ValueRange({}))
                        ->getResult(0);
      parse(arg, arg.getLoc(), rewriter, offsetMap);
      rewriter.replaceAllUsesWith(arg, tempVar);
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        auto srcOp = offsetMap.at(arg).getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        arg.setType(rewriter.getIntegerType(32));
        SmallVector<Value> newOffsets;
        for (auto offset: offsetMap.at(arg).getOffsets()) {
          newOffsets.push_back(*it);
          ++it;
        }
        --it;
        rewriter.replaceOpWithNewOp<triton::MakeTensorPtrOp>(
          tempVar.getDefiningOp(), tempVar.getType(), srcOp.getBase(), srcOp.getShape(), srcOp.getStrides(), newOffsets, srcOp.getOrder());  
      } else {
        auto src = offsetMap.at(arg).getPtr();
        arg.setType(rewriter.getIntegerType(64));
        rewriter.replaceOpWithNewOp<triton::AddPtrOp>(
            tempVar.getDefiningOp(), tempVar.getType(), src, arg);
      }
    }
  }
}

void convertTensorPtrPre(LoopLikeOpInterface op, RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    replaceArgs(whileOp.getBeforeArguments(), rewriter, offsetMap);
    replaceOperands(whileOp.getInitsMutable(), rewriter, offsetMap);
    replaceArgs(whileOp.getAfterArguments(), rewriter, offsetMap);
    replaceArgs(whileOp->getResults(), rewriter, offsetMap);
    replaceOperands(whileOp.getConditionOp().getArgsMutable(), rewriter,
                    offsetMap);
  } else {
    replaceArgs(op.getRegionIterArgs(), rewriter, offsetMap);
    replaceOperands(op.getInitsMutable(), rewriter, offsetMap);
  }
}

void convertTensorPtrPost(LoopLikeOpInterface op, RewriterBase &rewriter,
                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    replaceOperands(whileOp.getYieldOp()->getOpOperands(), rewriter, offsetMap);
  } else {
    replaceArgs(op->getResults(), rewriter, offsetMap);
    replaceOperands(*op.getYieldedValuesMutable(), rewriter, offsetMap);
  }
}

int getPtrTensorRank(Type type) {
  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    if (auto tensorType =
            dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      return tensorType.getRank();
    }
  }
  return 0;
}

SmallVector<Value> constructOperands(ValueRange operands, Value tempVar, IRMapping mapping) {
  SmallVector<Value> newOperands;
  for (auto opr : operands) {
    opr = mapping.lookupOrDefault(opr);
    newOperands.push_back(opr);
    auto numAppend = getPtrTensorRank(opr.getType()) - 1;
    if (numAppend > 0)
      newOperands.append(numAppend, tempVar);
  }
  return newOperands;
}

SmallVector<Type> constructTypes(TypeRange types) {
  SmallVector<Type> newTypes;
  for (auto type : types) {
    newTypes.push_back(type);
    if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      if (auto tensorType =
              dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
        if (tensorType.getRank() > 0)
          newTypes.append(tensorType.getRank() - 1,
                          IntegerType::get(type.getContext(), 32));
      }
    }
  }
  return newTypes;
}

void replacePtrLoopArguments(Operation *rootOp,
                             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  std::function<WalkResult(LoopLikeOpInterface)> convertTensorPtr =
      [&](LoopLikeOpInterface op) {
        IRRewriter rewriter(op.getContext());
        IRMapping mapping;
        LoopLikeOpInterface newOp;
        rewriter.setInsertionPointAfter(op);
        Value tempVar =
            rewriter
                .create<UnrealizedConversionCastOp>(
                    op.getLoc(), rewriter.getI32Type(), ValueRange({}))
                ->getResult(0);
        if (auto forOp = dyn_cast<scf::ForOp>(op.getOperation())) {
          newOp = rewriter.create<scf::ForOp>(
              forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
              forOp.getStep(), constructOperands(forOp.getInitArgs(), tempVar, mapping),
              [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
                mapping.map(forOp.getInductionVar(), iv);
                auto newArgIter = args.begin();
                for (auto oldArg : forOp.getRegionIterArgs()) {
                  mapping.map(oldArg, *newArgIter);
                  std::advance(newArgIter,
                               std::max(getPtrTensorRank(oldArg.getType()), 1));
                }
                for (auto &bodyOp : forOp.getBody()->without_terminator()) {
                  b.clone(bodyOp, mapping);
                }
                auto yieldOp =
                    cast<scf::YieldOp>(forOp.getBody()->getTerminator());
                b.create<scf::YieldOp>(
                    yieldOp.getLoc(),
                    constructOperands(yieldOp.getOperands(), tempVar, mapping));
              });
          newOp->setAttrs(op->getAttrs());
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
          newOp = rewriter.create<scf::WhileOp>(
              whileOp.getLoc(), constructTypes(whileOp->getResultTypes()),
              constructOperands(whileOp.getInits(), tempVar, mapping),
              [&](OpBuilder &b, Location loc, ValueRange args) {
                auto newArgIter = args.begin();
                for (auto oldArg : whileOp.getBeforeArguments()) {
                  mapping.map(oldArg, *newArgIter);
                  std::advance(newArgIter,
                               std::max(getPtrTensorRank(oldArg.getType()), 1));
                }
                for (auto &bodyOp :
                     whileOp.getBeforeBody()->without_terminator()) {
                  b.clone(bodyOp, mapping);
                }
                auto conditionOp = whileOp.getConditionOp();
                b.create<scf::ConditionOp>(
                    conditionOp.getLoc(), mapping.lookup(conditionOp.getCondition()),
                    constructOperands(conditionOp.getArgs(), tempVar, mapping));
              },
              [&](OpBuilder &b, Location loc, ValueRange args) {
                auto newArgIter = args.begin();
                for (auto oldArg : whileOp.getAfterArguments()) {
                  mapping.map(oldArg, *newArgIter);
                  std::advance(newArgIter,
                               std::max(getPtrTensorRank(oldArg.getType()), 1));
                }
                for (auto &bodyOp : whileOp.getAfterBody()->without_terminator()) {
                  b.clone(bodyOp, mapping);
                }
                auto yieldOp = whileOp.getYieldOp();
                b.create<scf::YieldOp>(
                    yieldOp.getLoc(),
                    constructOperands(yieldOp.getOperands(), tempVar, mapping));
              });
        } else {
          llvm_unreachable("Unsupported loop op");
        }
        auto resIter = newOp->result_begin();
        for (auto res: op->getResults()) {
          rewriter.replaceAllUsesWith(res, *resIter);
          std::advance(resIter,
                               std::max(getPtrTensorRank(res.getType()), 1));
        }
        rewriter.eraseOp(op);
        op = newOp;
        convertTensorPtrPre(op, rewriter, offsetMap);
        for (auto *region : op.getLoopRegions())
          region->walk<WalkOrder::PreOrder>(convertTensorPtr);
        convertTensorPtrPost(op, rewriter, offsetMap);
        return WalkResult::skip();
      };

  rootOp->walk<WalkOrder::PreOrder>(convertTensorPtr);
}

void TritonToUnstructurePass::runPreparse(LoopLikeOpInterface op) {
  IRRewriter rewriter(&getContext());
  auto loc = op.getLoc();

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Pre-parsing " << op->getName() << "\n" << op << "\n";
  });

  Block::BlockArgListType args;
  ValueRange yields;
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    args = whileOp.getBeforeArguments();
    yields = whileOp.getYieldOp().getOperands();
  } else {
    args = op.getRegionIterArgs();
    yields = op.getYieldedValues();
  }

  for (auto[arg, yield] : llvm::zip_equal(args, yields)) {
    if (auto tensorType = dyn_cast<RankedTensorType>(yield.getType())) {
      parse(yield, loc, rewriter, offsetMapForLoopArgs);
      offsetMap[arg] = offsetMapForLoopArgs.at(yield);
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Pre-parsing result of\n" << arg << "\nis ";
        for (auto structured : offsetMap[arg].getStructuredRef())
          os << structured;
        os << '\n';
      });
    }
  }
}

static bool isFromTensorArg(Value v, llvm::SmallDenseMap<Value, bool> &fromTensorArg) {
  if (fromTensorArg.contains(v))
    return fromTensorArg.at(v);
  auto *defOp = v.getDefiningOp();
  if (!defOp) {
    fromTensorArg[v] = isa<RankedTensorType>(v.getType());
    return isa<RankedTensorType>(v.getType());
  }
    for (auto opr : defOp->getOperands()) {
      if (isFromTensorArg(opr, fromTensorArg)) {
        fromTensorArg[v] = true;
        return true;
      }
    }
  fromTensorArg[v] = false;
  return false;
}

template <typename MemAccOpTy, typename>
void TritonToUnstructurePass::runParse(MemAccOpTy op) {
  IRRewriter rewriter(&getContext());
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing " << op->getName() << "\n" << op << "\n";
  });
  parse(op.getPtr(), op.getLoc(), rewriter, offsetMap);
  isFromTensorArg(op.getPtr(), fromTensorArg);
}

TritonToUnstructurePass::TritonToUnstructurePass(
    const TritonToUnstructureOptions &options)
    : TritonToUnstructureBase(options) {}

void TritonToUnstructurePass::runOnOperation() {
  compileOn91095Flag = this->compileOn91095;
  forceSimtTemplateFlag = this->forceSimtTemplate;

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "TritonToUnstructurePass started with options:\n";
    os << "  compileOn91095: " << compileOn91095Flag << "\n";
    os << "  forceSimtTemplate: " << forceSimtTemplateFlag << "\n";
  });

  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  replacePtrLoopArguments(moduleOp, offsetMapForLoopArgs);
  offsetMapForLoopArgs.clear();
  moduleOp->walk([this](LoopLikeOpInterface op) { runPreparse(op); });
  moduleOp->walk([this](Operation *op) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      runParse(loadOp);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      runParse(storeOp);
    } else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)) {
      runParse(atomicRMWOp);
    } else if (auto atomicCASOp = dyn_cast<triton::AtomicCASOp>(op)) {
      runParse(atomicCASOp);
    }
  });

  RewritePatternSet patterns(ctx);

  patterns.add<UnstructuredMemAccessConverter<triton::LoadOp>,
               UnstructuredMemAccessConverter<triton::StoreOp>,
               UnstructuredMemAccessConverter<triton::AtomicRMWOp>,
               UnstructuredMemAccessConverter<triton::AtomicCASOp>>(
      ctx, forceScalarizeMode, offsetMap, fromTensorArg);

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing done\n";
  });

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Patterns");
    signalPassFailure();
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }
}

void TritonToUnstructurePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect,
                  affine::AffineDialect, scf::SCFDialect, tensor::TensorDialect,
                  bufferization::BufferizationDialect, memref::MemRefDialect,
                  triton::TritonDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToUnstructurePass(
    const TritonToUnstructureOptions &options) {
  return std::make_unique<TritonToUnstructurePass>(options);
}
