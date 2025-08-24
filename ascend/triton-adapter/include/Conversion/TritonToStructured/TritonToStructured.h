#ifndef TRITON_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H
#define TRITON_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "Conversion/ConversionCommon.h"


namespace mlir {
namespace triton {


const unsigned INT_BIT_WIDTH = 32;
const unsigned SET_INIT_SIZE = 16;

std::unique_ptr<OperationPass<ModuleOp>> createTritonToStructuredPass();
void populateTritonToStructuredCanonicalizationPatterns(RewritePatternSet & canonicalizerPatterns);


// tempate class's impl must in header file
template <typename OpTy>
class LoadStoreCanonicalizer : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value ptrVal = op.getPtr();
    Type ptrTy = ptrVal.getType();
    auto ptrDefOp = ptrVal.getDefiningOp();
    if (isa<BlockArgument>(ptrVal))
      return failure();

    if (!isTensorPointerType(ptrTy) &&
        !isa_and_nonnull<triton::AddPtrOp>(ptrDefOp)) {
      if (isa<triton::BitcastOp>(ptrDefOp)) {
        auto castOp = cast<triton::BitcastOp>(ptrDefOp);
        auto castSrc = castOp.getSrc();
        if (!isa<BlockArgument>(castSrc)) {
          auto castSrcDefOp = castSrc.getDefiningOp();
          if (isa<triton::AddPtrOp>(castSrcDefOp)) {
            return rewriter.notifyMatchFailure(
                op, "BitcastCanonicalizer handles addptr->bitcast->load!");
          }
        }
      }

      Type zeroTy = getI32SameShape(ptrTy);
      Value zeroVal =
          createScalarOrSplatConstant(rewriter, op.getLoc(), zeroTy, 0);
      Value addptrVal = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTy,
                                                          ptrVal, zeroVal);
      rewriter.modifyOpInPlace(
          op, [&]() { op->replaceUsesOfWith(ptrVal, addptrVal); });
      return success();
    }
    return failure();
  }
};

class ScalarStoreCanonicalizer : public OpRewritePattern<triton::StoreOp> {
public:
  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

class ScalarAtomicRMWCanonicalizer
    : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

class ScalarAtomicCASCanonicalizer
    : public OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern<triton::AtomicCASOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override;
};


class AtomicMaxMinCanonicalizer : public OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

/*
 * Rewrite tt.make_tensor_ptr with non-contiguous order to
 * tt.make_tensor_ptr + tt.load + tt.trans.
 */
class MakeTensorPtrCanonicalizer
    : public OpRewritePattern<triton::MakeTensorPtrOp> {
public:
  using OpRewritePattern<triton::MakeTensorPtrOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override;
};


void setBlockArgumentAttr(BlockArgument blockArg, triton::FuncOp func, TensorKind tensorKind) ;


template <typename OpTy>
void addTensorKindToArguments(OpTy op, triton::FuncOp func, TensorKind tensorKind)
{
    Value ptr = op.getPtr();
    if (!ptr)
        return;

    Value cur = ptr;
    llvm::SmallPtrSet<Value, SET_INIT_SIZE> visited;
    // 回溯 def-use 链，找到起源 BlockArgument
    while (visited.insert(cur).second) {
        // 如果是 BlockArgument，则尝试设置属性
        if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
            if (blockArg.getOwner() == &func.getBody().front()) {
                auto type = blockArg.getType();
                // 检查是否是 triton::PointerType
                if (!isa<triton::PointerType>(type))
                    break;
                setBlockArgumentAttr(blockArg, func, tensorKind);
                break;
            }
        }

        Operation *defOp = cur.getDefiningOp();
        if (!defOp)
            break;
        cur = defOp->getOperand(0);
    }
}


struct Descriptor {
    Value base;
    SmallVector<Value> shape;
    SmallVector<Value> strides;
};

bool hasATensorDescriptorType(mlir::TypeRange types);

class DescriptorLoadConverter : public OpConversionPattern<triton::DescriptorLoadOp> {
public:
    using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

class DescriptorStoreConverter : public OpConversionPattern<triton::DescriptorStoreOp> {
public:
    using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOSTRUCTURED_TRITONTOSTRUCTURED_H
