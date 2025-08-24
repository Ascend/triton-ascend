//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#define DEBUG_TYPE "triton-arith-to-linalg"

#include "Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "Conversion/TritonArithToLinalg/TritonArithToLinalgPass.h"
#include "Conversion/ConversionCommon.h"
#include "Analysis/UseAnalysis.h"

// namespace mlir {
// namespace triton {
// #define GEN_PASS_DEF_TRITONARITHTOLINALG
// #include "Conversion/TritonArithToLinalg/Passes.h.inc"
// } // namespace triton
// } // namespace mlir

extern int nd2nzFlag;

using namespace mlir;
using namespace triton;

namespace {

TritonArithTypeConverter::TritonArithTypeConverter()
{
  addConversion([](Type type) { return type; });

  addConversion([](triton::PointerType ptrType) {
    return MemRefType::get({ShapedType::kDynamic}, ptrType.getPointeeType());
  });

  addConversion([](TensorType tensorType) -> Type {
    auto elemType = tensorType.getElementType();
    if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
      elemType = ptrType.getPointeeType();
    }
    return MemRefType::get(tensorType.getShape(), elemType);
  });
}

void TritonArithToLinalgPass::addProgramInfo(triton::FuncOp func,
                                        bool globalKernel)
                                        {
  OpBuilder b(func);

  auto origFuncType = func.getFunctionType();
  auto origInputTypes = origFuncType.getInputs();
  SmallVector<Type> newInputTypes(origInputTypes);
  newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

  auto newFuncType =
      b.getFunctionType(newInputTypes, origFuncType.getResults());

  func.setFunctionType(newFuncType);

  // Add empty attributes for each new argument if needed
  if (func.getAllArgAttrs()) {
    SmallVector<DictionaryAttr> newArgAttrs;
    func.getAllArgAttrs(newArgAttrs);
    newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
    func.setAllArgAttrs(newArgAttrs);
  }

  // Add the corresponding arguments to function body
  for (unsigned i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
    func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
  }

  if (globalKernel) {
    func->setAttr(globalKernelAttr, b.getStringAttr(""));
  } else {
    func->setAttr(globalKernelAttr, b.getStringAttr("local"));
  }

}

// 处理嵌套的if/else
void TritonArithToLinalgPass::transformNestedIfElse(Operation &op, OpBuilder &builder)
{
    auto nestedBranch = dyn_cast<cf::CondBranchOp>(&op);
    SmallVector<Operation*> nestedTrueOps;
    SmallVector<Operation*> nestedFalseOps;
    bool nestedTrueHasReturn = false;
    bool nestedFalseHasReturn = false;

    for (Operation &op : nestedBranch.getTrueDest()->without_terminator()) {
        if (dyn_cast<cf::CondBranchOp>(&op)) {
            transformNestedIfElse(op, builder);
        }
        nestedTrueOps.push_back(&op);
        if (isa<func::ReturnOp>(op)) {
            nestedTrueHasReturn = true;
        }
    }
    for (Operation &op : nestedBranch.getFalseDest()->without_terminator()) {
        if (dyn_cast<cf::CondBranchOp>(&op)) {
            transformNestedIfElse(op, builder);
        }
        nestedFalseOps.push_back(&op);
        if (isa<func::ReturnOp>(op)) {
            nestedFalseHasReturn = true;
        }
    }
    builder.setInsertionPoint(nestedBranch);
    auto nestedIfOp = builder.create<scf::IfOp>(
        nestedBranch.getLoc(),
        nestedBranch.getCondition(),
        [&](OpBuilder &thenBuilder, Location loc) {
            for (Operation *op : nestedTrueOps) {
                op->moveBefore(thenBuilder.getInsertionBlock(), thenBuilder.getInsertionPoint());
            }
            if (!nestedTrueHasReturn) {
                thenBuilder.create<scf::YieldOp>(loc);
            }
        },
        [&](OpBuilder &elseBuilder, Location loc) {
            for (Operation *op : nestedFalseOps) {
                op->moveBefore(elseBuilder.getInsertionBlock(), elseBuilder.getInsertionPoint());
            }
            if (!nestedTrueHasReturn) {
                elseBuilder.create<scf::YieldOp>(loc);
            }
        }
    );
    nestedBranch.erase();
    nestedBranch.getTrueDest()->erase();
    nestedBranch.getFalseDest()->erase();
}

void TritonArithToLinalgPass::convertTTFunc(triton::FuncOp func, const bool existDot)
{
  OpBuilder builder(func);

  auto name = func.getName();
  auto type = func.getFunctionType();

  SmallVector<DictionaryAttr> argAttrs, resAttrs;
  func.getAllArgAttrs(argAttrs);
  func.getAllResultAttrs(resAttrs);

  // bit-casted tt.ptr的特殊处理
  SmallVector<Type> inputTypes{type.getInputs()};
  SmallVector<Type> retTypes{type.getResults()};
  if (func.getSymVisibility() == "public" && !func.isDeclaration()) {
    for (size_t i = 0; i < func.getNumArguments(); ++i) {
      auto arg = func.getArgument(i);
      // Special method for i1 arg
      // original
      if (!isa<BaseMemRefType>(arg.getType()) ||
          dyn_cast<BaseMemRefType>(arg.getType()).getElementTypeBitWidth() !=
              1) {
        continue;
      }
      // if (!isa<triton::PointerType>(arg.getType()) ||
      //     dyn_cast<triton::PointerType>(arg.getType()).getPointeeType().getIntOrFloatBitWidth() !=
      //         1) {
      //   continue;
      // }

      SmallVector<Operation *> argVaildUser{arg.getUsers()};
      llvm::erase_if(argVaildUser, [](Operation *op) -> bool {
        return isOpTriviallyDead(op);
      });

      if (!argVaildUser.empty()) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << arg << " has users:\n";
          int cnt = 0;
          for (auto it : argVaildUser) {
            os << "users[" << cnt++ << "] = " << *it;
          }
        });
        if (llvm::all_of(argVaildUser, [](Operation *userOp) {
              // return isa<triton::BitcastOp>(userOp);
              return isa<UnrealizedConversionCastOp>(userOp);
            })) {
          // >>>>> origin
          auto castOp = cast<UnrealizedConversionCastOp>(*argVaildUser.begin());
          if (castOp.getInputs().size() == 1 &&
              castOp.getOutputs().size() == 1) {
            arg.setType(castOp.getOutputs()[0].getType());
            inputTypes[i] = arg.getType();
          }
          // <<<<< changed begin
          // auto castOp = cast<triton::BitcastOp>(*argVaildUser.begin());
          // arg.setType(castOp.getResult().getType());
          // inputTypes[i] = arg.getType();
          // >>>>> changed end
        } else {
          func->emitError(Twine("Unsupported use of func arg at index ") +
                          Twine(i));
        }
      } else {
        // Process unused bool ptr type specially, which guarantees bool pointer
        // argument's type is realistic and don't mislead backend compiler.
        // realistic memory layout of bool pointer is 8 bit width
        auto memType = dyn_cast<BaseMemRefType>(arg.getType())
                           .cloneWith(std::nullopt, builder.getI8Type());
        arg.setType(memType);
        inputTypes[i] = arg.getType();
      }
    }
  }
  auto castType = FunctionType::get(func.getContext(), inputTypes, retTypes);

  auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, castType);
  funcFunc.setAllArgAttrs(argAttrs);
  funcFunc.setAllResultAttrs(resAttrs);
  auto kernelAttr = func->getAttr(globalKernelAttr);
  if (kernelAttr) {
    funcFunc->setAttr(globalKernelAttr, kernelAttr);
  }
  std::string kernelMixMode = "aiv";
  if (existDot) {
    // mix also works for pure cube kernel by using the same MAGIC_ELF keyword
    kernelMixMode = "mix";
  }
  // Set mix_mode in the func attrs so that the backend could know
  // the mix_mode by parse the func attrs.
  // The backend needs to know the mix_mode because the host wrapper
  // needs to set the devbin.magic. Check npu_utils.cpp.
  funcFunc->setAttr(kernelMixModeName, builder.getStringAttr(kernelMixMode));

  auto &funcFuncBody = funcFunc.getBody();
  auto &funcBody = func.getBody();

  IRMapping map;
  funcBody.cloneInto(&funcFuncBody, map);

  for (Block &block : funcFuncBody.getBlocks()) {
    auto term = block.getTerminator();
    if (auto condBranch = dyn_cast<cf::CondBranchOp>(term)) {
        SmallVector<Operation*> trueOps;
        SmallVector<Operation*> falseOps;
        bool trueHasReturn = false;
        bool falseHasReturn = false;
        for (Operation &op : condBranch.getTrueDest()->without_terminator()) {
            if (dyn_cast<cf::CondBranchOp>(&op)) {
                transformNestedIfElse(op, builder);
            }
            trueOps.push_back(&op);
            if (isa<func::ReturnOp>(op)) {
                trueHasReturn = true;
            }
        }
        for (Operation &op : condBranch.getFalseDest()->without_terminator()) {
            if (dyn_cast<cf::CondBranchOp>(&op)) {
                transformNestedIfElse(op, builder);
            }
            falseOps.push_back(&op);
            if (isa<func::ReturnOp>(op)) {
                falseHasReturn = true;
            }
        }
        builder.setInsertionPoint(condBranch);
        auto ifOp = builder.create<scf::IfOp> (
            condBranch.getLoc(),
            condBranch.getCondition(),
            [&](OpBuilder &thenBuilder, Location loc) {
                for (Operation *op : trueOps) {
                    op->moveBefore(thenBuilder.getInsertionBlock(), thenBuilder.getInsertionPoint());
                }
                if (!trueHasReturn) {
                    thenBuilder.create<scf::YieldOp>(loc);
                }
            },
            [&](OpBuilder &elseBuilder, Location loc) {
                for (Operation *op : falseOps) {
                    op->moveBefore(elseBuilder.getInsertionBlock(), elseBuilder.getInsertionPoint());
                }
                if (!falseHasReturn) {
                    elseBuilder.create<scf::YieldOp>(loc);
                }
            }
        );
        if (!trueHasReturn && !falseHasReturn) {
            Block *afterBlock = condBranch->getBlock();
            if (!afterBlock->empty()) {
                builder.setInsertionPointToEnd(afterBlock);
                builder.create<func::ReturnOp>(condBranch.getLoc());
            }
        }
        condBranch.erase();
        condBranch.getTrueDest()->erase();
        condBranch.getFalseDest()->erase();
      } else {
        builder.setInsertionPoint(term);
        builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
        term->erase();
      }
  }
  func.erase();
}

void TritonArithToLinalgPass::getDependentDialects(DialectRegistry &registry) const
{
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                triton::TritonDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect>();
}


void TritonArithToLinalgPass::addDynamicLegal(
    ConversionTarget &target, TritonArithTypeConverter &tritonTypeConverter) {

  target.addLegalDialect<
      func::FuncDialect, arith::ArithDialect, math::MathDialect,
      linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
      cf::ControlFlowDialect, tensor::TensorDialect,
      bufferization::BufferizationDialect, ttx::TritonTilingExtDialect, memref::MemRefDialect,
      tts::TritonStructuredDialect>();


  // add legal dialect on condition
  target.addLegalOp<ModuleOp>();

  // 根据条件判断需要转换的OP
  target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
      [](mlir::Operation *op) {
        if (op->use_empty()) {
          return false;
        } else {
          return true;
        }
      });


  //kaixin fixme
  // target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
  //   return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
  // });
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<triton::FuncOp, triton::ReturnOp>();

  target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
    auto res = op.getResult();
    if (!isa<RankedTensorType>(res.getType())) {
      return true;
    }

    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
      if (!denseAttr.isSplat() ||
          !isa<FloatType, IntegerType>(denseAttr.getElementType())) {
        return true;
      }
      if (res.hasOneUse() && isa<tensor::ReshapeOp>(*res.user_begin())) {
        return true;
      }
      return false;
    }
    return true;
  });

  // target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>([](Operation *op) {
  //   return llvm::all_of(op->getOperandTypes(), [](Type t) {
  //     if (isa<triton::PointerType>(t)) {
  //       return false;
  //     }
  //     if (auto shapedType = dyn_cast<ShapedType>(t)) {
  //       return shapedType.getElementType().isIntOrFloat();
  //     }
  //     assert(t.isIntOrIndexOrFloat());
  //     return true;
  //   });
  // });

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
      [this](Operation *op) {
        if (op->hasAttr("MetaUse")) {
          return false;
        }

        if (isa<arith::ConstantOp>(op)) {
          return true;
        }

        bool operateOnTensors =
            llvm::all_of(op->getOperandTypes(),
                         [](Type type) { return isa<RankedTensorType>(type); });

        //return this->namedOps || !operateOnTensors;
        return true ;
      });

    target.addIllegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();

    target.addDynamicallyLegalOp<triton::AddPtrOp>([](triton::AddPtrOp op) {
       return !isa<ShapedType>(op.getResult().getType());
     });


    target.addLegalOp<triton::AssertOp>();
}


void  TritonArithToLinalgPass::runOnOperation() {
  auto moduleOp = getOperation();

    // Check if the kernel contains tl.dot. Without tl.dot,
    // the kernel would be pure AIV kernel.
    bool existDot = false;
    moduleOp.walk([&](triton::DotOp dotOp) {
        existDot = true;
        return WalkResult::interrupt();
    });

    RewritePatternSet canonicalizerPatterns(&getContext());


  // 1.标准化 LoadStore ScalarStoreCanonicalizer
    populateTritonArithToLinalgCanonicalizationPatterns(canonicalizerPatterns);
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(canonicalizerPatterns)))) {
      signalPassFailure();
    }

    // 2.使用分析
    // moduleOp.walk([this](triton::FuncOp op) {
    //   if (failed(runUseAnalysis(op))) {
    //     signalPassFailure();
    //   }
    // });

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonArithTypeConverter tritonTypeConverter{};
    // 3.标注合法方言
    this->addDynamicLegal(target, tritonTypeConverter);

  // 5.对非法Op注册Converter
    populateTritonArithToLinalgConversionPatterns(tritonTypeConverter, patterns,
                                                 LAUNCH_GRID_RANK);

    // 6.遍历kernel中的function，修改program id、number of programs参数
    for (auto func : getOperation().getOps<triton::FuncOp>()) {
      addProgramInfo(func);
    }

    // 7.做Op转换
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
    // 8.函数头尾转换
    // >>>>>  move to structured_to_memref pass
    // Convert tt.func and tt.return into func's counterparts
    if (ttToFuncFunc) {
      moduleOp.walk(
          [&](triton::FuncOp func) { this->convertTTFunc(func, existDot); });
    }
    // 9.清除无效代码，简化代码。
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }


  // Force to add an argument at the beginning of function arguments, which
  // represents stub arg for workspace. Default type is memref<?xi8>
  for (auto func : getOperation().getOps<func::FuncOp>()) {
    if (!func->hasAttr("global_kernel"))
      continue;

    auto context = func.getContext();
    constexpr int64_t syncBlockLockArgIdx = 0;
    NamedAttribute syncBlockLockArgAttr(StringAttr::get(context, "syncBlockLock"),
                                    UnitAttr::get(context));
    MemRefType syncBlockLockArgType =
        MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                        IntegerType::get(context, 8));
    func.insertArgument(syncBlockLockArgIdx, // argIndex
                        syncBlockLockArgType, // argType
                        nullptr, func->getLoc()); // dicAttr
    func->setAttr("SyncBlockLockArgIdx",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64), 0));  // 64: 64位整型

    constexpr int64_t workspaceArgIdx = 1;
    MemRefType workspaceArgType =
        MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                        IntegerType::get(context, 8));
    NamedAttribute workspaceArgAttr(StringAttr::get(context, "workspace"),
                                    UnitAttr::get(context));

    func.insertArgument(/*argIndex*/ workspaceArgIdx,
                        /*argType*/ workspaceArgType,
                        /*dicAttr*/ nullptr, func->getLoc());
    func->setAttr("WorkspaceArgIdx",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64), 1));  // 64: 64位整型
  }
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonArithToLinalgPass() {
  return std::make_unique<TritonArithToLinalgPass>();
}
