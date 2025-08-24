#include "TritonToLinalg/TritonToLinalgPass.h"
#include "TritonToLinalg/ArgMinMaxConverter.h"
#include "TritonToLinalg/FunctionConverter.h"
#include "TritonToLinalg/LoadStoreConverter.h"
#include "TritonToLinalg/TritonOpConverter.h"
#include "TritonToLinalg/DescriptorConverter.h"
#include "TritonToLinalg/UseAnalysis.h"
#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
#include <optional>

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

int nd2nzFlag = 0;

TritonTypeConverter::TritonTypeConverter() {
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

void TritonToLinalgPass::addProgramInfo(triton::FuncOp func,
                                        bool globalKernel) {
  OpBuilder b(func);

  auto origFuncType = func.getFunctionType();
  auto origInputTypes = origFuncType.getInputs();
  SmallVector<Type> newInputTypes(origInputTypes);
  newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

  auto newFuncType =
      b.getFunctionType(newInputTypes, origFuncType.getResults());

  func.setFunctionType(newFuncType);

  // 如果需要，给参数新增属性
  if (func.getAllArgAttrs()) {
    SmallVector<DictionaryAttr> newArgAttrs;
    func.getAllArgAttrs(newArgAttrs);
    newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
    func.setAllArgAttrs(newArgAttrs);
  }

  // 添加对应参数到函数体中
  for (unsigned i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
    func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
  }

  if (globalKernel) {
    func->setAttr(globalKernelAttr, b.getStringAttr(""));
  } else {
    func->setAttr(globalKernelAttr, b.getStringAttr("local"));
  }
}

static void setBlockArgumentAttr(BlockArgument blockArg, triton::FuncOp func, TensorKind tensorKind)
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

template <typename OpTy>
void TritonToLinalgPass::addTensorKindToArguments(OpTy op, triton::FuncOp func, TensorKind tensorKind)
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

void TritonToLinalgPass::convertTTFunc(triton::FuncOp func,
                                       const bool existDot) {
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
      if (!isa<BaseMemRefType>(arg.getType()) ||
          dyn_cast<BaseMemRefType>(arg.getType()).getElementTypeBitWidth() !=
              1) {
        continue;
      }

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
              return isa<UnrealizedConversionCastOp>(userOp);
            })) {
          auto castOp = cast<UnrealizedConversionCastOp>(*argVaildUser.begin());
          if (castOp.getInputs().size() == 1 &&
              castOp.getOutputs().size() == 1) {
            arg.setType(castOp.getOutputs()[0].getType());
            inputTypes[i] = arg.getType();
          }
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

// 处理嵌套的if/else
void TritonToLinalgPass::transformNestedIfElse(Operation &op, OpBuilder &builder) {
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

void TritonToLinalgPass::addDynamicLegal(
    ConversionTarget &target, TritonTypeConverter &tritonTypeConverter) {
  target.addLegalDialect<
      func::FuncDialect, arith::ArithDialect, math::MathDialect,
      linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
      cf::ControlFlowDialect, tensor::TensorDialect, LLVM::LLVMDialect,
      bufferization::BufferizationDialect, memref::MemRefDialect,
      annotation::AnnotationDialect, hivm::HIVMDialect>();

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

  target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
    return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
  });

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

  target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>([](Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type t) {
      if (isa<triton::PointerType>(t)) {
        return false;
      }
      if (auto shapedType = dyn_cast<ShapedType>(t)) {
        return shapedType.getElementType().isIntOrFloat();
      }
      assert(t.isIntOrIndexOrFloat());
      return true;
    });
  });

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

        return this->namedOps || !operateOnTensors;
      });
}

void TritonToLinalgPass::populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns)
{
    patterns.add<LoadStoreConverter::LoadStoreCanonicalizer<triton::LoadOp>,
                 LoadStoreConverter::LoadStoreCanonicalizer<triton::StoreOp>,
                 LoadStoreConverter::LoadStoreCanonicalizer<triton::AtomicRMWOp>,
                 LoadStoreConverter::LoadStoreCanonicalizer<triton::AtomicCASOp>>(patterns.getContext());
    patterns.add<TTOpConverters::SelectCanonicalizer>(patterns.getContext());
    patterns.add<TTOpConverters::BitcastCanonicalizer>(patterns.getContext());
    patterns.add<LoadStoreConverter::ScalarStoreCanonicalizer>(patterns.getContext());
    patterns.add<LoadStoreConverter::ScalarAtomicRMWCanonicalizer>(patterns.getContext());
    patterns.add<LoadStoreConverter::ScalarAtomicCASCanonicalizer>(patterns.getContext());
    patterns.add<LoadStoreConverter::AtomicMaxMinCanonicalizer>(patterns.getContext());
    patterns.add<
        TTOpConverters::ScalarMathCanonicalizer<math::AbsFOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AcosOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AcoshOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AsinOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AsinhOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AtanOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::Atan2Op>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AtanhOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::CeilOp>, TTOpConverters::ScalarMathCanonicalizer<math::CosOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::CoshOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::ErfOp>, TTOpConverters::ScalarMathCanonicalizer<math::ExpOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::Exp2Op>,
        // TTOpConverters::ScalarMathCanonicalizer<math::ExpM1Op>,
        TTOpConverters::ScalarMathCanonicalizer<math::FloorOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::FmaOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::LogOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::Log10Op>,
        // TTOpConverters::ScalarMathCanonicalizer<math::Log1pOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::Log2Op>,
        // TTOpConverters::ScalarMathCanonicalizer<math::PowFOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::RoundOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::RsqrtOp>, TTOpConverters::ScalarMathCanonicalizer<math::SinOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::SinhOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::SqrtOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::TanOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::TanhOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::TruncOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::AddFOp>, TTOpConverters::ScalarMathCanonicalizer<arith::SubFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MulFOp>, TTOpConverters::ScalarMathCanonicalizer<arith::DivFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::NegFOp>, TTOpConverters::ScalarMathCanonicalizer<arith::RemFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MaxNumFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MaximumFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MinNumFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MinimumFOp>
        // By test, the following ops do not need canonicalization.
        // TTOpConverters::ScalarMathCanonicalizer<arith::CmpFOp>
        // TTOpConverters::ScalarMathCanonicalizer<arith::ExtFOp>
        // TTOpConverters::ScalarMathCanonicalizer<arith::TruncFOp>
        >(patterns.getContext());
    patterns.add<TTOpConverters::MakeTensorPtrCanonicalizer>(patterns.getContext());
    patterns.add<TTOpConverters::ReduceSingleCanonicalizer>(patterns.getContext());
}

void TritonToLinalgPass::populateTritonToLinalgConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank) {
    nd2nzFlag = this->enableNd2nzOnVector;
  populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
      patterns, typeConverter);

  patterns.add<triton_adapter::MetaUseEraser>(patterns.getContext());
  patterns.add<LoadStoreConverter::StoreConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AddPtrConverter>(patterns.getContext());
  patterns.add<FunctionConverter::GetProgramIDConverter>(patterns.getContext());
  patterns.add<FunctionConverter::GetNumProgramsConverter>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::LoadConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicRMWConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicCASConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MakeRangeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SplatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ClampFConverter>(patterns.getContext());
  patterns.add<TTOpConverters::PreciseDivConverter>(patterns.getContext());
  // reduce converters
  patterns.add<TTOpConverters::ArgMinConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ArgMaxConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ReduceConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ScanConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ReshapeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ExpandDimsConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BroadcastConverter>(patterns.getContext());

  patterns.add<TTOpConverters::DenseConstantConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ExternElementwiseClOpConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::TritonMulhiuiConverter>(patterns.getContext());
  patterns.add<TTOpConverters::TritonPreciseSqrtConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::MakeTensorPtrConverter>(patterns.getContext());
  patterns.add<TTOpConverters::AdvanceConverter>(patterns.getContext());
  patterns.add<TTOpConverters::TransposeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SplitConverter>(patterns.getContext());
  patterns.add<TTOpConverters::JoinConverter>(patterns.getContext());
  patterns.add<TTOpConverters::CatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BitcastConverter>(patterns.getContext());
  patterns.add<TTOpConverters::LoopConverter>(patterns.getContext());
  patterns.add<TTOpConverters::YieldConverter>(patterns.getContext());
  patterns.add<TTOpConverters::GatherConverter>(patterns.getContext());

  patterns.add<TTOpConverters::DeviceAssertConverter>(patterns.getContext());
  patterns.add<TTOpConverters::DevicePrintConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MatmulConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SortOpConverter>(patterns.getContext());
  patterns.add<TTOpConverters::DotScaledConverter>(patterns.getContext());

  if (!this->namedOps) {
    linalg::populateElementwiseToLinalgConversionPatterns(patterns);
  }
}

void TritonToLinalgPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                  tensor::TensorDialect, bufferization::BufferizationDialect,
                  memref::MemRefDialect>();
}

LogicalResult TritonToLinalgPass::processDescriptorOperations(ModuleOp moduleOp)
{
    // --- ConversionTarget: 动态合法性判断 ---
    mlir::ConversionTarget target(getContext());

    // Dialect-level dynamic legality: 这些 dialect 在操作的 operand/result 中不包含 TensorDescType 时为合法
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect, mlir::scf::SCFDialect, triton::TritonDialect>(
        [](mlir::Operation *op) {
            return !DescriptorConverter::hasATensorDescriptorType(op->getOperandTypes()) &&
                   !DescriptorConverter::hasATensorDescriptorType(op->getResultTypes());
        });
    // 函数签名合法性：triton::FuncOp 的输入/输出 不含 TensorDescType 则合法
    target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
        return !DescriptorConverter::hasATensorDescriptorType(funcOp.getFunctionType().getInputs()) &&
               !DescriptorConverter::hasATensorDescriptorType(funcOp.getFunctionType().getResults());
    });
    target.addLegalOp<triton::MakeTensorDescOp>();
    target.addIllegalOp<triton::DescriptorLoadOp, triton::DescriptorStoreOp>();

    // --- Patterns ---
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<DescriptorConverter::DescriptorLoadConverter>(patterns.getContext());
    patterns.add<DescriptorConverter::DescriptorStoreConverter>(patterns.getContext());

    mlir::ConversionConfig config;
    config.buildMaterializations = true;
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns), config))) {
        moduleOp->emitError("failed to convert tensor descriptor operations");
        return failure();
    }

    return success();
}

void TritonToLinalgPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Check if the kernel contains tl.dot. Without tl.dot,
  // the kernel would be pure AIV kernel.
  bool existDot = false;
  moduleOp.walk([&](triton::DotOp dotOp) {
    existDot = true;
    return WalkResult::interrupt();
  });
    moduleOp.walk([&](triton::DotScaledOp dotScaledOp) {
        existDot = true;
        return WalkResult::interrupt();
    });

  RewritePatternSet canonicalizerPatterns(&getContext());

  // 0. 首先执行 tensor descriptor 操作转换
  if (failed(processDescriptorOperations(moduleOp))) {
    signalPassFailure();
  }

  // 遍历所有的triton::FuncOp，添加tensor_kind属性
  moduleOp.walk([&](triton::FuncOp func) {
    func.walk([&](triton::LoadOp loadOp) {
      addTensorKindToArguments(loadOp, func, TensorKind::INPUT);
    });
    func.walk([&](triton::StoreOp storeOp) {
      addTensorKindToArguments(storeOp, func, TensorKind::OUTPUT);
    });
    func.walk([&](triton::AtomicRMWOp atomicOp) {
      addTensorKindToArguments(atomicOp, func, TensorKind::INPUT_OUTPUT);
    });
    func.walk([&](triton::AtomicCASOp atomicOp) {
      addTensorKindToArguments(atomicOp, func, TensorKind::INPUT_OUTPUT);
    });
  });

  // 1.标准化 LoadStore ScalarStoreCanonicalizer
  this->populateTritonToLinalgCanonicalizationPatterns(canonicalizerPatterns);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(canonicalizerPatterns)))) {
    moduleOp->emitError("failed to apply Canonicalizer Patterns");
    signalPassFailure();
  }

  // 2.使用分析
  moduleOp.walk([this](triton::FuncOp op) {
    if (failed(triton_adapter::runUseAnalysis(op))) {
      signalPassFailure();
    }
  });

  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  TritonTypeConverter tritonTypeConverter{};

  // 3.标注合法方言
  this->addDynamicLegal(target, tritonTypeConverter);

  // 5.对非法Op注册Converter
  this->populateTritonToLinalgConversionPatterns(tritonTypeConverter, patterns,
                                                 LAUNCH_GRID_RANK);

  // 6.遍历kernel中的function，修改program id、number of programs参数
  for (auto func : getOperation().getOps<triton::FuncOp>()) {
    addProgramInfo(func, globalKernel);
  }

  // 7.做Op转换
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Convertion Patterns");
    signalPassFailure();
  }

  // 8.函数头尾转换
  moduleOp.walk(
      [&](triton::FuncOp func) { this->convertTTFunc(func, existDot); });

  // 9.清除无效代码，简化代码。
  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }

  // Try interleave optimization
  llvm::DenseMap<BlockArgument, SmallVector<Operation *>> interleaveCandidate;
  llvm::DenseMap<BlockArgument, SmallVector<Operation *>>
      interleaveCandidateWithMask;
  moduleOp.walk([&](bufferization::MaterializeInDestinationOp materializeOp) {
    if (auto reinterpretCastOp =
            materializeOp.getDest()
                .getDefiningOp<memref::ReinterpretCastOp>()) {
      if (llvm::isa<BlockArgument>(reinterpretCastOp.getSource()) &&
          reinterpretCastOp.getStaticStrides().back() == 2) {
        interleaveCandidate[llvm::cast<BlockArgument>(
                                reinterpretCastOp.getSource())]
            .push_back(materializeOp);
      }
    }

    // Difference is that converted op chain of store with mask has
    // `memref::SubViewOp`
    if (auto subviewOp =
            materializeOp.getDest().getDefiningOp<memref::SubViewOp>()) {
      if (!llvm::isa<tensor::ExtractSliceOp>(
              materializeOp.getSource().getDefiningOp()))
        return WalkResult::advance();

      if (auto reinterpretCastOp =
              subviewOp.getSource()
                  .getDefiningOp<memref::ReinterpretCastOp>()) {
        if (llvm::isa<BlockArgument>(reinterpretCastOp.getSource()) &&
            reinterpretCastOp.getStaticStrides().back() == 2) {
          interleaveCandidateWithMask[llvm::cast<BlockArgument>(
                                          reinterpretCastOp.getSource())]
              .push_back(materializeOp);
        }
      }
    }

    return WalkResult::advance();
  });

  for (auto [blockArg, materializeVec] : interleaveCandidate) {
    // Just enable optimization where exists double materializeOp with same
    // block argument destination.
    if (materializeVec.size() != 2)
      continue;
    auto result = InterleaveStatusOptimization(materializeVec);
  }

  for (auto [blockArg, materializeVec] : interleaveCandidateWithMask) {
    if (materializeVec.size() != 2)
      continue;
    auto result = InterleaveStatusWithMaskOptimization(materializeVec);
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

  // Fix the Location info
  moduleOp.walk([&](Operation *op) {
    auto loc = op->getLoc();
    if (isa<UnknownLoc>(loc)) {
      llvm::SmallPtrSet<Operation *, 16> stopOps;
      traverseForwardUpdateUserChainIf(
          op,
          /*conditionFn*/
          [](Operation *curOp) { return false; },
          /*stopFn*/
          [](Operation *curOp) { return !isa<UnknownLoc>(curOp->getLoc()); },
          /*actionFn*/
          nullptr, stopOps);
      if (stopOps.empty()) {
        op->emitWarning() << *op << " and its users all have no location!";
      } else {
        Operation *goodOp = *stopOps.begin();
        op->setLoc(goodOp->getLoc());
      }
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
