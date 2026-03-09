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

#include "TritonAffinityOpt/Passes.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "Utils/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

// #include "mlir/Pass/Pass.h"
// #include "mlir/Pass/PassManager.h"

// #include "mlir/Transforms/Canonicalizer.h"
// #include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DAGSSBUFFER
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {
struct DAGSSBufferPass
    : public mlir::triton::impl::DAGSSBufferBase<
          DAGSSBufferPass> {
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
};
} // namespace

void ControlSsbufV2(ModuleOp module) {
    mlir::OpBuilder builder(module.getContext());
    // 用于记录已经处理过的scope.scope操作
    llvm::DenseSet<mlir::Operation*> processedScopes;

    auto aiCAttr = hivm::TCoreTypeAttr::get(
            builder.getContext(),
            hivm::TCoreType::CUBE);
    int cubeControlIndex = 13;
    int vectorControlIndex = 12;

    llvm::DenseSet<mlir::Operation*> processedScopes2;
    module->walk([&](SyncBlockWaitOp op) {
        // 向上查找父scope.scope操作
        mlir::Operation* parentOp = op->getParentOp();
        mlir::Operation* scopeOp = nullptr;
        mlir::Operation* forOp = nullptr;
        
        // 向上遍历查找scope.scope操作
        while (parentOp) {
            if (dyn_cast<scope::ScopeOp>(parentOp)) {
                scopeOp = parentOp;
                break;
            }
            parentOp = parentOp->getParentOp();
        }
        parentOp = op->getParentOp();
        while (parentOp) {
            if (dyn_cast<scf::ForOp>(parentOp)) {
                forOp = parentOp;
                break;
            }
            parentOp = parentOp->getParentOp();
        }
        // 如果没有找到scope.scope操作，则跳过
        if (!scopeOp) {
            return;
        }
        if (!forOp) {
            return;
        }

        // 如果该scope已经处理过，则跳过
        if (processedScopes2.count(forOp) > 0) return;
        
        // 标记该scope为已处理
        processedScopes2.insert(forOp);

    });
    bool firstSet = true;
    bool firstWait = true;
    for (auto forOp : processedScopes2) {
        mlir::Operation* parentOp = forOp->getParentOp();
        mlir::Operation* scopeOp = nullptr;
        
        // 向上遍历查找scope.scope操作
        while (parentOp) {
            if (dyn_cast<scope::ScopeOp>(parentOp)) {
                scopeOp = parentOp;
                break;
            }
            parentOp = parentOp->getParentOp();
        }
        bool isAIC = false;
        // 1. 先检查操作是否有这个属性
        
        if (scopeOp->hasAttr("hivm.tcore_type")) {
            auto attr = scopeOp->getAttr("hivm.tcore_type");
            if (attr == aiCAttr) {
            isAIC = true;
            }
        }

        if (isAIC) {
            // 在for循环的开头插入代码
            builder.setInsertionPoint(scopeOp);
            // %ssb_ready_addr = llvm.mlir.constant(0 : i64) : i64
            auto i64Type = builder.getIntegerType(64);
            auto i32Type = builder.getIntegerType(32);

            builder.setInsertionPointToStart(&forOp->getRegion(0).front());
            // %ssb_ready_addr = llvm.mlir.constant(0 : i64) : i64
            // add sync_block_wait
            auto coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
            auto setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            auto waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            auto flagId = builder.getIntegerAttr(builder.getI64Type(), vectorControlIndex);
            builder.create<SyncBlockWaitOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);

            // 在循环末尾（yield之前）插入代码
            auto &loopBody = forOp->getRegion(0).front();
            // 找到循环体的terminator（应该是yield操作）
            auto *terminator = loopBody.getTerminator();
            builder.setInsertionPoint(terminator);
            
            // add sync_block_set
            coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
            setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            flagId = builder.getIntegerAttr(builder.getI64Type(), cubeControlIndex);
            builder.create<SyncBlockSetOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);

            if (firstWait) {
                auto &scopeBlock = scopeOp->getRegion(0).front();
                auto *scope_terminator = scopeBlock.getTerminator();
                builder.setInsertionPoint(scope_terminator);            
                // add sync_block_wait
                coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
                setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
                waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
                flagId = builder.getIntegerAttr(builder.getI64Type(), vectorControlIndex);
                builder.create<SyncBlockWaitOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);
                firstWait = false;
            }
        }
        else {
            // 1. 在scopeop的开头插入代码
            // 假设scopeOp是一个具有区域的操作，我们获取其第一个块
            if (firstSet) {
                auto &scopeBlock = scopeOp->getRegion(0).front();
                builder.setInsertionPointToStart(&scopeBlock);
                
                // add sync_block_wait
                auto coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
                auto setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
                auto waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
                auto flagId = builder.getIntegerAttr(builder.getI64Type(), vectorControlIndex);
                builder.create<SyncBlockSetOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);
                firstSet = false;
            }

            auto i64Type = builder.getIntegerType(64);
            auto i32Type = builder.getIntegerType(32);
            
            // 创建需要的常量
            auto c32ConstAttr = mlir::IntegerAttr::get(i64Type, 32);
            auto c32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i64Type, c32ConstAttr);
            
            auto c0i64ConstAttr = mlir::IntegerAttr::get(i64Type, 0);
            auto c0i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i64Type, c0i64ConstAttr);
            
            auto c0i32ConstAttr = mlir::IntegerAttr::get(i32Type, 0);
            auto c0i32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i32Type, c0i32ConstAttr);
            
            auto c1i32ConstAttr = mlir::IntegerAttr::get(i32Type, 1);
            auto c1i32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i32Type, c1i32ConstAttr);
            
            // %sub_id = hivm.hir.get_sub_block_idx -> i64
            // 这里假设有一个getSubBlockIdxOp操作
            auto subIdOp = builder.create<GetSubBlockIdxOp>(
                scopeOp->getLoc(), i64Type);
            
            // %ssb_addr_offset = arith.muli %sub_id, %c32_i64 : i64
            auto ssbAddrOffsetOp = builder.create<mlir::arith::MulIOp>(
                scopeOp->getLoc(),
                subIdOp.getResult(),
                c32ConstOp.getResult());
            
            // %ssb_addr = arith.addi %ssb_addr_offset, %c32_i64 : i64
            auto ssbAddrOp = builder.create<mlir::arith::AddIOp>(
                scopeOp->getLoc(),
                ssbAddrOffsetOp.getResult(),
                c32ConstOp.getResult());
            
            // %vec_id = arith.cmpi eq, %sub_id, %c0_i64 : i64
            auto vecIdOp = builder.create<mlir::arith::CmpIOp>(
                scopeOp->getLoc(),
                mlir::arith::CmpIPredicate::eq,
                subIdOp.getResult(),
                c0i64ConstOp.getResult());
            
            // 2. 在parentop的开头插入代码
            builder.setInsertionPointToStart(&forOp->getRegion(0).front());
            
            // add sync_block_wait
            auto coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
            auto setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            auto waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            auto flagId = builder.getIntegerAttr(builder.getI64Type(), cubeControlIndex);
            builder.create<SyncBlockWaitOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);
            
            // 在循环末尾（yield之前）插入代码
            auto &loopBody = forOp->getRegion(0).front();
            // 找到循环体的terminator（应该是yield操作）
            auto *terminator = loopBody.getTerminator();
            builder.setInsertionPoint(terminator);
            
            // add sync_block_wait
            coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
            setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            flagId = builder.getIntegerAttr(builder.getI64Type(), vectorControlIndex);
            builder.create<SyncBlockSetOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);
        }
    }
    
    auto i64Type = builder.getIntegerType(64);
    auto i32Type = builder.getIntegerType(32);
    auto initPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext(), 11);
    SmallVector<scope::ScopeOp> scopeOps;
    module->walk([&](mlir::Operation* op) {
        // 检查是否为目标操作
        if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
            scopeOps.push_back(scopeOp);
        }
    });
    for (auto scopeOp : scopeOps) {
        builder.setInsertionPoint(scopeOp);
        auto c0i64ConstAttr = mlir::IntegerAttr::get(i64Type, 0);
        auto c0i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
            scopeOp->getLoc(), i64Type, c0i64ConstAttr);
        auto c32i64ConstAttr = mlir::IntegerAttr::get(i64Type, 32);
        auto c32i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
            scopeOp->getLoc(), i64Type, c32i64ConstAttr);
        auto c64i64ConstAttr = mlir::IntegerAttr::get(i64Type, 64);
        auto c64i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
            scopeOp->getLoc(), i64Type, c64i64ConstAttr);
        auto c96i64ConstAttr = mlir::IntegerAttr::get(i64Type, 96);
        auto c96i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
            scopeOp->getLoc(), i64Type, c96i64ConstAttr);
        auto c0i32ConstAttr = mlir::IntegerAttr::get(i32Type, 0);
        auto c0i32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
            scopeOp->getLoc(), i32Type, c0i32ConstAttr);
        
        auto c0initInttoptrOp = builder.create<mlir::LLVM::IntToPtrOp>(
            scopeOp->getLoc(), initPtrType, c0i64ConstOp.getResult());
        auto c32initInttoptrOp = builder.create<mlir::LLVM::IntToPtrOp>(
            scopeOp->getLoc(), initPtrType, c32i64ConstOp.getResult());
        auto c64initInttoptrOp = builder.create<mlir::LLVM::IntToPtrOp>(
            scopeOp->getLoc(), initPtrType, c64i64ConstOp.getResult());
        auto c96initInttoptrOp = builder.create<mlir::LLVM::IntToPtrOp>(
            scopeOp->getLoc(), initPtrType, c96i64ConstOp.getResult());
        
        builder.create<LLVM::StoreOp>(
                scopeOp->getLoc(),
                c0i32ConstOp,
                c0initInttoptrOp
            );
        builder.create<LLVM::StoreOp>(
                scopeOp->getLoc(),
                c0i32ConstOp,
                c32initInttoptrOp
            );
        builder.create<LLVM::StoreOp>(
                scopeOp->getLoc(),
                c0i32ConstOp,
                c64initInttoptrOp
            );
        builder.create<LLVM::StoreOp>(
                scopeOp->getLoc(),
                c0i32ConstOp,
                c96initInttoptrOp
            );
        break;
    }
}

scf::ForOp transformLoop(scf::ForOp forOp, OpBuilder &builder) {
    
    // 1. 获取原始循环的信息
    Value originalLowerBound = forOp.getLowerBound();
    Value originalUpperBound = forOp.getUpperBound();
    Value originalStep = forOp.getStep();
    SmallVector<Value> iterArgs;
    for (auto arg : forOp.getInitArgs()) {
        iterArgs.push_back(arg);
    }
    auto yields = forOp.getBody()->getTerminator();
    
    // 2. 检查循环体中是否有特定操作
    int hasTargetOps = 0;
    forOp.walk([&](Operation* op) {
        if (isa<scf::IfOp>(op)) {
            hasTargetOps++;
        }
    });
    // 3. 如果存在目标操作，在迭代参数中添加计数器
    Value counterInit = nullptr;
    mlir::Operation* parentOp = forOp->getParentOp();
    mlir::Operation* scopeOp = nullptr;
    // 向上遍历查找scope.scope操作
    while (parentOp) {
        if (dyn_cast<scope::ScopeOp>(parentOp)) {
            scopeOp = parentOp;
            break;
        }
        parentOp = parentOp->getParentOp();
    }

    builder.setInsertionPoint(scopeOp);
    for (int i = 0; i < hasTargetOps; i++) {
        Location loc = forOp.getLoc();
        auto i32Type = builder.getI32Type();
        counterInit = builder.create<arith::ConstantIntOp>(loc, 0, i32Type);
        
        // 添加到迭代参数列表
        iterArgs.push_back(counterInit);
    }
    // 2. 创建新的上界：originalUpperBound * 2
    Location loc = forOp.getLoc();
    Type ubType = originalStep.getType();
    builder.setInsertionPoint(forOp);
    
    // 创建类型匹配的常数2
    Value two;
    if (ubType.isIndex()) {
        two = builder.create<arith::ConstantIndexOp>(loc, 2);
    } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
        // 对于整数类型，创建相应类型的常数2
        two = builder.create<arith::ConstantIntOp>(loc, 2, intType);
    } else {
        // 其他类型可能需要特殊处理
        llvm::errs() << "Warning: Unexpected type for upper bound: " << ubType << "\n";
        // 尝试创建索引类型的2然后转换
        auto indexTwo = builder.create<arith::ConstantIndexOp>(loc, 2);
        two = builder.create<arith::IndexCastOp>(loc, ubType, indexTwo);
    }
    
    // 创建乘法：originalUpperBound * 2
    auto newUpperBound = builder.create<arith::MulIOp>(
        forOp.getLoc(), 
        originalUpperBound, 
        two);
    
    // 3. 创建新的for循环
    auto newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(),
        originalLowerBound,
        newUpperBound,
        originalStep,
        iterArgs);
    
    // 4. 设置IR映射表，将旧循环的变量映射到新循环
    IRMapping mapper;
    
    // 映射迭代变量
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());
    
    // 映射迭代参数
    for (auto [oldArg, newArg] : 
         llvm::zip(forOp.getRegionIterArgs(), 
                  newForOp.getRegionIterArgs())) {
        mapper.map(oldArg, newArg);
    }
    
    SmallVector<Value> newCounterArgs;
    for (int i = forOp.getRegionIterArgs().size(); i < newForOp.getRegionIterArgs().size(); i++) {
        newCounterArgs.push_back(newForOp.getRegionIterArgs()[i]);
    }
    // 5. 克隆循环体内容到新循环
    auto &newLoopBody = *newForOp.getBody();
    builder.setInsertionPointToStart(&newLoopBody);
    
    for (auto &op : forOp.getBody()->without_terminator()) {
        builder.clone(op, mapper);
    }
    
    // 6. 克隆yield操作
    if (auto yieldOp = dyn_cast<scf::YieldOp>(yields)) {
        SmallVector<Value> newYieldOperands;
        for (auto operand : yieldOp.getOperands()) {
            newYieldOperands.push_back(mapper.lookupOrDefault(operand));
        }
        if (hasTargetOps) {
            for (auto currentCounter : newCounterArgs) {                
                // 将更新后的计数器添加到yield操作数中
                newYieldOperands.push_back(currentCounter);
            }
        }
        builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    }
    
    // 7. 替换原循环的结果
    if (hasTargetOps) {
        // 新循环有额外的计数器结果，但原循环没有对应结果
        // 我们可以选择只替换原循环对应的结果，或者忽略计数器结果
        unsigned numOriginalResults = forOp.getNumResults();
        SmallVector<Value> originalResults;
        for (unsigned i = 0; i < numOriginalResults; i++) {
            originalResults.push_back(newForOp.getResult(i));
        }
        forOp.replaceAllUsesWith(originalResults);
    } else {
        forOp.replaceAllUsesWith(newForOp.getResults());
    }
    
    // 8. 删除原循环
    forOp.erase();
    return newForOp;
    
}

SmallVector<bool> getWaitType(std::string CoreType, scf::ForOp forOp) {
    SmallVector<bool> waitTypes;
    auto scalarWaitPipe = PipeAttr::get(forOp.getContext(), hivm::PIPE::PIPE_S);
    auto cubeWaitPipe = PipeAttr::get(forOp.getContext(), hivm::PIPE::PIPE_FIX);
    auto vectorWaitPipe = PipeAttr::get(forOp.getContext(), hivm::PIPE::PIPE_MTE3);
    forOp.walk([&](Operation* op) {
        if (auto waitOp = dyn_cast<SyncBlockWaitOp>(op)) {
            if (isa<scf::IfOp>(op->getParentOp())) {
                auto waitPipe = waitOp.getPipe();
                if ((waitPipe == cubeWaitPipe && CoreType == "cube") || (waitPipe == vectorWaitPipe && CoreType == "vector")) {
                    waitTypes.push_back(0);
                }
                else if (waitPipe != scalarWaitPipe) {
                    waitTypes.push_back(1);
                }
            }
        }
    });
    return waitTypes;
}

DenseMap<int, int> getCounterOffset(scf::ForOp forOp) {
    int i = 0;
    DenseMap<int, int> bufferMap;
    auto scalarWaitPipe = PipeAttr::get(forOp.getContext(), hivm::PIPE::PIPE_S);
    forOp.walk([&](Operation* op) {
        bufferMap[i] = 0;
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            ifOp.walk([&](Operation* op) {
                if (auto waitOp = dyn_cast<SyncBlockWaitOp>(op)) {
                    if (isa<scf::IfOp>(op->getParentOp())) {
                        auto waitPipe = waitOp.getPipe();
                        if ((waitPipe != scalarWaitPipe)) {
                            bufferMap[i]++;
                        }
                    }
                }
            });
            i ++;
        }
    });
    return bufferMap;
}

SmallVector<Value> addBufValLoop(scf::ForOp forOp, int numBuffer, int subLoop, OpBuilder &builder) {
    auto aiCAttr = hivm::TCoreTypeAttr::get(
            builder.getContext(),
            hivm::TCoreType::CUBE);
    bool isAIC = false;
    // 向上查找父scope.scope操作
    mlir::Operation* parentOp = forOp->getParentOp();
    mlir::Operation* scopeOp = nullptr;
    // 向上遍历查找scope.scope操作
    while (parentOp) {
        if (dyn_cast<scope::ScopeOp>(parentOp)) {
            scopeOp = parentOp;
            break;
        }
        parentOp = parentOp->getParentOp();
    }
    if (scopeOp->hasAttr("hivm.tcore_type")) {
        auto attr = scopeOp->getAttr("hivm.tcore_type");
        if (attr == aiCAttr) {
            isAIC = true;
        }
    }
    auto bufferMap = getCounterOffset(forOp);
    SmallVector<Value> buf_vals;
    SmallVector<Value> if_conditions;
    builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());
    auto i32Type = builder.getIntegerType(32);

    // 1. 提取并处理end值
    Value endValue = forOp.getUpperBound();
    // 2. 提取并处理step值
    Value stepValue = forOp.getStep();
    builder.setInsertionPoint(forOp);
    Value subLoopValue_pre = builder.create<arith::DivSIOp>(
        forOp.getLoc(), // 位置信息
        i32Type,        // 结果类型
        endValue,       // 被除数（end）
        stepValue       // 除数（step）
    );
    auto subLoopConstAttr = mlir::IntegerAttr::get(i32Type, 2);
    auto subLoopValue_2 = builder.create<mlir::LLVM::ConstantOp>(
        scopeOp->getLoc(), i32Type, subLoopConstAttr);
    Value subLoopValue = builder.create<arith::DivSIOp>(
        forOp.getLoc(), // 位置信息
        i32Type,        // 结果类型
        subLoopValue_pre,       // 被除数（end）
        subLoopValue_2       // 除数（step）
    );

    SmallVector<bool> WaitType;
    SmallVector<Value> bufferPtrs;
    if (isAIC) {
        builder.setInsertionPointToStart(&forOp->getRegion(0).front());
        // 创建常量32和64
        Value c0 = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), 0, 32  // 值32，64位
        );
        Value c32 = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), 32, 64  // 值32，64位
        );
        Value c64 = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), 64, 64  // 值64，64位
        );
        // 创建inttoptr操作
        Value ssb_vec0_ptr = builder.create<LLVM::IntToPtrOp>(
            forOp.getLoc(),
            LLVM::LLVMPointerType::get(builder.getContext(), 11),  // 地址空间11
            c32
        );
        Value ssb_vec1_ptr = builder.create<LLVM::IntToPtrOp>(
            forOp.getLoc(),
            LLVM::LLVMPointerType::get(builder.getContext(), 11),  // 地址空间11
            c64
        );
        bufferPtrs.push_back(ssb_vec0_ptr);
        bufferPtrs.push_back(ssb_vec1_ptr);
        // 创建load操作
        Value status_vec0 = builder.create<LLVM::LoadOp>(
            forOp.getLoc(), builder.getI32Type(), ssb_vec0_ptr
        );
        
        Value status_vec1 = builder.create<LLVM::LoadOp>(
            forOp.getLoc(), builder.getI32Type(), ssb_vec1_ptr
        );

        WaitType = getWaitType("cube", forOp);

        for (auto i = 0; i < WaitType.size(); i++) {
            auto i32ConstAttr = mlir::IntegerAttr::get(i32Type, 1 << i);
            auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i32Type, i32ConstAttr);
            Value bufi_vec0_val = builder.create<arith::AndIOp>(
                forOp.getLoc(), status_vec0, buf_constant_set
            );
            Value bufi_vec1_val = builder.create<arith::AndIOp>(
                forOp.getLoc(), status_vec1, buf_constant_set
            );
            Value flag_bufi_vec0;
            Value flag_bufi_vec1;
            // 创建比较操作
            if (WaitType[i] == 0) {
                flag_bufi_vec0 = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_vec0_val, c0
                );
                flag_bufi_vec1 = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_vec1_val, c0
                );
            }
            else {
                flag_bufi_vec0 = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_vec0_val, buf_constant_set
                );
                flag_bufi_vec1 = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_vec1_val, buf_constant_set
                );
            }
            // 创建最终的and操作
            Value bufi_val = builder.create<arith::AndIOp>(
                forOp.getLoc(), flag_bufi_vec0, flag_bufi_vec1
            );
            buf_vals.push_back(bufi_val);
        }
        
    } else {
        builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());
        Value c0 = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), 0, 32  // 值32，64位
        );
        auto i64Type = builder.getIntegerType(64);
        // %sub_id = hivm.hir.get_sub_block_idx -> i64
        // 这里假设有一个getSubBlockIdxOp操作
        auto subIdOp = builder.create<GetSubBlockIdxOp>(
            scopeOp->getLoc(), i64Type);
        auto i64ConstAttr = mlir::IntegerAttr::get(i64Type, 32);
        auto cst_offset = builder.create<mlir::LLVM::ConstantOp>(
            scopeOp->getLoc(), i64Type, i64ConstAttr);
        auto ssb_addr_offset = builder.create<arith::MulIOp>(
            scopeOp->getLoc(), subIdOp, cst_offset);
        auto ssb_addr = builder.create<arith::AddIOp>(
            scopeOp->getLoc(), ssb_addr_offset, cst_offset);
        builder.setInsertionPointToStart(&forOp->getRegion(0).front());
        // 创建inttoptr操作
        Value ssb_cube_ptr = builder.create<LLVM::IntToPtrOp>(
            forOp.getLoc(),
            LLVM::LLVMPointerType::get(builder.getContext(), 11),  // 地址空间11
            ssb_addr
        );
        bufferPtrs.push_back(ssb_cube_ptr);
        // 创建load操作
        Value status_cube = builder.create<LLVM::LoadOp>(
            forOp.getLoc(), builder.getI32Type(), ssb_cube_ptr
        );

        WaitType = getWaitType("vector", forOp);
        for (auto i = 0; i < WaitType.size(); i++) {
            auto i32ConstAttr = mlir::IntegerAttr::get(i32Type, 1 << i);
            auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i32Type, i32ConstAttr);
            Value bufi_cube_val = builder.create<arith::AndIOp>(
                forOp.getLoc(), status_cube, buf_constant_set
            );

            Value flag_bufi_cube;
            // 创建比较操作
            if (WaitType[i] == 0) {
                flag_bufi_cube = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_cube_val, c0
                );
            }
            else {
                flag_bufi_cube = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_cube_val, buf_constant_set
                );
            }
            buf_vals.push_back(flag_bufi_cube);
        }
    }
    int bufIdx = 0;
    int groupIdx = 0;

    for (const auto &pair : bufferMap) {
        if (pair.second == 0) {
            continue;
        }
        
        // 获取对应的region迭代参数
        Value cnti = builder.create<arith::CmpIOp>(
            forOp.getLoc(), arith::CmpIPredicate::slt, 
            forOp.getRegionIterArgs()[forOp.getRegionIterArgs().size() - (bufferMap.size() - 1 - groupIdx)], 
            subLoopValue
        );
        
        // 计算该组中所有buffer值的AND
        Value finalBufVal = buf_vals[bufIdx];
        for (int count = 1; count < pair.second; count++) {
            finalBufVal = builder.create<arith::AndIOp>(
                forOp.getLoc(), finalBufVal, buf_vals[bufIdx + count]
            );
        }
        
        auto cond = builder.create<arith::AndIOp>(
            forOp.getLoc(), finalBufVal, cnti
        );
        if_conditions.push_back(cond);
        
        // 更新索引
        bufIdx += pair.second;
        groupIdx++;
    }
    int ifIndex = 0;
    int acc = 0;
    forOp.getBody()->walk([&](Operation* op) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        // 获取then区域
        Block* thenBlock = &ifOp.getThenRegion().front();
        
        // 找到then区域中的yield操作
        Operation* yieldOp = nullptr;
        for (auto& op : *thenBlock) {
            if (isa<scf::YieldOp>(op)) {
                yieldOp = &op;
                break;
            }
        }
        if (yieldOp) {
            builder.setInsertionPoint(yieldOp);
            
            if (isAIC) {
                // 创建插入的语句
                // %status_v2 = llvm.load %ssb_ptr : !llvm.ptr<11> -> i32
                Value status_v2_0 = builder.create<LLVM::LoadOp>(
                    yieldOp->getLoc(), 
                    builder.getIntegerType(32),  // i32类型
                    bufferPtrs[0]  // 假设ssb_ptr已在作用域中定义
                );
                Value status_v2_1 = builder.create<LLVM::LoadOp>(
                    yieldOp->getLoc(), 
                    builder.getIntegerType(32),  // i32类型
                    bufferPtrs[1]  // 假设ssb_ptr已在作用域中定义
                );
                Value buf_val_new_0 = status_v2_0;
                Value buf_val_new_1 = status_v2_1;
                auto bufferNum = bufferMap[ifIndex];
                for (int i = 0; i < bufferNum; i++) {
                    if (WaitType[ifIndex + i] == 0) {
                        auto i32ConstAttr = mlir::IntegerAttr::get(i32Type, 1 << (ifIndex + i));
                        auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                            scopeOp->getLoc(), i32Type, i32ConstAttr);
                        buf_val_new_0 = builder.create<arith::OrIOp>(
                            yieldOp->getLoc(),
                            buf_val_new_0,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                        buf_val_new_1 = builder.create<arith::OrIOp>(
                            yieldOp->getLoc(),
                            buf_val_new_0,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                    }
                    else {
                        int bitPos = ifIndex + i;
                        int basePattern = 0x7;
                        int finalValue = basePattern ^ (1 << bitPos);
                        auto i32ConstAttr = mlir::IntegerAttr::get(i32Type, finalValue);
                        auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                            scopeOp->getLoc(), i32Type, i32ConstAttr);
                        buf_val_new_0 = builder.create<arith::AndIOp>(
                            yieldOp->getLoc(),
                            buf_val_new_0,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                        buf_val_new_1 = builder.create<arith::AndIOp>(
                            yieldOp->getLoc(),
                            buf_val_new_1,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                    }
                }
                builder.create<LLVM::StoreOp>(
                    yieldOp->getLoc(),
                    buf_val_new_0,
                    bufferPtrs[0]
                );
                builder.create<LLVM::StoreOp>(
                    yieldOp->getLoc(),
                    buf_val_new_1,
                    bufferPtrs[1]
                );
                
            }
            else {
                // 创建插入的语句
                // %status_v2 = llvm.load %ssb_ptr : !llvm.ptr<11> -> i32
                Value status_v2 = builder.create<LLVM::LoadOp>(
                    yieldOp->getLoc(), 
                    builder.getIntegerType(32),  // i32类型
                    bufferPtrs[0]  // 假设ssb_ptr已在作用域中定义
                );
                Value buf_val_new = status_v2;
                auto bufferNum = bufferMap[ifIndex];
                for (int i = 0; i < bufferNum; i++) {
                    if (WaitType[acc + i] == 0) {
                        auto i32ConstAttr = mlir::IntegerAttr::get(i32Type, 1 << (acc + i));
                        auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                            scopeOp->getLoc(), i32Type, i32ConstAttr);
                        buf_val_new = builder.create<arith::OrIOp>(
                            yieldOp->getLoc(),
                            buf_val_new,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                    }
                    else {
                        int bitPos = acc + i;
                        int basePattern = 0x7;
                        int finalValue = basePattern ^ (1 << bitPos);
                        auto i32ConstAttr = mlir::IntegerAttr::get(i32Type, finalValue);
                        auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                            scopeOp->getLoc(), i32Type, i32ConstAttr);
                        buf_val_new = builder.create<arith::AndIOp>(
                            yieldOp->getLoc(),
                            buf_val_new,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                    }
                }
                acc += bufferNum;
                builder.create<LLVM::StoreOp>(
                    yieldOp->getLoc(),
                    buf_val_new,
                    bufferPtrs[0]
                );
            }
            ifIndex ++;
        }
      }
    });

    return if_conditions;
}

void ReplaceIf(scf::ForOp forOp, SmallVector<Value> conditions, OpBuilder &builder, ModuleOp moduleOp) {
    SmallVector<scf::IfOp> ifToProcess;
    auto aiCAttr = hivm::TCoreTypeAttr::get(
            builder.getContext(),
            hivm::TCoreType::CUBE);
    forOp.getBody()->walk([&](Operation* op) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        ifToProcess.push_back(ifOp);
      }
    });
    auto i32Type = builder.getIntegerType(32);
    IRMapping IRMap;
    for (int i = 0; i < ifToProcess.size(); i++) {
        auto ifOp = ifToProcess[i];
        auto parentOp = ifOp->getParentOp();
        auto loc = ifOp.getLoc();
        // 获取for循环的iterargs（迭代参数）
        auto iterArgs = forOp.getRegionIterArgs();
        if (iterArgs.size() < conditions.size()) {
            return;
        }
        auto thenYieldOp = dyn_cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
        SmallVector<Value> thenResults;
        if (thenYieldOp) {
            // 如果已有返回值，保留它们
            for (auto result : thenYieldOp.getResults()) {
                thenResults.push_back(result);
            }
        }
        // 创建新的else区域，返回两个迭代参数
        SmallVector<Value> elseResults;
        auto elseYieldOp = dyn_cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
        if (elseYieldOp) {
            for (auto result : elseYieldOp.getResults()) {
                elseResults.push_back(result);
            }
        }
        // 获取最后两个迭代参数
        Value iterArgMinus = iterArgs[iterArgs.size() - (conditions.size() - i)];
        // 创建新的then区域，返回两个迭代参数
        thenResults.push_back(iterArgMinus);
        elseResults.push_back(iterArgMinus);
        
        // 保存原有的操作，以便后续克隆
        SmallVector<Operation*> thenOps;
        for (auto &op : ifOp.getThenRegion().front()) {
            thenOps.push_back(&op);
        }
        
        SmallVector<Operation*> elseOps;
        if (!ifOp.getElseRegion().empty()) {
            for (auto &op : ifOp.getElseRegion().front()) {
                elseOps.push_back(&op);
            }
        }
        SmallVector<Type> resultTypes;
        for (auto val : thenResults) {
            resultTypes.push_back(val.getType());
        }
        // 创建新的scf.if操作
        builder.setInsertionPoint(ifOp);
        auto newIfOp = builder.create<scf::IfOp>(
            loc,
            resultTypes,
            conditions[i],
            /*withElseRegion=*/true);
        
        // 处理then区域
        auto &newThenBlock = newIfOp.getThenRegion().front();
        builder.setInsertionPointToStart(&newThenBlock);
        
        // 克隆then区域的操作
        for (auto op : thenOps) {
            if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
                // 处理yield的操作数映射
                SmallVector<Value> mappedOperands;
                for (auto operand : yieldOp->getOperands()) {
                    mappedOperands.push_back(IRMap.lookupOrDefault(operand));
                }
                // 获取最后两个迭代参数
                Value iterArgMinus = iterArgs[iterArgs.size() - (conditions.size() - i)];

                auto c0i32ConstAttr = mlir::IntegerAttr::get(i32Type, 1);
                auto c0i32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                    forOp->getLoc(), i32Type, c0i32ConstAttr);
        
                // %ssb_addr = arith.addi %ssb_addr_offset, %c32_i64 : i64
                auto AddIOp = builder.create<mlir::arith::AddIOp>(
                    forOp->getLoc(),
                    iterArgMinus,
                    c0i32ConstOp.getResult());
                // 这里加个add1
                mappedOperands.push_back(AddIOp);
                builder.create<scf::YieldOp>(loc, mappedOperands);
            } else {
                auto newOp = builder.clone(*op, IRMap);
                IRMap.map(op->getResults(), newOp->getResults());
            }
        }
        
        // 处理else区域
        auto &newElseBlock = newIfOp.getElseRegion().front();
        builder.setInsertionPointToStart(&newElseBlock);
        // 克隆else区域的操作
        for (auto op : elseOps) {
            if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
                // 处理yield的操作数映射
                SmallVector<Value> mappedOperands;
                for (auto operand : yieldOp->getOperands()) {
                    mappedOperands.push_back(IRMap.lookupOrDefault(operand));
                }
                Value iterArgMinus = iterArgs[iterArgs.size() - (conditions.size() - i)];
                mappedOperands.push_back(iterArgMinus);
                builder.create<scf::YieldOp>(loc, mappedOperands);
            } else {
                auto newOp = builder.clone(*op, IRMap);
                IRMap.map(op->getResults(), newOp->getResults());
            }
        }
            
        // 替换原有if操作的使用
        // 首先，将原if操作的结果替换为新if操作的对应结果
        for (unsigned j = 0; j < ifOp.getNumResults(); ++j) {
            ifOp.getResult(j).replaceAllUsesWith(newIfOp.getResult(j));
        }
        // 获取新if操作所在的块
        Block* newIfBlock = ifOp->getBlock();
        // 在for循环体内替换迭代参数的使用
        forOp.getBody()->walk([&](Operation* op) {
            // 检查操作是否与新ifOp在同一个块中
            Block* opBlock = op->getBlock();
            if (opBlock != newIfBlock) {
                // 不在同一个块中，跳过
                return;
            }
            if (op->isBeforeInBlock(newIfOp)) {
                return; // 只处理if操作之后的use
            }
            for (unsigned j = 0; j < op->getNumOperands(); ++j) {
                for (auto argIndex = 0; argIndex < conditions.size(); argIndex ++) {
                    // 获取最后两个迭代参数
                    Value iterArgMinus = iterArgs[iterArgs.size() - (conditions.size() - i)];
                    if (op->getOperand(j) == iterArgMinus) {
                        op->setOperand(j, newIfOp.getResults()[newIfOp.getNumResults() - 1]);
                    }
                }
            }
        });
        
        // // 删除原有的if操作
        ifOp.erase();
        }
}

void FlowSssbuf(ModuleOp module) {
    mlir::OpBuilder builder(module.getContext());
    // 收集所有需要转换的循环
    SmallVector<scf::ForOp> targetLoops;
    int numBuffer = 0;
    int subLoop = 0;
    module.walk([&](Operation* op) {
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            // 检查循环是否包含特定的 sync_block_set 操作
            bool hasSyncBlockSet = false;
            forOp.walk([&](Operation *op) {
                if (isa<SyncBlockSetOp>(op)) {
                    if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
                        if (forOp == ifOp->getParentOp()) {
                            hasSyncBlockSet = true;
                        }
                    }
                }
            });
            
            if (hasSyncBlockSet) {
                if (llvm::find(targetLoops, forOp) == targetLoops.end()) {
                    targetLoops.push_back(forOp);
                }
            }
        }
        else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
            // 检查操作数是否由常量操作定义
            Value operand1 = makeTensorPtrOp.getOperands()[1];
            Value operand2 = makeTensorPtrOp.getOperands()[3];
            uint64_t SeqValue = 0;
            uint64_t BNValue = 1;
            if (auto constOp = operand1.getDefiningOp<arith::ConstantOp>()) {
                // 获取常量的值
                Attribute valueAttr = constOp.getValue();
                
                // 尝试提取整数属性
                if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
                    APInt apValue = intAttr.getValue();
                    // 根据位宽获取值
                    if (apValue.getBitWidth() <= 64) {
                        SeqValue = apValue.getZExtValue();
                    }
                } 
            }
            if (auto constOp = operand2.getDefiningOp<arith::ConstantOp>()) {
                // 获取常量的值
                Attribute valueAttr = constOp.getValue();
                
                // 尝试提取整数属性
                if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
                    APInt apValue = intAttr.getValue();
                    // 根据位宽获取值
                    if (apValue.getBitWidth() <= 64) {
                        BNValue = apValue.getZExtValue();
                    }
                } 
            }
            subLoop = SeqValue / BNValue;
        }
        else if (dyn_cast<hivm::FixpipeOp>(op)) {
            numBuffer ++;
        }
        else if (dyn_cast<hivm::CopyOp>(op)) {
            numBuffer ++;
        }
      
    });
    SmallVector<scf::ForOp> transformLoops;
    // 转换每个目标循环
    for (scf::ForOp forOp : targetLoops) {
      auto newforOp = transformLoop(forOp, builder);
      transformLoops.push_back(newforOp);
    }
    
    for (scf::ForOp forOp : transformLoops) {
        auto bufvals = addBufValLoop(forOp, numBuffer, subLoop, builder);
        ReplaceIf(forOp, bufvals, builder, module);
    }
}

bool isTransOp(mlir::Operation *op) {
  auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op);
  if (fixpipeOp)
    return true;

  auto copyOp = dyn_cast<hivm::CopyOp>(op);
  if (!copyOp)
    return false;
  else {
      
      // if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
      //   MemRefType MemRefTy = dyn_cast<MemRefType>(allocOp.getResult().getType());
      //   auto AddrSpace = dyn_cast_or_null<hivm::AddressSpaceAttr>(MemRefTy.getMemorySpace());

      llvm::outs() << "Copy Op: " << *op << "\n";
      Value copySrc = copyOp.getODSOperands(0).front();
      llvm::outs() << "copySrc: " << copySrc << "\n";
      MemRefType copySrcTy = dyn_cast<MemRefType>(copySrc.getType());
      auto SrcAddrSpace = dyn_cast_or_null<hivm::AddressSpaceAttr>(copySrcTy.getMemorySpace());
      bool isSrcUbSpace = SrcAddrSpace.getAddressSpace() == hivm::AddressSpace::UB;

      Value copyDst = copyOp.getODSOperands(1).front();
      MemRefType copyDstTy = dyn_cast<MemRefType>(copyDst.getType());
      auto DstAddrSpace = dyn_cast_or_null<hivm::AddressSpaceAttr>(copyDstTy.getMemorySpace());
      bool isDstCbufSpace = DstAddrSpace.getAddressSpace() == hivm::AddressSpace::L1;

      return isSrcUbSpace && isSrcUbSpace;
  }
}

void FindAndMarkBuffer(ModuleOp module) {
  OpBuilder builder(module.getContext());
  unsigned int BufferIdx = 0;
  Type idxType = builder.getI32Type();
  StringAttr setFlagAttr = builder.getStringAttr("Set flag");
  StringAttr waitFlagAttr = builder.getStringAttr("Wait flag");
  IntegerAttr idxAttr = builder.getI32IntegerAttr(BufferIdx);

  module.walk([&](mlir::Operation *op) {

    if (isTransOp(op)) {
      llvm::outs() << "Buffer idx" << BufferIdx << "\n";
      llvm::outs() << "Trans Op" << *op << "\n";
      Value SharedBuffer;
      if (auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op)) {
        SharedBuffer = fixpipeOp.getODSOperands(1).front();
      } else {
        auto copyOp = dyn_cast<hivm::CopyOp>(op);
        SharedBuffer = copyOp.getODSOperands(1).front();
      }
      llvm::outs() << "SharedBuffer" << SharedBuffer << "\n";

      if (!SharedBuffer) {
        op->emitWarning("fixpipe op has empty output operand!");
        return;
      }

      // 在Buffer的生产op后set flag标记，在Buffer消费op前增加wait flag标记
      op->setAttr("Buffer idx", builder.getI32IntegerAttr(BufferIdx));
      op->setAttr("Wait Flag", builder.getI32IntegerAttr(0));
      op->setAttr("Set Flag", builder.getI32IntegerAttr(1));

      for (Operation *consumerOp : SharedBuffer.getUsers()) {
        if (consumerOp == op) 
          continue;
        if (!consumerOp) continue;
        
        llvm::outs() << "consumerOp: " << *consumerOp << "\n";
        
        consumerOp->setAttr("Buffer idx", builder.getI32IntegerAttr(BufferIdx));
        consumerOp->setAttr("Wait Flag", builder.getI32IntegerAttr(0));
      }
      BufferIdx++;
    }
  });
}

// 结构体存 wait-set 区块信息
struct WaitSetRegion {
  Operation *waitOp;
  Operation *lastSetOp;
  SmallVector<Operation *> opsToMove;
  bool hasCopyOrFixpipe = false;
};

struct MergedRegion {
  SmallVector<WaitSetRegion *> regions;
  SmallVector<Operation *> opsToMove;
  SmallVector<Value> yieldValues;
  SmallVector<Type> resultTypes;
};

void MoveIterArgUsersIntoIf(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // iter_arg -> mergedRegion index
  DenseMap<BlockArgument, int> iterArgToRegion;

  for (int r = 0; r < mergedRegions.size(); ++r) {
    MergedRegion &mr = mergedRegions[r];

    for (Operation *op : mr.opsToMove) {
      for (Value v : op->getOperands()) {
        if (auto barg = mlir::dyn_cast<BlockArgument>(v)) {
          if (barg.getOwner() == &body) {
            iterArgToRegion.try_emplace(barg, r);
          }
        }
      }
    }
  }

  if (iterArgToRegion.empty())
    return;

  // 找最后一个 mergedRegion 的最后一个 op
  Operation *lastOp = nullptr;
  for (MergedRegion &mr : mergedRegions)
    lastOp = mr.opsToMove.back();

  if (!lastOp)
    return;

  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body)
    opIndex[&op] = idx++;

  int startIdx = opIndex[lastOp] + 1;

  // 扫描 for body 尾部 op
  for (Operation &op : body) {
    if (opIndex[&op] < startIdx)
      continue;

    llvm::SmallDenseSet<int, 2> usedRegions;
    for (Value v : op.getOperands()) {
      if (auto barg = mlir::dyn_cast<BlockArgument>(v)) {
        auto it = iterArgToRegion.find(barg);
        if (it != iterArgToRegion.end())
          usedRegions.insert(it->second);
      }
    }

    // 必须且只能依赖一个 mergedRegion
    if (usedRegions.size() != 1)
      continue;

    int target = *usedRegions.begin();

    mergedRegions[target].opsToMove.push_back(&op);
  }
}

void ComputeYieldForMergedRegion(
    MergedRegion &mr, Block &body) {

  mr.yieldValues.clear();
  mr.resultTypes.clear();

  SmallPtrSet<Operation *, 32> inRegion(
      mr.opsToMove.begin(), mr.opsToMove.end());

  for (Operation *op : mr.opsToMove) {
    for (Value res : op->getResults()) {
      bool usedOutside = false;

      for (OpOperand &use : res.getUses()) {
        Operation *user = use.getOwner();

        // 不在同一个 for body，交给外层处理（通常不会出现）
        if (user->getBlock() != &body)
          continue;

        // 只要有一个 use 在 region 外，就必须 yield
        if (!inRegion.contains(user)) {
          usedOutside = true;
          break;
        }
      }

      if (usedOutside) {
        mr.yieldValues.push_back(res);
        mr.resultTypes.push_back(res.getType());
      }
    }
  }
}

int findTargetRegion(
    Operation *startOp,
    Block &body,
    DenseMap<Operation *, int> &opToRegion) {

  SmallVector<Operation *> worklist{startOp};
  SmallPtrSet<Operation *, 16> visited;

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!visited.insert(op).second)
      continue;

    auto it = opToRegion.find(op);
    if (it != opToRegion.end())
      return it->second;

    for (Value operand : op->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;

      Operation *defOp = operand.getDefiningOp();
      if (defOp && defOp->getBlock() == &body)
        worklist.push_back(defOp);
    }
  }

  return -1;
}

void greedyAbsorbToRegion(
    Operation *startOp,
    int regionIdx,
    int lowerBound,
    Block &body,
    DenseMap<Operation *, int> &opIndex,
    DenseMap<Operation *, int> &opToRegion,
    SmallVector<MergedRegion> &mergedRegions) {

  auto &mr = mergedRegions[regionIdx];

  SmallVector<Operation *> worklist;
  SmallPtrSet<Operation *, 32> visited(
      mr.opsToMove.begin(), mr.opsToMove.end());

  // 先把 startOp 本身吸收（如果还没被吸收）
  if (!opToRegion.count(startOp)) {
    mr.opsToMove.push_back(startOp);
    opToRegion[startOp] = regionIdx;
    visited.insert(startOp);
  }

  worklist.push_back(startOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    for (Value operand : op->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;

      Operation *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != &body)
        continue;

      int defIdx = opIndex[defOp];

      // 超过前一个 region 的末尾
      if (defIdx < lowerBound)
        continue;

      auto it = opToRegion.find(defOp);

      // 不能跨到其他 region
      if (it != opToRegion.end() &&
          it->second != regionIdx)
        continue;

      // 去重
      if (!visited.insert(defOp).second)
        continue;

      // 吸收 defOp
      mr.opsToMove.push_back(defOp);
      opToRegion[defOp] = regionIdx;
      worklist.push_back(defOp);
    }
  }
}

// 以 forOp 的 yield value 为中心
// 决定它应该归属哪个 mergedRegion, 然后再向前吸 operand
void ExpandMergedRegionOpsForAIV(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // 记录 block 中 op 顺序
  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body)
    opIndex[&op] = idx++;

  // 建立 op -> region 映射
  DenseMap<Operation *, int> opToRegion;
  for (int r = 0; r < mergedRegions.size(); ++r)
    for (Operation *op : mergedRegions[r].opsToMove)
      opToRegion[op] = r;

  // 取 scf.yield
  auto yieldOp =
      cast<scf::YieldOp>(body.getTerminator());

  // 依次处理每个 yield value（按编号顺序）
  for (Value yv : yieldOp.getOperands()) {

    Operation *defOp = yv.getDefiningOp();
    if (!defOp || defOp->getBlock() != &body)
      continue;

    int targetRegion = -1;

    // 如果已经在 region 中
    auto it = opToRegion.find(defOp);
    if (it != opToRegion.end()) {
      targetRegion = it->second;
    } else {
      // 否则向前搜索确定归属
      targetRegion =
          findTargetRegion(defOp, body, opToRegion);
    }

    if (targetRegion == -1)
      continue;

    // 计算边界 lowerBound
    int lowerBound = 0;

    if (targetRegion > 0) {
      Operation *prevLast =
          mergedRegions[targetRegion - 1]
              .opsToMove.back();
      lowerBound = opIndex[prevLast] + 1;
    }

    // 真正贪心吸收
    greedyAbsorbToRegion(defOp,
                         targetRegion,
                         lowerBound,
                         body,
                         opIndex,
                         opToRegion,
                         mergedRegions);
  }

  // 每个 region 内按 block 顺序排序
  for (auto &mr : mergedRegions) {
    llvm::sort(mr.opsToMove,
               [&](Operation *a, Operation *b) {
                 return opIndex[a] < opIndex[b];
               });
  }
}

// 以 mergedRegion 为中心, 向前吸 operand
void ExpandMergedRegionOpsForAIC(scf::ForOp forOp,
                           SmallVector<MergedRegion> &mergedRegions) {
  Block &body = forOp.getRegion().front();

  // 记录每个 mergedRegion 的起始 op index
  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body) {
    opIndex[&op] = idx++;
  }

  for (int r = 0; r < mergedRegions.size(); ++r) {
    MergedRegion &mr = const_cast<MergedRegion &>(mergedRegions[r]);

    // 本 mergedRegion 的最早 op
    Operation *firstOp = mr.opsToMove.front();
    int lowerBound = 0;

    // 边界: 前一个 mergedRegion 的最后一个 op
    if (r > 0) {
      Operation *prevLast =
          mergedRegions[r - 1].opsToMove.back();
      lowerBound = opIndex[prevLast] + 1;
    }

    SmallVector<Operation *> worklist(mr.opsToMove.begin(),
                                      mr.opsToMove.end());
    SmallPtrSet<Operation *, 32> visited(
        mr.opsToMove.begin(), mr.opsToMove.end());

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      
      // 往前吸收operand
      for (Value operand : op->getOperands()) {
        // BlockArgument
        if (mlir::isa<BlockArgument>(operand))
          continue;

        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;

        // 不在 for body
        if (defOp->getBlock() != &body)
          continue;

        int defIdx = opIndex[defOp];

        // 超出允许向前吸收的边界
        if (defIdx < lowerBound)
          continue;

        // 已经在 opsToMove
        if (!visited.insert(defOp).second)
          continue;

        // 吸收这个 defOp
        mr.opsToMove.push_back(defOp);
        worklist.push_back(defOp);
      }
    }

    // 最后按原 block 顺序排序
    llvm::sort(mr.opsToMove,
               [&](Operation *a, Operation *b) {
                 return opIndex[a] < opIndex[b];
               });
  }
}

void ExpandMergedRegionOps(scf::ForOp forOp,
                           SmallVector<MergedRegion> &mergedRegions) {
  bool isInAIV = false;
  auto scopeOp = forOp->getParentOfType<scope::ScopeOp>();
  if (!scopeOp)
    return;
  
  auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::TCoreTypeAttr::name);

  if (coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR) {
    isInAIV = true;
  }

  if (isInAIV) ExpandMergedRegionOpsForAIV(forOp, mergedRegions);
  else ExpandMergedRegionOpsForAIC(forOp, mergedRegions);
}

void MergeWaitSetRegions(SmallVector<WaitSetRegion> &regions,
                         SmallVector<MergedRegion> &merged) {
  for (int i = 0; i < regions.size();) {
    MergedRegion mr;
    mr.regions.push_back(&regions[i]);
    mr.opsToMove.append(regions[i].opsToMove);

    int j = i;
    while (!regions[j].hasCopyOrFixpipe &&
           j + 1 < regions.size()) {
      j++;
      mr.regions.push_back(&regions[j]);
      mr.opsToMove.append(regions[j].opsToMove);
    }

    merged.push_back(std::move(mr));
    i = j + 1;
  }

  for (MergedRegion &mr : merged) {
    SmallPtrSet<Value, 16> regionValues;
    SmallPtrSet<Operation *, 16> opSet;

    for (Operation *op : mr.opsToMove)
      opSet.insert(op);

    for (Operation *op : mr.opsToMove) {
      for (Value v : op->getResults()) {
        bool usedOutside = false;
        for (OpOperand &use : v.getUses()) {
          Operation *user = use.getOwner();
          if (!opSet.contains(user) &&
              user->getBlock() == op->getBlock()) {
            usedOutside = true;
            break;
          }
        }
        if (usedOutside) {
          mr.yieldValues.push_back(v);
          mr.resultTypes.push_back(v.getType());
        }
      }
    }
  }
}

void GetBlockInfos(SmallVector<WaitSetRegion> &regions, Block &body) {
  for (auto it = body.begin(); it != body.end();) {
    Operation *op = &*it;
    // llvm::outs() <<"op: "<< *op << "\n";
    if (!isa<SyncBlockWaitOp>(op)) {
      it++;
      continue;
    }

    Operation *waitOp = op;
    Operation *lastSetOp = nullptr;

    // 扫描到下一个 wait, 收集所有 set
    auto curIt = std::next(it);
    auto endIt = curIt;
    int setOpCount = 0;
    SmallVector<Operation *> opsInRegion;
    for (; curIt != body.end(); ++curIt) {
      Operation *curOp = &*curIt;
      if (isa<SyncBlockWaitOp>(curOp) && setOpCount >= 1) break;
      if (isa<SyncBlockSetOp>(curOp)) {
        setOpCount++;
        endIt = curIt; //setop的位置
        lastSetOp = curOp;  // 最后一个 set
      }
    }

    if (!lastSetOp) {
      it = curIt;
      continue;
    }// 没有 set, 不包

    // 收集 [wait, ..., lastSet] 之间的 ops
    bool hasCopyOrFixpipe = false;
    for (auto it2 = it; it2 != std::next(endIt); ++it2) {
      Operation *curOp = &*it2;
      opsInRegion.push_back(curOp);
      if (isa<CopyOp>(curOp) || isa<FixpipeOp>(curOp)) {
        hasCopyOrFixpipe = true;
      }
    }
    
    it = endIt++;
    regions.push_back({waitOp, lastSetOp, opsInRegion, hasCopyOrFixpipe});
  }
}

Value findIterArg(Value v, Type t) {
    SmallVector<Value> worklist = {v};
    SmallPtrSet<Value, 16> visited;

    while (!worklist.empty()) {
        Value cur = worklist.front();
        worklist.erase(worklist.begin());
        if (!visited.insert(cur).second)
            continue;

        // 匹配scf.for原始迭代参数, 直接返回
        if (auto b = mlir::dyn_cast<BlockArgument>(cur)) {
            auto forOp = mlir::dyn_cast<scf::ForOp>(b.getOwner()->getParentOp());
            if (forOp && b.getType() == t) {
                for (Value iterArg : forOp.getRegionIterArgs()) {
                    if (iterArg.getAsOpaquePointer() == b.getAsOpaquePointer()) {
                        return b;
                    }
                }
            }
        }

        Operation *defOp = cur.getDefiningOp();
        if (!defOp) continue;

        // 核心逻辑：如果当前值是scf.if的结果
        // 进入then块找源头
        if (auto ifOp = mlir::dyn_cast<scf::IfOp>(defOp)) {
            Block &thenBlock = ifOp.getThenRegion().front();
            // 找到then块最后一个op（scf.yield）
            // 取其operands（即ifOp结果的源头值）
            for (auto &innerOp : llvm::reverse(thenBlock)) {
                if (auto yieldOp = mlir::dyn_cast<scf::YieldOp>(&innerOp)) {
                    // 按索引匹配: cur是ifOp的第n个结果, 取yieldOp的第n个operand
                    for (auto [idx, res] : llvm::enumerate(ifOp.getResults())) {
                        if (res.getAsOpaquePointer() == cur.getAsOpaquePointer()) {
                            Value srcVal = yieldOp.getOperand(idx);
                            if (!visited.count(srcVal)) worklist.push_back(srcVal);
                            break;
                        }
                    }
                    break; // 找到yield即退出, 无需遍历其他op
                }
            }
        } else {
            // 非if结果值
            // 正常往前追溯operands
            for (Value operand : defOp->getOperands()) {
                if (!visited.count(operand)) worklist.push_back(operand);
            }
        }
    }

    llvm::outs() << "未找到迭代参数, 返回原值: "; v.print(llvm::outs()); llvm::outs() << "\n";
    return v;
}

void FindDependValues (SmallVector<Value> &dependValues, SmallVector<MergedRegion> mergedRegions) {
  dependValues.clear();
  for (auto &curMR : mergedRegions) {
    for (Value yieldValue : curMR.yieldValues) {
      llvm::outs() << "yieldValue: "<< yieldValue << "\n";
      // 遍历当前区域的yieldValue的所有user OP，判断是否存在依赖关系
      for (OpOperand &use : yieldValue.getUses()) {
        Operation *userOp = use.getOwner();

        llvm::outs() << "userOp: "<< *userOp << "\n";
        bool isUserInOtherRegion = false;
        for (auto &otherMR : mergedRegions) {
          // 跳过当前区域，只检查yieldValue是否被其他区域使用
          if (&otherMR == &curMR) continue;

          // 只要有一个 userOp在 otherMR 的 opsToMove 列表中，就认为是dependValue
          llvm::outs() << "judge comtain\n";
          for (size_t k = 0; k < otherMR.opsToMove.size(); k++) {
            llvm::outs() << "otherMR op: " << *(otherMR.opsToMove[k]) << "\n";
          }
          llvm::outs() << "otherMR end\n";
          if (llvm::is_contained(otherMR.opsToMove, userOp)) {
            isUserInOtherRegion = true;
            llvm::outs() << "is_contained\n";
            break;
          }
        }

        // 无重复的添加依赖变量
        if (isUserInOtherRegion) {
          if (!llvm::is_contained(dependValues, yieldValue)) {
            dependValues.push_back(yieldValue);
          }
          break;
        }
      }
    }
  }
}

void UpdateMergedRegionsWithNewForOp(SmallVector<MergedRegion> &mergedRegions, IRMapping &mapper) {
  for (auto &mr : mergedRegions) {
    // WaitSetRegion 后续已经不使用了，直接释放，否则会出现野指针
    SmallVector<WaitSetRegion *> newRegions;
    newRegions.clear();
    mr.regions = newRegions;
    // 更新 opsToMove 列表
    llvm::outs() << "before \n";
    for (auto &op : mr.opsToMove) {
      llvm::outs() << "opsToMove: " << op << ", " << *op << '\n';
    }
    SmallVector<Operation *> newOpsToMove;
    newOpsToMove.clear();
    for (Operation *op : mr.opsToMove) {
      if (op) {
        Operation *newOp = mapper.lookupOrNull(op);
        newOpsToMove.push_back(newOp);
      }
    }
    mr.opsToMove = newOpsToMove;
    llvm::outs() << "after \n";
    for (auto &op : mr.opsToMove) {
      llvm::outs() << "opsToMove: " << op << ", " << *op << '\n';
    }
    // 更新 yieldValues 列表
    SmallVector<Value> newYieldValues;
    newYieldValues.clear();
    for (Value v : mr.yieldValues) {
      if (v) {
        newYieldValues.push_back(mapper.lookupOrNull(v));
      }
    }
    mr.yieldValues = newYieldValues;
    // resultTypes 是type 类型，无需更新
  }
}


void AddArgsForDependValues (scf::ForOp forOp, SmallVector<Value> &dependValues, SmallVector<MergedRegion> &mergedRegions, ModuleOp module) {
  OpBuilder moduleBuilder(module.getContext());
  SmallVector<Type> valueTypes;
  valueTypes.clear();

  if (dependValues.empty()) {
    return ;
  } else {
    for (Value v : dependValues) {
      Type valueType = v.getType();
      valueTypes.push_back(valueType);
    }
  }

  // 为每个 dependValue 创建一个初始值（可能不存在相同shape和type的常量tensor）
  SmallVector<Value> initTensors;
  initTensors.clear();
  module.walk([&](Operation *op) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      moduleBuilder.setInsertionPoint(constOp);
      for (Type valueType : valueTypes) {
        auto tensorType = dyn_cast<RankedTensorType>(valueType);
        auto zeroAttr = moduleBuilder.getZeroAttr(tensorType);
        Value zeroTensor = moduleBuilder.create<arith::ConstantOp>(constOp.getLoc(), tensorType, zeroAttr);
        initTensors.push_back(zeroTensor);
      }
    return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  auto initArgs = forOp.getInitArgs();

  // 构建新的初始化参数列表
  SmallVector<Value> newInitArgs(initArgs.begin(), initArgs.end());
  // 添加 dependValue 的初始化参数
  for (Value initTensor : initTensors) {
    newInitArgs.push_back(initTensor);
  }

  // 获取原循环的边界和步长
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();

  // 创建新的 ForOp，插入点位于原操作之前
  OpBuilder builder(forOp);
  auto newForOp = builder.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, newInitArgs);

  // 获取新循环的 region 块（已自动包含循环索引和迭代参数）
  Block &newBlock = newForOp.getRegion().front();
  Block &oldBlock = forOp.getRegion().front();

  // 建立块参数的映射：原块参数 -> 新块参数
  IRMapping mapper;
  for (unsigned i = 0; i < oldBlock.getNumArguments(); ++i) {
      mapper.map(oldBlock.getArgument(i), newBlock.getArgument(i));
  }
  // 将原循环体中的操作（不包括终结符）克隆到新块中
  // 同时按照顺序克隆新的 dependValues
  SmallVector<Value> newDependValues = dependValues;
  int cnt = 0;
  builder.setInsertionPointToStart(&newBlock);
  for (auto &op : oldBlock) {
      auto newOp = builder.clone(op, mapper);
      // dependValue 的定义OP 可能有多个 result
      for (size_t i = 0; i < dependValues.size(); i++) {
        Operation *defineOp = dependValues[i].getDefiningOp();
        if (defineOp == &op) {
          unsigned int index = cast<OpResult>(dependValues[i]).getResultNumber();
          newDependValues[i] = newOp->getResult(index);
          cnt++;
          break;
        }
      }
  }
  // 判断是否找到了所有的 dependValue
  if (newDependValues.size() != cnt) {
    llvm::outs() << "can not find the depend value! \n";
    return;
  }
  dependValues = newDependValues;

  // 更新 mergedRegions 中的 op 为新的for循环的 op
  UpdateMergedRegionsWithNewForOp(mergedRegions, mapper);
  
  // 创建新的循环 yield 操作：原操作数 + dependValues
  auto oldYield = cast<scf::YieldOp>(newBlock.getTerminator());
  SmallVector<Value> newYieldOps(oldYield.getOperands());
  // 按顺序增加找到的 dependvalue
  for (Value v : newDependValues) {
    newYieldOps.push_back(v);
  }
  builder.setInsertionPointToEnd(&newBlock);
  builder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOps);
  oldYield.erase();

  // 将原 forOp 的所有使用替换为新 forOp
  int oldResultNum = forOp->getResults().size();
  for (auto it : llvm::zip(forOp->getResults(), newForOp->getResults().take_front(oldResultNum))) {
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }
  forOp.erase();
}

void ComputeElseYieldValues (MergedRegion mergedRegion, SmallVector<Value> &elseYieldValues, SmallVector<Value> dependValues) {
  int idx = 0;
  for (Value v : mergedRegion.yieldValues) {
      Type yieldType = mergedRegion.resultTypes[idx];
      elseYieldValues.push_back(findIterArg(v, yieldType));
      idx++;
  }
}

void ComputeElseYieldValuesV2 (MergedRegion mergedRegion, SmallVector<Value> &elseYieldValues, SmallVector<Value> dependValues) {
  // 对于yieldValues，其中的 yield value 一定是被 for op yield 所引用，或者被其他 region 所使用
  auto forOp = dyn_cast<scf::ForOp>(mergedRegion.yieldValues[0].getDefiningOp()->getBlock()->getParentOp());
  if (!forOp) {
    llvm::outs() << "define op's parent is not ForOp \n";
    return;
  }
  auto iterArgs = forOp.getRegionIterArgs();
  auto forYieldValues = forOp.getYieldedValues();
  
  // 新增的与 dependvalue 相关的 initarg 是接在原本for循环args后面，数量与dependvalue数量相等
  int baseDependIdx = iterArgs.size() - dependValues.size();

  int idx = 0;
  for (Value v : mergedRegion.yieldValues) {
      Type yieldType = mergedRegion.resultTypes[idx];
      // yieldValue 中是dependvalue 的情况下
      // else yield value 使用对应的新增 iterargs
      if (llvm::is_contained(dependValues, v)) {
        int dependIdx = 0;
        for (; dependIdx < dependValues.size(); dependIdx++) {
          if (v == dependValues[dependIdx]) {
            break;
          }
        }
        elseYieldValues.push_back(iterArgs[baseDependIdx + dependIdx]);
      } else {
        elseYieldValues.push_back(findIterArg(v, yieldType));
      }
      idx++;
  }
}

void CreateIfOps (SmallVector<MergedRegion> &mergedRegions, SmallVector<Value> dependValues) {
  for (auto &region : mergedRegions) {
    Operation *insertPt = region.opsToMove.front();
    OpBuilder builder(insertPt);
    Location loc = insertPt->getLoc();
    Value cond = builder.create<arith::ConstantOp>(
        loc, builder.getI1Type(), builder.getBoolAttr(true));

    bool needsYield = !region.yieldValues.empty();
    scf::IfOp ifOp;
    if (needsYield)
      ifOp = builder.create<scf::IfOp>(loc, region.resultTypes, cond, true);
    else
      ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);

    // 获取if yield value 在 else块 返回值
    SmallVector<Value> elseYieldValues;
    if (needsYield) {
    //   ComputeElseYieldValues(region, elseYieldValues, dependValues);
      ComputeElseYieldValuesV2(region, elseYieldValues, dependValues);
    }

    // 将op移进then块
    Block &thenBlock = ifOp.getThenRegion().front();
    for (Operation *m : llvm::reverse(region.opsToMove)) {
      m->moveBefore(&thenBlock, thenBlock.begin());
    }

    // 创建 then/else yield
    if (needsYield) {
      OpBuilder thenBuilder(builder.getContext());
      thenBuilder.setInsertionPointToEnd(&thenBlock);
      thenBuilder.create<scf::YieldOp>(loc, region.yieldValues);

    //   else block
      Block &elseBlock = ifOp.getElseRegion().front();
      OpBuilder elseBuilder(&elseBlock, elseBlock.end());
      elseBuilder.create<scf::YieldOp>(loc, elseYieldValues);

      // 替换外部使用
      Block *block = ifOp->getBlock();
      auto ifIt = Block::iterator(ifOp);

      for (size_t i = 0; i < region.yieldValues.size(); ++i) {
        Value oldVal = region.yieldValues[i];
        Value newVal = ifOp.getResult(i);

        SmallVector<OpOperand *> usesToReplace;

        for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
          Operation *user = use.getOwner();
          // 同一个 block, user 必须在 ifOp 之后, 不能在 ifOp 内部（then / else）
          if (user->getBlock() != ifOp->getBlock() || !ifOp->isBeforeInBlock(user) || user->getParentOp() == ifOp)
            continue;
          usesToReplace.push_back(&use);
        }

        for (OpOperand *use : usesToReplace)
          use->set(newVal);
      }

    }

    llvm::outs() <<"Create ifOp: "<< *ifOp << "\n";
  }
}

// AIV:
// 1. 先收集基本的wait/set region, 依据hivm.hir.copy完成基本的合并
// 2. 根据yield value: 
//     1) 如果yield value已经在region里，往前搜寻它的operand，加到if region里
//     2) 不在region里，往前找operand，直到某个op的operand存在于if region里，把这块区域加到region里

// AIC:
// 1. 先收集基本的wait/set region, 依据hivm.hir.fixpipe完成基本的合并
// 2. 往前吸收operand
// 3. 把for op最后的对于iter_arg的操作移进对应的if region
void AddIfCondition(ModuleOp module) {
  SmallVector<scf::ForOp> forOpList;
  SmallVector<SmallVector<MergedRegion>, 1> regionList;

  module.walk([&](scf::ForOp forOp) {
    Block &body = forOp.getRegion().front();
    SmallVector<WaitSetRegion> regions;

    // 获取基本的wait-set分块信息
    GetBlockInfos(regions, body);

    SmallVector<MergedRegion> mergedRegions;
    // 合并wait-set块, 依据copyop / fixpipeop合并
    MergeWaitSetRegions(regions, mergedRegions);

    // 扩展if包裹的op范围
    // AIV、AIC处理有区别
    ExpandMergedRegionOps(forOp, mergedRegions);

    // 处理forop的末尾对于iter_arg的自增操作, 如tt.advance, 移进对应的if op
    MoveIterArgUsersIntoIf(forOp, mergedRegions);
    
    // 获取if yield的value, 并更新if内op的user为yield value
    for (MergedRegion &mr : mergedRegions) {
      ComputeYieldForMergedRegion(mr, body);
    }

    forOpList.push_back(forOp);
    regionList.push_back(mergedRegions);
  });

  for (size_t i = 0; i < forOpList.size(); ++i) {
    scf::ForOp oldForOp = forOpList[i];
    SmallVector<MergedRegion> newMergedRegions = regionList[i];

    // 找到所有的VV或CC依赖
    SmallVector<Value> dependValues;
    llvm::outs() << "FindDependValues! \n ";
    FindDependValues(dependValues, newMergedRegions);
    
    // 如果存在VV或CC依赖，更新ForOp添加新的对应args
    if (dependValues.size() != 0) {
      AddArgsForDependValues(oldForOp, dependValues, newMergedRegions, module);
    }
    
    // 创建最终的if op
    llvm::outs() << "before create if ops" << '\n';
    CreateIfOps(newMergedRegions, dependValues);
  }
}

void ChangeAdvanceOpForm(ModuleOp module) {
  module.walk([&](scf::ForOp forOp) {
    Block &body = forOp.getRegion().front();

    SmallVector<scf::IfOp, 4> ifOps;
    for (Operation &op : body)
      if (auto ifOp = dyn_cast<scf::IfOp>(&op))
        ifOps.push_back(ifOp);

    for (scf::IfOp ifOp : ifOps) {
      // 找 then region 中的 advance
      triton::AdvanceOp advanceOp;
      for (Operation &thenOp : ifOp.getThenRegion().front()) {
        if (auto adv = dyn_cast<triton::AdvanceOp>(thenOp)) {
          advanceOp = adv;
          break;
        }
      }
      if (!advanceOp) continue;

      // base 必须是 for的iter_arg
      Value base = advanceOp.getPtr();
      auto barg = dyn_cast<BlockArgument>(base);
      if (!barg || barg.getOwner() != &body) continue;

      // yield 去掉 advance 的返回值
      auto thenYield = cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto elseYield = cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

      int advanceIdx = -1;
      for (auto it : llvm::enumerate(thenYield.getOperands())) {
        if (it.value() == advanceOp.getResult()) {
          advanceIdx = it.index();
          break;
        }
      }
      
      if (advanceIdx == -1) continue;

      // 删除 advance
      SmallVector<Value> thenOps(thenYield.getOperands().begin(), thenYield.getOperands().end());
      SmallVector<Value> elseOps(elseYield.getOperands().begin(), elseYield.getOperands().end());

      thenOps.erase(thenOps.begin() + advanceIdx);
      elseOps.erase(elseOps.begin() + advanceIdx);

      thenYield->setOperands(thenOps);
      elseYield->setOperands(elseOps);
      
      // 重建 ifOp（去掉 advance 对应的 result）
      OpBuilder ifBuilder(ifOp);
      ifBuilder.setInsertionPoint(ifOp);

      // 构造新的 result types
      SmallVector<Type> newResultTypes;
      for (int i = 0; i < ifOp.getNumResults(); ++i) {
        if (i != advanceIdx)
          newResultTypes.push_back(ifOp.getResult(i).getType());
      }

      // 创建新的 if
      auto newIf = ifBuilder.create<scf::IfOp>(
          ifOp.getLoc(),
          newResultTypes,
          ifOp.getCondition(),
          /*withElseRegion=*/true);

      // 把已经修改过 yield 的 region 搬过去
      newIf.getThenRegion().takeBody(ifOp.getThenRegion());
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());

      // 替换if result的user
      int newIdx = 0;
      for (int oldIdx = 0; oldIdx < ifOp.getNumResults(); ++oldIdx) {
        if (oldIdx == advanceIdx)
          continue;
        ifOp.getResult(oldIdx).replaceAllUsesWith(newIf.getResult(newIdx++));
      }

      OpBuilder builder(newIf);
      builder.setInsertionPointAfter(newIf);

      Value flag = newIf.getCondition();

      SmallVector<Value, 4> newOffsets;
      for (Value off : advanceOp.getOffsets()) {
        auto intTy = cast<IntegerType>(off.getType());
        auto zero = builder.create<arith::ConstantIntOp>(
            newIf.getLoc(), 0, intTy.getWidth());
        auto sel = builder.create<arith::SelectOp>(
            newIf.getLoc(), flag, off, zero);
        newOffsets.push_back(sel);
      }

      auto newAdvance = builder.create<triton::AdvanceOp>(
        newIf.getLoc(), base.getType(), base, newOffsets);

      // 原 if 的 advance result 的 users，接到 newAdvance
      ifOp.getResult(advanceIdx).replaceAllUsesWith(newAdvance.getResult());

      // 删除旧的ifOp和advance
      advanceOp.erase();
      ifOp.erase();
    }
  });
}

void processRedudantIf(ModuleOp module) {
    SmallVector<scf::ForOp> forOps;
    llvm::outs()<<module<<" wwwww\n\n\n";
    module.walk([&](scf::ForOp forOp) {
        auto initArgs = forOp.getInitArgs();
        if (initArgs.size() == 5)
        {
            forOps.push_back(forOp);
        }  
    });

    for (auto forOp : forOps) {
        auto initArgs = forOp.getInitArgs();
        Value newInit = initArgs[2];

        // 构建新的初始化参数列表
        SmallVector<Value> newInitArgs(initArgs.begin(), initArgs.end());
        newInitArgs.push_back(newInit);

        // 获取原循环的边界和步长
        Value lb = forOp.getLowerBound();
        Value ub = forOp.getUpperBound();
        Value step = forOp.getStep();

        // 创建新的 ForOp，插入点位于原操作之前
        OpBuilder builder(forOp);
        auto newForOp = builder.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, newInitArgs);

        // 获取新循环的 region 块（已自动包含循环索引和迭代参数）
        Block &newBlock = newForOp.getRegion().front();
        Block &oldBlock = forOp.getRegion().front();

        // 建立块参数的映射：原块参数 -> 新块参数（前6个对应）
        IRMapping mapper;
        for (unsigned i = 0; i < oldBlock.getNumArguments(); ++i) {
            mapper.map(oldBlock.getArgument(i), newBlock.getArgument(i));
        }
        // 将原循环体中的操作（不包括终结符）克隆到新块中
        builder.setInsertionPointToStart(&newBlock);
        for (auto &op : oldBlock) {
            auto newOp = builder.clone(op, mapper);
        }

        // 在新块中查找第一个 scf::IfOp（即原代码中的第一个 if）
        scf::IfOp firstIfOp = nullptr;
        for (auto &op : newBlock.getOperations()) {
            if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
                firstIfOp = ifOp;
                break;
            }
        }
        assert(firstIfOp && "Expected at least one if op in the loop body");

        // 修改第一个 if 的 else 分支的 yield 操作：
        // 将其第二个操作数（索引1）从原来的 %arg9 改为新迭代参数（新块参数索引6）
        Block &elseBlock = firstIfOp.getElseRegion().front();
        auto elseYield = cast<scf::YieldOp>(elseBlock.getTerminator());
        SmallVector<Value> newElseYieldOps(elseYield.getOperands());
        newElseYieldOps[1] = newBlock.getArgument(6); // 新迭代参数
        builder.setInsertionPoint(elseYield);
        builder.create<scf::YieldOp>(elseYield.getLoc(), newElseYieldOps);
        elseYield->erase();

        // 创建新的循环 yield 操作：原5个操作数 + 第一个 if 的第二个结果
        auto oldYield = cast<scf::YieldOp>(newBlock.getTerminator());
        SmallVector<Value> newYieldOps(oldYield.getOperands());
        newYieldOps.push_back(firstIfOp.getResult(1)); // 第一个 if 的第二个结果
        builder.setInsertionPointToEnd(&newBlock);
        builder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOps);
        oldYield.erase();

        // 将原 forOp 的所有使用替换为新 forOp 的前5个结果
        for (auto it : llvm::zip(forOp->getResults(), newForOp->getResults().take_front(5))) {
            std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
        }
    }
    for (auto forOp : forOps) {
        forOp.erase();
    }
}
// 针对依赖变量，对原本的for op增加double buffer相关的迭代参数
scf::ForOp addDoubleBuffForArgs(ModuleOp module, SmallVector<Value> uniqueDeps, int bufferNum) {
    mlir::OpBuilder builder(module.getContext());
    SmallVector<int64_t> depValueForIdxs;

    // ========== 找到scf.if所在的scf::ForOp ==========
    if (!isa<scf::ForOp>(uniqueDeps[0].getDefiningOp()->getParentOp())) {
        llvm::errs() << "Error: parent op of scf.if is not scf.for";
    }
    scf::ForOp forOp = dyn_cast<scf::ForOp>(uniqueDeps[0].getDefiningOp()->getParentOp());

    for(Value dependencyValue : uniqueDeps){
        // ========== 步骤1：验证目标Value是scf.if的返回值，并找到对应的scf::IfOp ==========
        Operation *ifOp = dependencyValue.getDefiningOp();
        if (!ifOp || !isa<scf::IfOp>(ifOp)) {
            llvm::errs() << "Error: 目标Value不是scf.if的返回值\n";
            return nullptr;
        }
        scf::IfOp targetIfOp = dyn_cast<scf::IfOp>(ifOp);
        
        // 确认当前Value是scf.if的第几个返回值
        int64_t depValueIdx = -1;
        for (auto [idx, result] : llvm::enumerate(targetIfOp.getResults())) {
            if (result == dependencyValue) {
                depValueIdx = idx;
                break;
            }
        }

        // ========== 步骤2：找到%38#2关联的scf.for迭代参数以及索引 ==========
        // %38#2对应scf.if else分支yield的第2个操作数 → 即%arg10
        Operation *elseYield = targetIfOp.elseYield();
        Value dependencyArg = elseYield->getOperand(depValueIdx); // depValueIdx=2，对应else yield的第2个参数

        int64_t depValueForIdx = -1;
        for (auto [idx, result] : llvm::enumerate(forOp.getRegionIterArgs())) {
            if (result == dependencyArg) {
                depValueForIdx = idx;
                break;
            }
        }
        depValueForIdxs.push_back(depValueForIdx);
        llvm::outs() << "depValueForIdx: " << depValueForIdx << '\n';
    }

    llvm::outs() << "oldFor: " << forOp << '\n';
    
    // 获取原始循环的信息
    Value originalLowerBound = forOp.getLowerBound();
    Value originalUpperBound = forOp.getUpperBound();
    Value originalStep = forOp.getStep();
    SmallVector<Value> originalInitArgs = forOp.getInitArgs();
    SmallVector<Value> iterArgs;
    for (auto arg : originalInitArgs) {
        iterArgs.push_back(arg);
    }
    auto yields = forOp.getBody()->getTerminator();

    // 创建计数器初始零值
    Value counterInit = nullptr;
    mlir::Operation* parentOp = forOp->getParentOp();
    mlir::Operation* scopeOp = nullptr;
    // 向上遍历查找scope.scope操作
    while (parentOp) {
        if (dyn_cast<scope::ScopeOp>(parentOp)) {
            scopeOp = parentOp;
            break;
        }
        parentOp = parentOp->getParentOp();
    }

    builder.setInsertionPoint(scopeOp);
    Location loc = forOp.getLoc();
    auto i32Type = builder.getI32Type();
    counterInit = builder.create<arith::ConstantIntOp>(loc, 0, i32Type);

    // 添加和depValueForIdxs相同的迭代参数和计数器
    for (int64_t idx : depValueForIdxs) {
        for (int i = 0; i < bufferNum - 1; i++) {
            iterArgs.push_back(originalInitArgs[idx]);
        }
        
        // 在迭代参数中添加计数器
        for (int i = 0; i < 2; i++) {
            iterArgs.push_back(counterInit);
        }
    }

    builder.setInsertionPoint(forOp);
    // 创建新的for循环
    auto newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(),
        originalLowerBound,
        originalUpperBound,
        originalStep,
        iterArgs);
    
    // 设置IR映射表，将旧循环的变量映射到新循环
    IRMapping mapper;
    
    // 映射迭代变量
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());
    
    // 映射迭代参数
    for (auto [oldArg, newArg] : 
         llvm::zip(forOp.getRegionIterArgs(), 
                  newForOp.getRegionIterArgs())) {
        mapper.map(oldArg, newArg);
    }
    
    SmallVector<Value> newArgs;
    for (int i = forOp.getRegionIterArgs().size(); i < newForOp.getRegionIterArgs().size(); i++) {
        newArgs.push_back(newForOp.getRegionIterArgs()[i]);
    }
    // 克隆循环体内容到新循环
    auto &newLoopBody = *newForOp.getBody();
    builder.setInsertionPointToStart(&newLoopBody);
    
    for (auto &op : forOp.getBody()->without_terminator()) {
        builder.clone(op, mapper);
    }
    
    // 克隆yield操作
    if (auto yieldOp = dyn_cast<scf::YieldOp>(yields)) {
        SmallVector<Value> newYieldOperands;
        for (auto operand : yieldOp.getOperands()) {
            newYieldOperands.push_back(mapper.lookupOrDefault(operand));
        }
        // 将新增的迭代参数添加到yield操作数中
        for (auto currentCounter : newArgs) {                
            newYieldOperands.push_back(currentCounter);
        }
        builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    }
    
    // 替换原循环的结果
    unsigned numOriginalResults = forOp.getNumResults();
    SmallVector<Value> originalResults;
    for (unsigned i = 0; i < numOriginalResults; i++) {
        originalResults.push_back(newForOp.getResult(i));
    }
    forOp.replaceAllUsesWith(originalResults);
    
    // 8. 删除原循环
    forOp.erase();

    llvm::outs() << "for op erased!\n";
    return newForOp;
}

void addDoubleBuffCaculate(ModuleOp module, SmallVector<Value> newUniqueDeps, scf::ForOp &newForOp, int bufferNum) {
    mlir::OpBuilder builder(module.getContext());
    int cnt = 0;

    for (auto [depValueIdx, depValue] : llvm::enumerate(newUniqueDeps)) { 
        // ========== 找到depValue是scf.if的第几个返回值 ==========
        Operation *ifOp = depValue.getDefiningOp();
        if (!ifOp || !isa<scf::IfOp>(ifOp)) {
            llvm::outs() << "Error: 目标Value不是scf.if的返回值\n";
            break;
        }

        // 此为产生该depValue的IfOp
        scf::IfOp frontIfOp = dyn_cast<scf::IfOp>(ifOp);
        
        int64_t depValueIfIdx = -1;
        for (auto [idx, result] : llvm::enumerate(frontIfOp.getResults())) {
            if (result == depValue) {
                depValueIfIdx = idx;
                break;
            }
        }  

        // 找到对应yield值的定义OP
        Value depYieldValue = frontIfOp.thenYield()->getOperand(depValueIfIdx);
        Operation* depDefineOp = depYieldValue.getDefiningOp();

        // 找到ForOp iterArgs中的multi buffer和计数器
        int64_t extraArgBaseIdx = newForOp.getRegionIterArgs().size() - (((2+bufferNum-1) * newUniqueDeps.size() - (cnt++)));

        llvm::outs() << "extraArgBaseIdx: " << extraArgBaseIdx << "\n";
        Value buff0 = frontIfOp.elseYield()->getOperand(depValueIfIdx);
        Value buff1 = newForOp.getRegionIterArgs()[extraArgBaseIdx];
        Value frontCnt = newForOp.getRegionIterArgs()[extraArgBaseIdx + 1];
        Value postCnt = newForOp.getRegionIterArgs()[extraArgBaseIdx + 2];

        // ==================== 准备 ====================
        OpBuilder::InsertionGuard guard(builder);
        auto ifLoc = frontIfOp.getLoc();
        auto cond = frontIfOp.getCondition();

        // 原 then/else block
        auto &oldThenBlock = frontIfOp.getThenRegion().front();
        auto &oldElseBlock = frontIfOp.getElseRegion().front();

        // 原结果类型 + 新结果
        SmallVector<Type> newResultTypes(frontIfOp.getResultTypes().begin(),
                                 frontIfOp.getResultTypes().end());

        // 用已知类型，不用 select2Op/addOp
        newResultTypes.push_back(buff1.getType());
        newResultTypes.push_back(frontCnt.getType());
        auto oldNumResults = frontIfOp.getNumResults();

        // ==================== 创建新的 IfOp ====================
        builder.setInsertionPoint(frontIfOp);
        auto newIfOp = builder.create<scf::IfOp>(ifLoc, newResultTypes, cond, /*hasElse=*/true);
        // ==================== then region ====================
        int indexBuffer1, indexBuffer0, frontCntIndex;
        {
            mlir::IRMapping mapping;
            auto &newThenBlock = newIfOp.getThenRegion().front();
            builder.setInsertionPointToStart(&newThenBlock);
            // clone 原 then body
            for (auto &op : oldThenBlock.without_terminator()) {
                builder.clone(op, mapping);
            }

            Value newDepVal = depYieldValue;
            if (mapping.contains(depYieldValue))
                newDepVal = mapping.lookup(depYieldValue);
            builder.setInsertionPointAfter(newDepVal.getDefiningOp());
            Location loc = ifLoc;
            auto i32Type = builder.getI32Type();

            auto c0 = builder.create<arith::ConstantOp>(loc, i32Type,
                                                        builder.getI32IntegerAttr(0));
            auto c1 = builder.create<arith::ConstantOp>(loc, i32Type,
                                                        builder.getI32IntegerAttr(1));
            auto c2 = builder.create<arith::ConstantOp>(loc, i32Type,
                                                        builder.getI32IntegerAttr(2));

            auto remOp = builder.create<arith::RemSIOp>(loc, frontCnt, c2);
            auto cmpOp = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, remOp, c0);

            auto select1Op = builder.create<arith::SelectOp>(
                loc, cmpOp, newDepVal, buff0);

            auto select2Op = builder.create<arith::SelectOp>(
                loc, cmpOp, buff1, newDepVal);

            auto addOp = builder.create<arith::AddIOp>(loc, frontCnt, c1);

            SmallVector<Value> thenOperands;
            for (auto v : oldThenBlock.getTerminator()->getOperands()) {
                if(mapping.lookupOrDefault(v) == newDepVal){
                    indexBuffer0 = thenOperands.size();
                    thenOperands.push_back(select1Op);
                }else {
                    thenOperands.push_back(mapping.lookupOrDefault(v));
                }
            }
            indexBuffer1 = thenOperands.size();
            thenOperands.push_back(select2Op);
            frontCntIndex = thenOperands.size();
            thenOperands.push_back(addOp);
            builder.setInsertionPointToEnd(&newThenBlock);
            builder.create<scf::YieldOp>(loc, thenOperands);
        }


        // ==================== else region ====================
        {
            mlir::IRMapping mapping;
            auto &newElseBlock = newIfOp.getElseRegion().front();
            builder.setInsertionPointToStart(&newElseBlock);
            for (auto &op : oldElseBlock.without_terminator()) {
                builder.clone(op, mapping);
            }
            builder.setInsertionPointToEnd(&newElseBlock);
            SmallVector<Value> elseOperands;
            for (auto v : oldElseBlock.getTerminator()->getOperands()) {
                elseOperands.push_back(mapping.lookupOrDefault(v));
            }
            elseOperands.push_back(buff1);
            elseOperands.push_back(frontCnt);
            builder.create<scf::YieldOp>(ifLoc, elseOperands);
        }
        // 只替换原来数量的 result
        frontIfOp.replaceAllUsesWith(
            newIfOp.getResults().take_front(oldNumResults));
        // needToDel.push_back(frontIfOp);
        frontIfOp.erase();

        auto newDepValue = newIfOp.getResult(depValueIfIdx);
        // ------------------ 安全的第二个 If 修改方案 ------------------
        scf::IfOp postIfOp = nullptr;
        for (auto &use : newDepValue.getUses()) {
            if (auto candidateIf = dyn_cast<scf::IfOp>(use.getOwner()->getParentOp())) {
                postIfOp = candidateIf;
                break;
            }
        }

        if (!postIfOp) {
            llvm::outs() << "Error: no consuming IfOp found for depValue.\n";
            return;
        }

        ifLoc = postIfOp.getLoc();
        cond = postIfOp.getCondition();
        // 原 then/else block
        auto &oldThenBlockV2 = postIfOp.getThenRegion().front();
        auto &oldElseBlockV2 = postIfOp.getElseRegion().front();
        SmallVector<Type> newResultTypesv2(postIfOp.getResultTypes().begin(),
                            postIfOp.getResultTypes().end());
        newResultTypesv2.push_back(postCnt.getType());
        builder.setInsertionPoint(postIfOp);
        auto newIfOpv2 = builder.create<scf::IfOp>(ifLoc, newResultTypesv2, cond, /*hasElse=*/true);
        
        mlir::IRMapping mappingv2;
        auto &newThenBlockv2 = newIfOpv2.getThenRegion().front();
        builder.setInsertionPointToStart(&newThenBlockv2);
        // clone 原 then body
        for (auto &op : oldThenBlockV2.without_terminator()) {
            builder.clone(op, mappingv2);
        }
        builder.setInsertionPointToStart(&newThenBlockv2);

        OpOperand *needToReplace; 
        for (auto &use : newDepValue.getUses()) {
            if(newIfOpv2 == dyn_cast<scf::IfOp>(use.getOwner()->getParentOp())){
                needToReplace = &use;
                break;
            }
        }
        
        // loc 和常量定义
        auto loc = postIfOp.getLoc();
        auto c0_i32 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        auto c1_i32 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
        auto c2_i32 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
        auto thenYieldOp = postIfOp.thenYield();
        // 使用 iter_args 中的 buffer 和 counter
        Value oldBuffer0 = newIfOp.getResult(indexBuffer0);
        Value oldBuffer1 = newIfOp.getResult(indexBuffer1);

        // 计算 double buffer 条件
        auto rem = builder.create<arith::RemSIOp>(loc, postCnt, c2_i32);
        auto cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rem, c0_i32);

        // select 操作：偶数轮用 oldBuffer0，奇数轮用 oldBuffer1
        auto selectOp = builder.create<arith::SelectOp>(loc, cmp, oldBuffer0, oldBuffer1);
        auto addOp = builder.create<arith::AddIOp>(loc, postCnt, c1_i32);
        needToReplace->set(selectOp);

        SmallVector<Value> newThenOperands;
        for (auto v : oldThenBlockV2.getTerminator()->getOperands()) {
            newThenOperands.push_back(mappingv2.lookupOrDefault(v));
        }
        auto postCntIndex = newThenOperands.size();
        newThenOperands.push_back(addOp);
        builder.setInsertionPointToEnd(&newThenBlockv2);
        builder.create<scf::YieldOp>(loc, newThenOperands);

        // ==================== 修改 Else Yield ====================
        auto &newElseBlockv2 = newIfOpv2.getElseRegion().front();
        for (auto &op : oldElseBlockV2.without_terminator()) {
                builder.clone(op, mappingv2);
            }

        builder.setInsertionPointToEnd(&newElseBlockv2);

        SmallVector<Value> elseOperandsv2;
        for (auto v : oldElseBlockV2.getTerminator()->getOperands()) {
            elseOperandsv2.push_back(mappingv2.lookupOrDefault(v));
        }

        elseOperandsv2.push_back(postCnt);

        builder.create<scf::YieldOp>(loc, elseOperandsv2);
        auto oldNumResultsv2 = postIfOp.getNumResults();

        postIfOp.replaceAllUsesWith(
            newIfOpv2.getResults().take_front(oldNumResultsv2));
        postIfOp.erase();

        // 替换newfor的yeild
        OpOperand *buff1YeildUse;
        for(auto &use : buff1.getUses()) {
            if (isa<scf::YieldOp>(use.getOwner()) && (newForOp == use.getOwner()->getParentOp())) {
                buff1YeildUse  = &use;
                break;
            }
        }
        buff1YeildUse->set(newIfOp.getResult(indexBuffer1));

        OpOperand *frontCntYeildUse;
        for(auto &use : frontCnt.getUses()) {
            if (isa<scf::YieldOp>(use.getOwner()) && (newForOp == use.getOwner()->getParentOp())) {
                frontCntYeildUse  = &use;
                break;
            }
        }
        frontCntYeildUse->set(newIfOp.getResult(frontCntIndex));
        
        OpOperand *postCntYeildUse;
        for(auto &use : postCnt.getUses()) {
            if (isa<scf::YieldOp>(use.getOwner()) && (newForOp == use.getOwner()->getParentOp())) {
                llvm::outs()<<"222use.getOwner()="<<*(use.getOwner())<<"\n";
                postCntYeildUse  = &use;
                break;
            }
        }
        postCntYeildUse->set(newIfOpv2.getResult(postCntIndex));


    }

}
// 处理当前 IfOp 与前一个 IfOp 的依赖关系
SmallVector<Value> ProcessIfOpWithDeps(scf::IfOp curIf, scf::IfOp prevIf,
                         DenseMap<scf::IfOp, SmallVector<Value>> &ifResultDeps) {
    SmallVector<Value> deps;
    llvm::outs()<<"ifResultDeps.size()= "<<ifResultDeps.size()<<"\n";

    auto checkRegion = [&](Region &region) {
        for (auto &block : region) {
            for (auto &op : block) {
                for (Value operand : op.getOperands()) {
                    for (Value res : prevIf.getResults()) {
                        if (operand == res) {
                            deps.push_back(res);
                        }
                    }
                }
            }
        }
    };

    // 只检查 thenRegion，不考虑 elseRegion
    checkRegion(curIf.getThenRegion());
    // checkRegion(curIf.getElseRegion());

    llvm::outs()<<"deps.size()="<<deps.size()<<"\n";
    if (deps.empty())
        return {};


    // 去重
    SmallPtrSet<Value, 4> depsSet(deps.begin(), deps.end());
    SmallVector<Value> uniqueDeps(depsSet.begin(), depsSet.end());
    ifResultDeps[curIf] = uniqueDeps;

    return uniqueDeps;

}

// 遍历单个 ForOp 内的 IfOp
SmallVector<Value> WalkForOpsAndProcessIfOnForOp(scf::ForOp forOp,
                                   DenseMap<scf::IfOp, SmallVector<Value>> &ifResultDeps) {
    SmallVector<Value> allDeps;                                
    scf::IfOp prevIf = nullptr;
    forOp.walk([&](scf::IfOp curIf) {
        if (prevIf) {
            llvm::outs() << " WalkForOpsAndProcessIf times: ";
            // ProcessIfOpWithDeps(curIf, prevIf, ifResultDeps);
            SmallVector<Value> deps = ProcessIfOpWithDeps(curIf, prevIf, ifResultDeps);
            if(!deps.empty())
                allDeps.append(deps.begin(), deps.end());
        }
        prevIf = curIf;
    });
    SmallPtrSet<Value, 8> uniqueSet(allDeps.begin(), allDeps.end());
    SmallVector<Value> uniqueDeps(uniqueSet.begin(), uniqueSet.end());
    // llvm::outs() << "uniqueDeps: \n" << uniqueDeps << '\n';
    return uniqueDeps;
}

bool isCube(scope::ScopeOp scope) {
    bool ret = false;
    scope.walk([&](Operation* op){
        if(isa<triton::DotOp>(op)){
            ret = true;
        }
    });
    return ret;
}

// 遍历每个 Cube scope，找到外层 ForOp 并处理内部 IfOp
void WalkAIVNestedForAndProcess(ModuleOp module,
                                DenseMap<scf::IfOp, SmallVector<Value>> &ifResultDeps) {
    llvm::outs() << " WalkAIVNestedForAndProcess times: \n";
    module.walk([&](scope::ScopeOp scope) {
        llvm::outs() << "Come in Scope: \n";
        if (isCube(scope)) return;

        // 遍历 Cube scope 内的 ForOp（外层循环）
        SmallVector<scf::ForOp> targetFors;
        scope.walk([&](scf::ForOp outerFor) {
            // 新增判断：只收集最内层 ForOp
            bool hasInnerFor = false;
            outerFor.walk([&](scf::ForOp inner) {
                if (inner != outerFor)
                    hasInnerFor = true;
            });
            if (!hasInnerFor)
                targetFors.push_back(outerFor);
        });
        llvm::outs()<<"targetFors:\n"<<targetFors.size();

        for (auto outerFor : targetFors) {
            ifResultDeps.clear();
            llvm::outs() << " scope.walk times: ";
            auto uniqueDeps = WalkForOpsAndProcessIfOnForOp(outerFor, ifResultDeps);
            auto newForOp = addDoubleBuffForArgs(module, uniqueDeps, 2);
            DenseMap<scf::IfOp, SmallVector<Value>> newIfResultDeps;
            auto uniqueList = WalkForOpsAndProcessIfOnForOp(newForOp, newIfResultDeps);
            addDoubleBuffCaculate(module , uniqueList, newForOp, 2);
        }
            
    });

}
void DAGSSBufferPass::runOnOperation() {
  auto module = getOperation();
  
  llvm::outs()<<module<<"  before ssbuffer\n\n";
  llvm::outs() <<"ModuleOp: "<< *module << "\n";

  // ControlSsbuf(module);
  // cv同步控制流
  AddIfCondition(module);

  FlowSssbuf(module);

  ControlSsbufV2(module);

  // advance不能出现在if里, 规避处理
  ChangeAdvanceOpForm(module);
//   llvm::outs()<<module<<"  after ssbuffer\n\n";
//   llvm::outs()<<module<<"  after ssbuffer\n\n";
//   llvm::outs()<<module<<"  after ssbuffer\n\n";
//   llvm::outs()<<module<<"  after ssbuffer\n\n";
//   llvm::outs() <<"ModuleOp: "<< *module << "\n";
//   processRedudantIf(module);
//   llvm::outs()<<module<<"  after if\n\n";
//   llvm::outs()<<module<<"  after if\n\n";
//   llvm::outs()<<module<<"  after if\n\n";


  DenseMap<scf::IfOp, SmallVector<Value>> ifResultDeps;
  WalkAIVNestedForAndProcess(module, ifResultDeps);
  
  llvm::outs()<<module<<"  after double ssbuffer\n\n";
  return;
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createDAGSSBufferPass() {
  return std::make_unique<DAGSSBufferPass>();
}