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

#ifndef TRITON_LINERIZE_ANALYSISSTRUCTURED_PTRANALYSIS_H
#define TRITON_LINERIZE_ANALYSISSTRUCTURED_PTRANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include <cstddef>
#include <set>

namespace mlir {

class OpBuilder;

namespace triton {

const extern std::string ptrAnalysisAttr;

// Data structure used to decode pointer arithmetics. offsets, sizes, and
// strides are in unit of elements in a linearly laid-out memory, which is the
// same as pointer arithmetic operations in Triton language. scalar is a
// shortcut used when the entire state describes a single scalar value. source
// is the base pointer. If order is present, PtrState describes block pointer;
// otherwise it describes non-block pointers. When it describes block pointer,
// shape field means the same field as tt.make_tensor_ptr; when it describes a
// non-block pointer, shape field indicates how address wraps around (i.e.,
// modulo); a constant 0 indicates no modulo for the dimension.

struct StateInfo{
  OpFoldResult offset; // final offset with base address
  OpFoldResult stride;
  OpFoldResult shape; // rem value
  OpFoldResult mask; //  div value
  Value dimVar;  // new dim var (for div/rem) 
  OpFoldResult dimOffset; // offset for new dimVar 
  size_t dim;
  bool remIsBeforeDiv = false;

  StateInfo() : dim(0) {}
  StateInfo(OpFoldResult offset, OpFoldResult stride, 
            OpFoldResult shape, OpFoldResult mask, size_t dim = 0)
        : offset(offset), stride(stride), shape(shape), mask(mask), dim(dim) {}
  void dump() const;
  
  bool hasModulo() const {
      auto intAttr = getIntAttr(shape);
      if (!intAttr.has_value()) {
          return false;
      }
      return intAttr.value() != 0;
  };
  bool hasDivision() const {
      auto intAttr = getIntAttr(mask);
      if (!intAttr.has_value()) {
          return false;
      }
      return intAttr.value() != 0;
  };

};

// Group StateInfo by original dimension
struct StateInfoGroup {
  size_t dim; // original dim
  SmallVector<size_t> idxes; // indexes of stateInfo belonging to this dim (in original order)
  int64_t minStride = std::numeric_limits<int64_t>::max(); // The smallest stride within this group

  void dump() const;
};

enum class MemAccVal { Undefined = 0, StrucMemAcc = 1, UnstrucMemAcc = 2, Fallback = 3 };

struct MemAccType {

  MemAccVal value;

  explicit constexpr MemAccType(MemAccVal v = MemAccVal::Undefined)
      : value(v) {}

  constexpr operator MemAccVal() const { return value; }
  explicit operator bool() = delete;

  constexpr bool isUndefined() const { return value == MemAccVal::Undefined; }
  constexpr bool isStructured() const {
    return value == MemAccVal::StrucMemAcc;
  }
  constexpr bool isUnstructured() const {
    return value == MemAccVal::UnstrucMemAcc;
  }
  constexpr bool isFallback() const {
    return value == MemAccVal::Fallback;
  }

  void merge(MemAccType &other) {
    this->value = (this->value > other.value) ? this->value : other.value;
  }

  std::string_view toString() const {
    static constexpr std::string_view names[] = {"Undefined", "StrucMemAcc",
                                                 "UnstrucMemAcc"};
    return names[static_cast<int>(value)];
  }
};


/*
limits:

sizes.size() can greater than stateInfo.size()
dimLenth.size() = sizes.size() + 1, which initialized in countDims() called by addPtrState(), only triggered when meeting addptrOp.
when sizes.size()==1 or stateInfo.size()==0 : (scalar || source) must be True.
*/
struct PtrState {

  SmallVector<StateInfo> stateInfo; // shape info when load , maintained with visitOps
  SmallVector<OpFoldResult> sizes; // original shape, maintained with visitOps
  Attribute blockedAttr; // mark if isa tt.make_block_ptr, maintained TODO: maybe unused


  SmallVector<int32_t> order; // maintained in makeTensorPtr and advance
  SmallVector<size_t> dimLenth; // maintained only in countDim()
  SmallVector<int32_t> permute; // used if hasPermute()
  SmallVector<StateInfoGroup> stateInfoGroups; // stateInfo grouped by stride descendingly, maintained in analyzePermute, sortStateByStride
  bool ptrIsTensor = true; // maintained in addPtrState and Splat // TODO refactor ptrIsTensor
  MemAccType memAccTy;

  Value source; // base address (ptr), maintained with visitOps
  Value scalar; // scalar offset (int), maintained with visitOps
  
  void dump() const ;
  int32_t getRank() const;
  bool isLegal() const;

  bool isSameSizeAs(const PtrState& x) const;

  bool shouldRemove(const StateInfo& x) const;

  bool opFoldResultIsZero(OpFoldResult op) const;

  bool isEmpty() const;

  bool hasModulo() const;

  bool hasDivision() const;

  bool hasPermute() const;

  bool hasBroadcast() const;

  bool dimHasDivision(uint32_t dim) const;

  bool dimHasModulo(uint32_t dim) const;

  bool isBlockPtr() const;

  bool countDims();

  void setMemAccTy(const MemAccType &);
  void setMemAccVal(const MemAccVal);
  MemAccType getMemAccType() const ;
  MemAccType &getMemAccTypeRef() ;
  void removeSource() { this->source = nullptr; };
  bool hasSource() const { return this->source != nullptr; }

  // This function transforms constant dimensions in a matrix into offsets, 
  // effectively removing those dimensions. It simplifies the matrix by 
  // eliminating unnecessary constant dimensions.
  LogicalResult removeConstDim(SmallVector<StateInfo> &InfoPerDim,
                               Operation *op, OpBuilder &builder);

  // Check whether the current read count is less than the batch size and 
  // broadcast the data along the highest dimension if needed.
  LogicalResult broadcastIfNeeded(SmallVector<StateInfo> &InfoPerDim,
                                          Operation *op, OpBuilder &builder);

  // Process addition of two PtrStates.
  LogicalResult addState(const PtrState &lhsState, const PtrState &rhsState,
                         Operation *op, OpBuilder &builder);

  // Process multiplication of two PtrStates
  LogicalResult mulState(const PtrState &lhsState, const PtrState &rhsState,
                         Operation *op, OpBuilder &builder);

  LogicalResult subState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder);

  // Process addition of ptr and offset.
  LogicalResult addPtrState(const PtrState &lhsState, const PtrState &rhsState,
                            Operation *op, OpBuilder &builder);

  LogicalResult ExpandInfo(SmallVector<StateInfo> &InfoPerDim,
                                      Operation *op, OpBuilder &builder);

  // analyze Permute info.
  LogicalResult analyzePermute(SmallVector<StateInfo> &InfoPerDim,
                               Operation *op, OpBuilder &builder);

  // sort stateInfo by stride descendingly.
  LogicalResult sortStateByStride(SmallVector<StateInfo> &InfoPerDim,
                                  Operation *op, OpBuilder &builder);


  triton::AddPtrOp createAddPtrOp(OpBuilder &builder, Location loc);
};

class PtrAnalysis {
  // This function is internally used by getLoopIterArgPtrState and
  // getLoopResultPtrState to get the correct PtrState for either an iter-arg or
  // a loop's result.
  //
  // A PtrState of an scf.for's iter-arg is the same as its corresponding
  // init-arg, except that the strides and offsets have to point to the loop's
  // iter-args that were created to carry the offsets and strides.
  //
  // For instance, for a pointer with index i and rank 2, 4 additional args
  // starting at index i + 1 are created. The PtrState's strides and offsets
  // value of the pointer's iter-arg must point to these 4 additionally created
  // iter-args.
  //
  // A similar process is used for getting the PtrState of the loop's i'th
  // result: its strides and offsets have to point to the corresponding stride
  // and offset values returned by the loop.
  PtrState reconcileLoopPtrState(
      scf::ForOp forOp, size_t ptrArgIndex, const PtrState &state,
      llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal);

  PtrState reconcileWhilePtrState(
    scf::WhileOp whileOp, size_t argIndex, const PtrState &state,
    llvm::function_ref<Value(scf::WhileOp op, size_t)> getReplacementVal);

  DenseSet<Value> maybeStructuredArgs;

public:
  void initializeMaybeStructuredArgs(Operation *op);
  using IndexMapSet = std::map<int, std::set<int>>;
  IndexMapSet levelToBlockArgIndex;
  int level = 0;
  // AddptrOp result -> PtrState
  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  // AddptrOp result -> new AddPtrOp result
  IRMapping ptrMap;

  // Recursively parse a Value; call the corresponding
  // function based on the defining operation and argument type.
  LogicalResult visitOperand(Value operand, PtrState &state, const Location loc, OpBuilder &builder);

  // Operand is a result of an scf.for. Such cases occur when there are multiple
  // levels of nested loops where the results of the inner scf.for (pointer) are
  // yielded by the outer loop.
  LogicalResult visitOperandForOp(scf::ForOp forOp, Value operand,
                                  PtrState &state, const Location loc,
                                  OpBuilder &builder);

  LogicalResult visitOperandWhileOp(scf::WhileOp whileOp, Value operand,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder);

  // Operand is the result of arith.addi. Process both arguments and insert any
  // arith.addi instruction as needed.
  // Main assumptions:
  //  Only one of lhsState and rhsState has source field set
  //  Current PtrState should be empty
  // Expected result:
  //  source = lhsState.source ? lhsState.source : rhsState.source
  //  sizes[i] = lhsState.sizes[i] (which should match rhsState.sizes[i])
  //  offsets[i] = lhsState.offsets[i] + rhsState.offsets[i]
  //  strides[i] = lhsState.strides[i] + rhsState.strides[i]
  LogicalResult visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of arith.muli. Process both arguments and insert any
  // arith.muli instruction as needed.
  // Main assumptions:
  //  Neither lhsState nor rhsState has source field set
  //  Current PtrState should be empty
  //  Currently only support one of the operand is a scalar index
  // Expected result (scalar and tensorState represent the two operands):
  //  source = null
  //  sizes[i] = tensorState.sizes[i]
  //  offsets[i] = tensorState.offsets[i] * scalar
  //  strides[i] = tensorState.strides[i] * scalar
  LogicalResult visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                const Location loc, OpBuilder &builder);
  
  LogicalResult visitOperandSub(arith::SubIOp subOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder);

  LogicalResult visitOperandRem(arith::RemSIOp mulOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandDiv(arith::DivSIOp divOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of make_range.
  // Main assumptions:
  //  start, end, and shape are all statically known
  //  The output of make_range is 1-dimensional
  //  Does not check validity of inputs (e.g., stride > 0)
  // Expected result:
  //  source = null
  //  sizes[0] = shape[0]
  //  offset[0] = start
  //  strides[0] = ceiling( (end - start) / shape[0] )
  LogicalResult visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                      PtrState &state, Location loc,
                                      OpBuilder &builder);

  // Operand is the result of expand_dims
  // Main assumptions:
  //  Only 1 dimension changes for each invocation of reshape
  //  The changed dimension must have size of 1
  // Expected result:
  //  Insert a dimension of size 1, stride 0, and offset 0
  LogicalResult visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder);

  // Operand is the result of broadcast
  // Main assumptions:
  //  Rank of soure and result is the same
  // Expected result:
  //  Update sizes[i] only, no changes to other fields
  LogicalResult visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                      PtrState &state, const Location loc,
                                      OpBuilder &builder);

  // Operand is the result of splat
  // Main assumptions:
  //  Source is a scalar value (i.e., an integer or a pointer, not a tensor)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] = 0
  //  if source is an integer, offset[0] = scalar = source
  LogicalResult visitOperandSplat(triton::SplatOp splatOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  // Operand is the result of arith.constant that is a splat
  // Main assumptions:
  //  Source is a constant op that produces a constant dense tensor where all
  //  elements are the same (i.e.: a constant that is splatted)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] =
  //  splat value if i == 0, otherwise 0
  LogicalResult visitOperandConstSplat(arith::ConstantOp op, PtrState &state,
const Location loc, OpBuilder &builder);

  LogicalResult visitOperandExtSI(arith::ExtSIOp, PtrState &state,
                                       const Location loc, OpBuilder &builder);

  // Operand is the result of addptr.
  // Main assumptions:
  //  The ptr field should populate the source field
  //  ptr and offset fields should result in same rank
  // Expected result:
  //  The resulting state for ptr and offset wil be added
  LogicalResult visitOperandAddptr(triton::AddPtrOp addptrOp, PtrState &state,
                                   const Location loc, OpBuilder &builder);

  // Operand is the result of tt.make_tensor_ptr.
  // Expected result:
  //  Parse source pointer and grab results
  LogicalResult visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                          PtrState &state, const Location loc,
                                          OpBuilder &builder);

  template <typename OpType>
  LogicalResult visitOperandIndirectLoad(OpType op, PtrState &state, 
                                      const Location &loc,
                                      OpBuilder &builder) ;

  LogicalResult visitOperandMakeTensorDescOp(triton::MakeTensorDescOp tensorDescType,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) ;

  LogicalResult visitOperandDescriptorLoad(triton::DescriptorLoadOp descLoadOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder);

// Get the computed PtrState for the forOp's init-arg at the provided index.
  FailureOr<PtrState> getLoopInitArgPtrState(scf::ForOp forOp, size_t index);

  // Get the computed PtrState for the forOp's iter-arg at the provided index.
  FailureOr<PtrState> getLoopIterArgPtrState(scf::ForOp forOp, size_t index);

  // Get the computed PtrState for the forOp's result at the provided index.
  FailureOr<PtrState> getLoopResultPtrState(scf::ForOp forOp, size_t index);

  // After PtrAnalysis finishes, rewrite the GetStructuredStateOp by creating
  // the correct initialization ops for offsets and strides and passing them to
  // any loop's init-args.
  LogicalResult rewriteGetStructuredStateOp(tts::GetStructuredStateOp op);

  // Parse the state of AddPtrOp, insert any instruction needed to
  // calculate strides and offsets, build PtrState for this operand, and record
  // PtrState for knownPtrs.
  LogicalResult rewriteAddptrOp(triton::AddPtrOp op);


  // Parse the state of YieldOp, insert any instruction needed to calculate
  // strides and offsets, build PtrState for this operand, and record PtrState
  // in knownPtrs.
  LogicalResult
  rewriteYieldOp(scf::YieldOp op,
                 llvm::SmallDenseMap<int, PtrState> &knownPtrsFor);

  // Rewrite eligible tt.addptr in loop init args so loop can update the ; such
  // pointers over iterations. Insert any instruction needed to calculate
  // strides, offsets, and modulos.
  LogicalResult rewriteForOp(scf::ForOp op);
  LogicalResult rewriteWhileOp(scf::WhileOp op);
  FailureOr<PtrState> getWhileInitArgPtrState(scf::WhileOp whileOp, size_t index);
  FailureOr<PtrState> getWhileResultPtrState(scf::WhileOp whileOp, size_t index);
  FailureOr<PtrState> getWhileAfterArgPtrState(scf::WhileOp whileOp, size_t index);
  FailureOr<PtrState> getWhileBeforeArgPtrState(scf::WhileOp whileOp, size_t index);

  LogicalResult analysisSplat(Operation *op, OpBuilder &builder, Value &ptr, PtrState &ptrState);
  LogicalResult extractScalarFromLoadedTensor(Operation* op, OpBuilder &builder, 
                                                Value &loadResult, const Location loc);

  LogicalResult rewriteScalarLoadOp(triton::LoadOp op, OpBuilder &builder, 
                                                Value &loadResult, const Location loc);
  
  LogicalResult createBroadcast(Operation *op, SmallVector<int64_t> &loadShape,
                                            Value &loadResult);

  LogicalResult createReshape(Operation *op, Value &loadResult, SmallVector<int64_t> &srcShape);
  
  LogicalResult analyzeMask(Operation * op, PtrState &ptrState, triton_linearize::MaskState& maskState, 
                              SmallVector<OpFoldResult> &dims, SmallVector<int64_t> &dimMode);

  Value buildMaskValue(Value& ptr, PtrState& ptrState, triton_linearize::MaskState& maskState, OpBuilder& builder, Location loc);

  Value alignMaskForFallback(Value mask, ArrayRef<int64_t> targetShape, const PtrState &ptrState, OpBuilder &builder, Location loc);

  LogicalResult checkModifiedByAddPtrConverter(triton::LoadOp &op) const;
  LogicalResult rewriteLoadOp(triton::LoadOp op);

  LogicalResult rewriteStoreOp(triton::StoreOp op);

  LogicalResult rewriteOp(Operation *op);
};

} // namespace triton

} // namespace mlir

#endif
