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

#include "TritonLinearize/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

Value materializeValue(OpBuilder &builder, Location loc, OpFoldResult ofr) {
  if (auto val = ofr.dyn_cast<Value>()) { 
    return val; 
  }
  
  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(intVal.value()));
  }
  assert(intVal.has_value());
  return Value();

  // return builder.create<arith::ConstantIndexOp>(
  //     loc, dyn_cast<IntegerAttr>(attr).getInt());
}  

std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (ofr.is<Attribute>() && isa<IntegerAttr>(ofr.get<Attribute>()))
    return dyn_cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
  

  return std::nullopt;
}

bool hasConstZero(const OpFoldResult ofr) {
  auto intAttr = getIntAttr(ofr);
  if (intAttr.has_value()) {
    if (intAttr.value() == 0) {
      return true;
    }
    return false;
  }

  auto val = dyn_cast<Value>(ofr);
  assert(val);
  auto constOp = val.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;

  intAttr = getIntAttr(constOp.getValue());
  if (intAttr.has_value()) {
    if (intAttr.value() == 0) {
      return true;
    }
    return false;
  }

  return false;
}

Value ofrToIndexValue(const OpFoldResult ofr, const Location loc,
                      OpBuilder &b) {
  if (Value val = dyn_cast<Value>(ofr)) {
    assert(val.getType().isIntOrIndex());
    if (!val.getType().isIndex()) {
      val = b.create<arith::IndexCastOp>(loc, b.getIndexType(), val);
    }
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return b.create<arith::ConstantOp>(loc, b.getIndexAttr(intVal.value()));
  }
  llvm_unreachable("Unexpected OpFoldResult state");
  return nullptr;
}

SmallVector<Value> ofrsToIndexValues(ArrayRef<OpFoldResult> ofrs,
                                     const Location loc, OpBuilder &b) {
  return llvm::to_vector<4>(
      llvm::map_range(ofrs, [&](OpFoldResult ofr) -> Value {
        return ofrToIndexValue(ofr, loc, b);
      }));
}

OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;
  if (!rhsIntAttr && lhsIntAttr && lhsIntAttr.value() == 0)
    return rhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() + rhsIntAttr.value());

  auto indextype = b.getIndexType();
  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  } else {
    if(isa<IntegerType>(lhsValue.getType())){
      lhsValue = b.create<arith::IndexCastOp>(loc, indextype, lhsValue).getResult();
    }
    assert(isa<IndexType>(lhsValue.getType()));
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  } else {
    if(isa<IntegerType>(rhsValue.getType())){
      rhsValue = b.create<arith::IndexCastOp>(loc, indextype, rhsValue).getResult();
    }
    assert(isa<IndexType>(rhsValue.getType()));
  }

  return b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() - rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto sumOp = b.create<arith::SubIOp>(loc, lhsValue, rhsValue);
  return sumOp.getResult();
}

OpFoldResult mulOFRValue(const OpFoldResult lhs, const Value rhs,
                         const Location loc, OpBuilder &b) {
   auto lhsIntAttr = getIntAttr(lhs);

  auto rhsIsConst = false;
  // if rhs is not a const, use max value since min is used to represent
  // dynamic size or stride
  auto rhsConstValue = std::numeric_limits<int64_t>::max();
  auto rhsOp = rhs.getDefiningOp<arith::ConstantOp>();
  if (rhsOp) {
    rhsIsConst = true;
    rhsConstValue = cast<IntegerAttr>(rhsOp.getValue()).getInt();
  }

  // 0. both lhs and rhs are constants
  if (lhsIntAttr && rhsIsConst)
    return b.getIndexAttr(lhsIntAttr.value() * rhsConstValue);

  // shortcuts for special cases
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
    }
    if (rhsIsConst) {
    if (rhsConstValue == 0)
      return rhsOp.getResult();
    if (rhsConstValue == 1)
      return lhs;
  }


  // 1. if lhs is constant but rhs is not
  if (lhsIntAttr && !rhsIsConst) {
    auto lhsConstOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    auto mulOp = b.create<arith::MulIOp>(loc, lhsConstOp.getResult(), rhs);
    return mulOp.getResult();
  }

  // 2. if lhs is not constant
  assert(!lhsIntAttr);
  auto mulOp = b.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs);
  return mulOp.getResult();
}

OpFoldResult divOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

   // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() / rhsIntAttr.value());

  // shortcut for special cases
  if (rhsIntAttr && rhsIntAttr.value() == 1)
    return lhs;

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = lhs.dyn_cast<Value>();
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = rhs.dyn_cast<Value>();
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto divOp = b.create<arith::DivSIOp>(loc, lhsValue, rhsValue);
  return divOp.getResult();
}

OpFoldResult remOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

   // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() % rhsIntAttr.value());

  // shortcut for special cases
  if (rhsIntAttr && rhsIntAttr.value() == 1)
    return b.getIndexAttr(0);

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = lhs.dyn_cast<Value>();
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = rhs.dyn_cast<Value>();
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto remOp = b.create<arith::RemSIOp>(loc, lhsValue, rhsValue);
  return remOp.getResult();
}

OpFoldResult mulOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() * rhsIntAttr.value());
  
  // shortcuts for special cases
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
  }
  if (rhsIntAttr) {
    if (rhsIntAttr.value() == 0)
      return rhs;
    if (rhsIntAttr.value() == 1)
      return lhs;
  }


  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto mulOp = b.create<arith::MulIOp>(loc, lhsValue, rhsValue);
  return mulOp.getResult();
}

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::min(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto minOp = b.create<arith::MinSIOp>(loc, lhsValue, rhsValue);
  return minOp.getResult();
}

OpFoldResult maxOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::max(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto maxOp = b.create<arith::MaxSIOp>(loc, lhsValue, rhsValue);
  return maxOp.getResult();
}

} // namespace mlir
