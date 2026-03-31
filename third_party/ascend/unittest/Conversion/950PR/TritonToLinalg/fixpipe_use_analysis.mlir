// RUN: triton-opt -allow-unregistered-dialect '--triton-to-linalg=named-ops=True enable-nd2nz-on-vector=True compile-on-910-95=True' --split-input-file %s -verify-each 2>&1 | FileCheck %s --check-prefix=NOERR
// NOERR-NOT: failed to legalize unresolved materialization
// CHECK: module
// CHECK: func.func public @_hstu_attn_fwd

module {
  tt.func public @_hstu_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: f32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c256 = arith.constant 256 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0> : tensor<256x1xi64>
    %cst_0 = arith.constant dense<false> : tensor<256x32xi1>
    %cst_1 = arith.constant dense<0> : tensor<32x1xi64>
    %cst_2 = arith.constant dense<false> : tensor<32x32xi1>
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i64 = arith.constant 8 : i64
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i64 = arith.constant 2 : i64
    %c32_i64 = arith.constant 32 : i64
    %c256_i64 = arith.constant 256 : i64
    %c128_i64 = arith.constant 128 : i64
    %c255_i32 = arith.constant 255 : i32
    %c1_i64 = arith.constant 1 : i64
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<256x32xf16>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<32x32xf16>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<32x256xf32>
    %cst_6 = arith.constant dense<128> : tensor<256x1xi64>
    %cst_7 = arith.constant dense<256> : tensor<32x1xi64>
    %cst_8 = arith.constant dense<1.000000e+00> : tensor<32x256xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(32 : i64) : i64
    %2 = llvm.mlir.constant(64 : i64) : i64
    %3 = llvm.mlir.constant(96 : i64) : i64
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(2 : i64) : i64
    %7 = llvm.mlir.constant(2 : i32) : i32
    %8 = llvm.mlir.constant(4 : i32) : i32
    %9 = llvm.mlir.constant(6 : i32) : i32
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.mlir.constant(3 : i32) : i32
    %c64_i64 = arith.constant 64 : i64
    %12 = llvm.mlir.constant(5 : i32) : i32
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>
    %alloc_10 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
    %alloc_11 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
    %13 = tt.get_program_id x : i32
    %14 = tt.get_num_programs x : i32
    %15 = arith.cmpi sle, %arg10, %c32_i32 : i32
    %16 = scf.if %15 -> (i64) {
      scf.yield %c2_i64 : i64
    } else {
      %41 = tt.addptr %arg5, %c2_i32 : !tt.ptr<i32>, i32
      %42 = tt.load %41 : !tt.ptr<i32>
      %43 = arith.extsi %42 : i32 to i64
      scf.yield %43 : i64
    }
    %17 = arith.muli %16, %c8_i64 : i64
    %18 = arith.extsi %14 : i32 to i64
    %19 = arith.minsi %18, %17 : i64
    %20 = arith.divsi %17, %19 : i64
    %21 = arith.addi %20, %c1_i64 : i64
    %22 = arith.remsi %17, %19 : i64
    %23 = arith.extsi %13 : i32 to i64
    %24 = arith.cmpi slt, %23, %19 : i64
    %25 = arith.cmpi slt, %23, %22 : i64
    %26 = arith.muli %23, %21 : i64
    %27 = arith.muli %22, %21 : i64
    %28 = arith.subi %23, %22 : i64
    %29 = arith.muli %28, %20 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = arith.select %25, %26, %30 : i64
    %32 = arith.select %24, %31, %c0_i64 : i64
    %33 = arith.select %25, %21, %20 : i64
    %34 = arith.select %24, %33, %c0_i64 : i64
    %35 = arith.cmpi sge, %23, %19 : i64
    cf.cond_br %35, ^bb1, ^bb2
  ^bb1:  // 2 preds: ^bb0, ^bb2
    tt.return
  ^bb2:  // pred: ^bb0
    %36 = arith.cmpi sle, %34, %c0_i64 : i64
    cf.cond_br %36, ^bb1, ^bb3
  ^bb3:  // pred: ^bb2
    %37 = llvm.inttoptr %0 : i64 to !llvm.ptr<11>
    %38 = llvm.inttoptr %1 : i64 to !llvm.ptr<11>
    %39 = llvm.inttoptr %2 : i64 to !llvm.ptr<11>
    %40 = llvm.inttoptr %3 : i64 to !llvm.ptr<11>
    llvm.store %4, %37 : i32, !llvm.ptr<11>
    llvm.store %4, %38 : i32, !llvm.ptr<11>
    llvm.store %4, %39 : i32, !llvm.ptr<11>
    llvm.store %4, %40 : i32, !llvm.ptr<11>
    scope.scope : () -> () {
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
      %41 = hivm.hir.get_sub_block_idx -> i64
      %42 = arith.muli %41, %1 : i64
      %43 = arith.addi %42, %1 : i64
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
      %44 = arith.addi %arg11, %c255_i32 : i32
      %45 = arith.divsi %44, %c256_i32 : i32
      %46 = arith.extsi %45 : i32 to i64
      %47 = arith.muli %34, %46 : i64
      %48 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %49 = arith.extsi %48 : tensor<32xi32> to tensor<32xi64>
      %50 = tt.splat %arg8 : f32 -> tensor<32x256xf32>
      %51 = tt.splat %arg9 : f32 -> tensor<32x256xf32>
      %52 = arith.muli %47, %c2_i64 : i64
      %53 = arith.divsi %52, %6 : i64
      %54:5 = scf.for %arg13 = %c0_i64 to %52 step %c1_i64 iter_args(%arg14 = %c0_i64, %arg15 = %cst_1, %arg16 = %cst_2, %arg17 = %c0_i64, %arg18 = %c0_i64) -> (i64, tensor<32x1xi64>, tensor<32x32xi1>, i64, i64)  : i64 {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
        %55 = llvm.inttoptr %43 : i64 to !llvm.ptr<11>
        %56 = llvm.load %55 : !llvm.ptr<11> -> i32
        %57 = arith.andi %56, %5 : i32
        %58 = arith.cmpi eq, %57, %5 : i32
        %59 = arith.andi %56, %7 : i32
        %60 = arith.cmpi eq, %59, %c0_i32 : i32
        %61 = arith.andi %56, %8 : i32
        %62 = arith.cmpi eq, %61, %8 : i32
        %63 = arith.cmpi slt, %arg17, %53 : i64
        %64 = arith.andi %58, %60 : i1
        %65 = arith.andi %64, %63 : i1
        %66 = arith.cmpi slt, %arg18, %53 : i64
        %67 = arith.andi %62, %66 : i1
        %68:4 = scf.if %65 -> (i64, tensor<32x1xi64>, tensor<32x32xi1>, i64) {
          %70 = arith.divsi %arg13, %46 : i64
          %71 = arith.addi %32, %70 : i64
          %72 = arith.divsi %71, %16 : i64
          %73 = arith.remsi %71, %16 : i64
          %74:2 = scf.if %15 -> (i64, i64) {
            scf.yield %73, %c0_i64 : i64, i64
          } else {
            %108:2 = scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg20 = %c0_i32, %arg21 = %c3_i32) -> (i32, i32)  : i32 {
              %115 = arith.addi %arg20, %arg21 : i32
              %116 = arith.divsi %115, %c2_i32 : i32
              %117 = tt.addptr %arg5, %116 : !tt.ptr<i32>, i32
              %118 = tt.load %117 : !tt.ptr<i32>
              %119 = arith.extsi %118 : i32 to i64
              %120 = arith.cmpi sle, %119, %73 : i64
              %121 = arith.select %120, %arg21, %116 : i32
              %122 = scf.if %120 -> (i32) {
                %123 = arith.addi %116, %c1_i32 : i32
                scf.yield %123 : i32
              } else {
                scf.yield %arg20 : i32
              }
              scf.yield %122, %121 : i32, i32
            }
            %109 = arith.subi %108#0, %c1_i32 : i32
            %110 = arith.extsi %109 : i32 to i64
            %111 = tt.addptr %arg5, %110 : !tt.ptr<i32>, i64
            %112 = tt.load %111 : !tt.ptr<i32>
            %113 = arith.extsi %112 : i32 to i64
            %114 = arith.subi %73, %113 : i64
            scf.yield %110, %114 : i64, i64
          }
          %75 = tt.addptr %arg3, %74#0 : !tt.ptr<i64>, i64
          %76 = tt.load %75 : !tt.ptr<i64>
          %77 = tt.addptr %75, %c1_i32 : !tt.ptr<i64>, i32
          %78 = tt.load %77 : !tt.ptr<i64>
          %79 = arith.subi %78, %76 : i64
          %80 = arith.muli %72, %c32_i64 : i64
          %81 = arith.muli %76, %c256_i64 : i64
          %82 = arith.addi %80, %81 : i64
          %83 = arith.muli %74#1, %c32_i64 : i64
          %84 = tt.splat %83 : i64 -> tensor<32xi64>
          %85 = arith.addi %84, %49 : tensor<32xi64>
          %86 = tt.splat %79 : i64 -> tensor<32xi64>
          %87 = arith.cmpi slt, %85, %86 : tensor<32xi64>
          %88 = tt.expand_dims %85 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
          %89 = arith.muli %88, %cst_7 : tensor<32x1xi64>
          %90 = tt.expand_dims %87 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
          %91 = tt.broadcast %90 : tensor<32x1xi1> -> tensor<32x32xi1>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %memspacecast = memref.memory_space_cast %alloc_10 : memref<32x256xf32, #hivm.address_space<ub>> to memref<32x256xf32>
          %92 = bufferization.to_tensor %memspacecast restrict writable : memref<32x256xf32>
          %93 = arith.mulf %92, %50 : tensor<32x256xf32>
          %94 = arith.subf %cst_5, %93 : tensor<32x256xf32>
          %95 = math.exp %94 : tensor<32x256xf32>
          %96 = arith.addf %95, %cst_8 : tensor<32x256xf32>
          %97 = arith.divf %cst_8, %96 : tensor<32x256xf32>
          %98 = arith.mulf %93, %97 : tensor<32x256xf32>
          %99 = arith.mulf %98, %51 : tensor<32x256xf32>
          %100 = arith.truncf %99 : tensor<32x256xf32> to tensor<32x256xf16>
          %101 = tt.reshape %100 : tensor<32x256xf16> -> tensor<2x16x16x16xf16>
          %102 = tt.trans %101 {order = array<i32: 2, 0, 1, 3>} : tensor<2x16x16x16xf16> -> tensor<16x2x16x16xf16>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
          %103 = bufferization.to_memref %102 : memref<16x2x16x16xf16, #hivm.address_space<ub>>
          hivm.hir.copy ins(%103 : memref<16x2x16x16xf16, #hivm.address_space<ub>>) outs(%alloc : memref<16x2x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          %104 = llvm.load %55 : !llvm.ptr<11> -> i32
          %105 = arith.andi %104, %9 : i32
          %106 = arith.ori %105, %7 : i32
          llvm.store %106, %55 : i32, !llvm.ptr<11>
          %107 = arith.addi %arg17, %10 : i64
          scf.yield %82, %89, %91, %107 : i64, tensor<32x1xi64>, tensor<32x32xi1>, i64
        } else {
          scf.yield %arg14, %arg15, %arg16, %arg17 : i64, tensor<32x1xi64>, tensor<32x32xi1>, i64
        }
        %69 = scf.if %67 -> (i64) {
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast = memref.memory_space_cast %alloc_11 : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
          %70 = bufferization.to_tensor %memspacecast restrict writable : memref<32x32xf32>
          scf.for %arg19 = %c0 to %c32 step %c1 {
            scf.for %arg20 = %c0 to %c32 step %c1 {
              %extracted = tensor.extract %68#1[%arg19, %c0] {DiscreteMemAccess} : tensor<32x1xi64>
              %74 = arith.addi %68#0, %extracted : i64
              %75 = arith.index_cast %arg20 : index to i32
              %76 = arith.extsi %75 : i32 to i64
              %77 = arith.addi %74, %76 : i64
              %78 = tt.addptr %arg7, %77 : !tt.ptr<f16>, i64
              %extracted_12 = tensor.extract %70[%arg19, %arg20] {DiscreteMemAccess} : tensor<32x32xf32>
              %79 = arith.truncf %extracted_12 : f32 to f16
              %extracted_13 = tensor.extract %68#2[%arg19, %arg20] {DiscreteMemAccess} : tensor<32x32xi1>
              tt.store %78, %79, %extracted_13 {DiscreteMemAccess} : !tt.ptr<f16>
            } {ExtractedLoadOrStore}
          } {ExtractedLoadOrStore}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
          %71 = llvm.load %55 : !llvm.ptr<11> -> i32
          %72 = arith.andi %71, %11 : i32
          llvm.store %72, %55 : i32, !llvm.ptr<11>
          %73 = arith.addi %arg18, %10 : i64
          scf.yield %73 : i64
        } else {
          scf.yield %arg18 : i64
        }
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
        scf.yield %68#0, %68#1, %68#2, %68#3, %69 : i64, tensor<32x1xi64>, tensor<32x32xi1>, i64, i64
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      %41 = arith.addi %arg11, %c255_i32 : i32
      %42 = arith.divsi %41, %c256_i32 : i32
      %43 = arith.extsi %42 : i32 to i64
      %44 = arith.muli %34, %43 : i64
      %45 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %46 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %47 = arith.extsi %45 : tensor<32xi32> to tensor<32xi64>
      %48 = tt.expand_dims %45 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %49 = tt.broadcast %48 : tensor<1x32xi32> -> tensor<32x32xi32>
      %50 = arith.extsi %46 : tensor<256xi32> to tensor<256xi64>
      %51 = tt.broadcast %48 : tensor<1x32xi32> -> tensor<256x32xi32>
      %52 = arith.muli %44, %c2_i64 : i64
      %53 = arith.divsi %52, %6 : i64
      %54:5 = scf.for %arg13 = %c0_i64 to %52 step %c1_i64 iter_args(%arg14 = %c0_i64, %arg15 = %cst, %arg16 = %cst_0, %arg17 = %c0_i64, %arg18 = %c0_i64) -> (i64, tensor<256x1xi64>, tensor<256x32xi1>, i64, i64)  : i64 {
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
        %55 = llvm.inttoptr %c32_i64 : i64 to !llvm.ptr<11>
        %56 = llvm.inttoptr %c64_i64 : i64 to !llvm.ptr<11>
        %57 = llvm.load %55 : !llvm.ptr<11> -> i32
        %58 = llvm.load %56 : !llvm.ptr<11> -> i32
        %59 = arith.andi %57, %5 : i32
        %60 = arith.andi %58, %5 : i32
        %61 = arith.cmpi eq, %59, %c0_i32 : i32
        %62 = arith.cmpi eq, %60, %c0_i32 : i32
        %63 = arith.andi %61, %62 : i1
        %64 = arith.andi %57, %7 : i32
        %65 = arith.andi %58, %7 : i32
        %66 = arith.cmpi eq, %64, %7 : i32
        %67 = arith.cmpi eq, %65, %7 : i32
        %68 = arith.andi %66, %67 : i1
        %69 = arith.andi %57, %8 : i32
        %70 = arith.andi %58, %8 : i32
        %71 = arith.cmpi eq, %69, %c0_i32 : i32
        %72 = arith.cmpi eq, %70, %c0_i32 : i32
        %73 = arith.andi %71, %72 : i1
        %74 = arith.cmpi slt, %arg17, %53 : i64
        %75 = arith.andi %63, %74 : i1
        %76 = arith.cmpi slt, %arg18, %53 : i64
        %77 = arith.andi %68, %73 : i1
        %78 = arith.andi %77, %76 : i1
        %79:4 = scf.if %75 -> (i64, tensor<256x1xi64>, tensor<256x32xi1>, i64) {
          %81 = arith.divsi %arg13, %43 : i64
          %82 = arith.addi %32, %81 : i64
          %83 = arith.remsi %arg13, %43 : i64
          %84 = arith.divsi %82, %16 : i64
          %85 = arith.remsi %82, %16 : i64
          %86:2 = scf.if %15 -> (i64, i64) {
            scf.yield %85, %c0_i64 : i64, i64
          } else {
            %140:2 = scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg20 = %c0_i32, %arg21 = %c3_i32) -> (i32, i32)  : i32 {
              %147 = arith.addi %arg20, %arg21 : i32
              %148 = arith.divsi %147, %c2_i32 : i32
              %149 = tt.addptr %arg5, %148 : !tt.ptr<i32>, i32
              %150 = tt.load %149 : !tt.ptr<i32>
              %151 = arith.extsi %150 : i32 to i64
              %152 = arith.cmpi sle, %151, %85 : i64
              %153 = arith.select %152, %arg21, %148 : i32
              %154 = scf.if %152 -> (i32) {
                %155 = arith.addi %148, %c1_i32 : i32
                scf.yield %155 : i32
              } else {
                scf.yield %arg20 : i32
              }
              scf.yield %154, %153 : i32, i32
            }
            %141 = arith.subi %140#0, %c1_i32 : i32
            %142 = arith.extsi %141 : i32 to i64
            %143 = tt.addptr %arg5, %142 : !tt.ptr<i32>, i64
            %144 = tt.load %143 : !tt.ptr<i32>
            %145 = arith.extsi %144 : i32 to i64
            %146 = arith.subi %85, %145 : i64
            scf.yield %142, %146 : i64, i64
          }
          %87 = arith.divsi %84, %c2_i64 : i64
          %88 = tt.addptr %arg3, %86#0 : !tt.ptr<i64>, i64
          %89 = tt.load %88 : !tt.ptr<i64>
          %90 = tt.addptr %88, %c1_i32 : !tt.ptr<i64>, i32
          %91 = tt.load %90 : !tt.ptr<i64>
          %92 = tt.addptr %arg4, %86#0 : !tt.ptr<i64>, i64
          %93 = tt.load %92 : !tt.ptr<i64>
          %94 = tt.addptr %92, %c1_i32 : !tt.ptr<i64>, i32
          %95 = tt.load %94 : !tt.ptr<i64>
          %96 = arith.subi %91, %89 : i64
          %97 = arith.subi %95, %93 : i64
          %98 = arith.muli %84, %c32_i64 : i64
          %99 = arith.muli %89, %c256_i64 : i64
          %100 = arith.addi %98, %99 : i64
          %101 = tt.addptr %arg0, %100 : !tt.ptr<f16>, i64
          %102 = arith.muli %87, %c32_i64 : i64
          %103 = arith.muli %93, %c128_i64 : i64
          %104 = arith.addi %102, %103 : i64
          %105 = tt.addptr %arg1, %104 : !tt.ptr<f16>, i64
          %106 = arith.muli %83, %c256_i64 : i64
          %107 = arith.muli %86#1, %c32_i64 : i64
          %108 = tt.splat %107 : i64 -> tensor<32xi64>
          %109 = arith.addi %108, %47 : tensor<32xi64>
          %110 = tt.splat %96 : i64 -> tensor<32xi64>
          %111 = arith.cmpi slt, %109, %110 : tensor<32xi64>
          %112 = tt.expand_dims %109 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
          %113 = arith.muli %112, %cst_7 : tensor<32x1xi64>
          %114 = tt.splat %101 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
          %115 = tt.addptr %114, %113 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi64>
          %116 = tt.broadcast %115 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x32x!tt.ptr<f16>>
          %117 = tt.addptr %116, %49 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
          %118 = tt.expand_dims %111 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
          %119 = tt.broadcast %118 : tensor<32x1xi1> -> tensor<32x32xi1>
          %120 = tt.load %117, %119, %cst_4 : tensor<32x32x!tt.ptr<f16>>
          %121 = tt.splat %106 : i64 -> tensor<256xi64>
          %122 = arith.addi %121, %50 : tensor<256xi64>
          %123 = tt.splat %97 : i64 -> tensor<256xi64>
          %124 = arith.cmpi slt, %122, %123 : tensor<256xi64>
          %125 = tt.expand_dims %122 {axis = 1 : i32} : tensor<256xi64> -> tensor<256x1xi64>
          %126 = arith.muli %125, %cst_6 : tensor<256x1xi64>
          %127 = tt.splat %105 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>>
          %128 = tt.addptr %127, %126 : tensor<256x1x!tt.ptr<f16>>, tensor<256x1xi64>
          %129 = tt.broadcast %128 : tensor<256x1x!tt.ptr<f16>> -> tensor<256x32x!tt.ptr<f16>>
          %130 = tt.addptr %129, %51 : tensor<256x32x!tt.ptr<f16>>, tensor<256x32xi32>
          %131 = tt.expand_dims %124 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
          %132 = tt.broadcast %131 : tensor<256x1xi1> -> tensor<256x32xi1>
          %133 = tt.load %130, %132, %cst_3 : tensor<256x32x!tt.ptr<f16>>
          %134 = tt.trans %133 {order = array<i32: 1, 0>} : tensor<256x32xf16> -> tensor<32x256xf16>
          %135 = tt.dot %120, %134, %cst_5 : tensor<32x32xf16> * tensor<32x256xf16> -> tensor<32x256xf32>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%135 : tensor<32x256xf32>) outs(%alloc_10 : memref<32x256xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %136 = llvm.load %55 : !llvm.ptr<11> -> i32
          %137 = arith.ori %136, %5 : i32
          %138 = arith.ori %137, %5 : i32
          llvm.store %137, %55 : i32, !llvm.ptr<11>
          llvm.store %138, %56 : i32, !llvm.ptr<11>
          %139 = arith.addi %arg17, %10 : i64
          scf.yield %104, %126, %132, %139 : i64, tensor<256x1xi64>, tensor<256x32xi1>, i64
        } else {
          scf.yield %arg14, %arg15, %arg16, %arg17 : i64, tensor<256x1xi64>, tensor<256x32xi1>, i64
        }
        %80 = scf.if %78 -> (i64) {
          %81 = tensor.empty() : tensor<256x32xf16>
          %82 = scf.for %arg19 = %c0 to %c256 step %c1 iter_args(%arg20 = %81) -> (tensor<256x32xf16>) {
            %extracted = tensor.extract %79#1[%arg19, %c0] {DiscreteMemAccess} : tensor<256x1xi64>
            %92 = arith.addi %79#0, %extracted : i64
            %93 = tt.splat %92 : i64 -> tensor<1x32xi64>
            %94 = arith.extsi %48 : tensor<1x32xi32> to tensor<1x32xi64>
            %95 = arith.addi %93, %94 : tensor<1x32xi64>
            %96 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>>
            %97 = tt.addptr %96, %95 : tensor<1x32x!tt.ptr<f16>>, tensor<1x32xi64>
            %98 = tt.load %97 {DiscreteMemAccess} : tensor<1x32x!tt.ptr<f16>>
            %inserted_slice = tensor.insert_slice %98 into %arg20[%arg19, 0] [1, 32] [1, 1] : tensor<1x32xf16> into tensor<256x32xf16>
            scf.yield {DiscreteMemAccess} %inserted_slice : tensor<256x32xf16>
          } {ExtractedLoadOrStore}
          %83 = arith.select %79#2, %82, %cst_3 : tensor<256x32xi1>, tensor<256x32xf16>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          %84 = hivm.hir.convert_layout %alloc {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<16x2x16x16xf16, #hivm.address_space<cbuf>>) -> memref<32x256xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %84 : memref<32x256xf16, #hivm.address_space<cbuf>> to memref<32x256xf16>
          %85 = bufferization.to_tensor %memspacecast restrict writable : memref<32x256xf16>
          %86 = tt.dot %85, %83, %cst_9 : tensor<32x256xf16> * tensor<256x32xf16> -> tensor<32x32xf32>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%86 : tensor<32x32xf32>) outs(%alloc_11 : memref<32x32xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %87 = llvm.load %55 : !llvm.ptr<11> -> i32
          %88 = arith.andi %87, %12 : i32
          %89 = arith.ori %88, %8 : i32
          %90 = arith.ori %89, %8 : i32
          llvm.store %89, %55 : i32, !llvm.ptr<11>
          llvm.store %90, %56 : i32, !llvm.ptr<11>
          %91 = arith.addi %arg18, %10 : i64
          scf.yield %91 : i64
        } else {
          scf.yield %arg18 : i64
        }
        hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
        scf.yield %79#0, %79#1, %79#2, %79#3, %80 : i64, tensor<256x1xi64>, tensor<256x32xi1>, i64, i64
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    tt.return
  }
}
