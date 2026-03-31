// RUN: triton-opt -allow-unregistered-dialect '--triton-to-linalg=named-ops=True enable-nd2nz-on-vector=True compile-on-910-95=True' --split-input-file %s -verify-each 2>&1 | FileCheck %s --check-prefix=NOERR
// NOERR-NOT: failed to legalize unresolved materialization
// CHECK: module
// CHECK: func.func public @dsa_prefill_kernel

module {
  tt.func public @dsa_prefill_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: f32) attributes {noinline = false} {
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<16x128xbf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x192xbf16>
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<16xf32>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<16xf32>
    %cst_3 = arith.constant dense<9.99999996E-13> : tensor<16xf32>
    %cst_4 = arith.constant dense<0xFF800000> : tensor<16x16xf32>
    %cst_5 = arith.constant dense<0> : tensor<16x16xi8>
    %cst_6 = arith.constant dense<1024> : tensor<1x16xi32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_8 = arith.constant dense<1024> : tensor<16x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<16x128xf32>
    %cst_11 = arith.constant dense<false> : tensor<16x1xi1>
    %cst_12 = arith.constant dense<0> : tensor<16x1xi32>
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(32 : i64) : i64
    %2 = llvm.mlir.constant(64 : i64) : i64
    %3 = llvm.mlir.constant(96 : i64) : i64
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %c2_i32 = arith.constant 2 : i32
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.constant(4 : i32) : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c6_i32 = arith.constant 6 : i32
    %8 = llvm.mlir.constant(6 : i32) : i32
    %9 = llvm.mlir.constant(3 : i32) : i32
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %10 = llvm.mlir.constant(5 : i32) : i32
    %alloc = memref.alloc() : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>>
    %alloc_13 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    %alloc_14 = memref.alloc() : memref<16x128xf32, #hivm.address_space<ub>>
    %11 = tt.get_program_id x : i32
    %12 = llvm.inttoptr %0 : i64 to !llvm.ptr<11>
    %13 = llvm.inttoptr %1 : i64 to !llvm.ptr<11>
    %14 = llvm.inttoptr %2 : i64 to !llvm.ptr<11>
    %15 = llvm.inttoptr %3 : i64 to !llvm.ptr<11>
    llvm.store %4, %12 : i32, !llvm.ptr<11>
    llvm.store %4, %13 : i32, !llvm.ptr<11>
    llvm.store %4, %14 : i32, !llvm.ptr<11>
    llvm.store %4, %15 : i32, !llvm.ptr<11>
    scope.scope : () -> () {
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
      %16 = hivm.hir.get_sub_block_idx -> i64
      %17 = arith.muli %16, %1 : i64
      %18 = arith.addi %17, %1 : i64
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
      %19 = arith.divsi %11, %c16_i32 : i32
      %20 = arith.remsi %11, %c16_i32 : i32
      %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %22 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %23 = tt.splat %arg19 : f32 -> tensor<16x16xf32>
      %24 = arith.muli %19, %arg17 : i32
      %25 = tt.splat %arg18 : i32 -> tensor<16x1xi32>
      %26 = tt.splat %24 : i32 -> tensor<16x1xi32>
      %27 = tt.splat %arg4 : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>>
      %28 = arith.muli %19, %arg14 : i32
      %29 = arith.muli %20, %arg15 : i32
      %30 = arith.addi %28, %29 : i32
      %31 = tt.splat %arg16 : i32 -> tensor<16x1xi32>
      %32 = tt.splat %30 : i32 -> tensor<16x1xi32>
      %33 = tt.expand_dims %21 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %34 = tt.broadcast %33 : tensor<1x128xi32> -> tensor<16x128xi32>
      %35 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>>
      scf.for %arg20 = %c0_i32 to %c1024_i32 step %c16_i32  : i32 {
        %36 = tt.splat %arg20 : i32 -> tensor<16xi32>
        %37 = arith.addi %36, %22 : tensor<16xi32>
        %38 = tt.expand_dims %37 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
        %39 = arith.cmpi slt, %38, %cst_8 : tensor<16x1xi32>
        %40 = arith.addi %arg20, %c1_i32 : i32
        %41 = arith.muli %38, %25 : tensor<16x1xi32>
        %42 = arith.addi %26, %41 : tensor<16x1xi32>
        %43 = tt.broadcast %42 : tensor<16x1xi32> -> tensor<16x16xi32>
        %44 = tt.broadcast %39 : tensor<16x1xi1> -> tensor<16x16xi1>
        %45 = arith.muli %40, %c2_i32 : i32
        %46 = arith.divsi %45, %c16_i32 : i32
        %47 = arith.divsi %46, %6 : i32
        %48:20 = scf.for %arg21 = %c0_i32 to %45 step %c16_i32 iter_args(%arg22 = %cst_10, %arg23 = %cst_2, %arg24 = %cst_10, %arg25 = %cst_9, %arg26 = %c0_i32, %arg27 = %c0_i32, %arg28 = %cst_9, %arg29 = %cst_9, %arg30 = %cst_9, %arg31 = %cst_9, %arg32 = %cst_9, %arg33 = %c0_i32, %arg34 = %c0_i32, %arg35 = %cst_10, %arg36 = %cst_10, %arg37 = %cst_10, %arg38 = %cst_10, %arg39 = %cst_10, %arg40 = %c0_i32, %arg41 = %c0_i32) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, i32, i32, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, i32, i32, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, i32, i32)  : i32 {
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          %57 = llvm.inttoptr %18 : i64 to !llvm.ptr<11>
          %58 = llvm.load %57 : !llvm.ptr<11> -> i32
          %59 = arith.andi %58, %5 : i32
          %60 = arith.cmpi eq, %59, %5 : i32
          %61 = arith.andi %58, %6 : i32
          %62 = arith.cmpi eq, %61, %c0_i32 : i32
          %63 = arith.andi %58, %7 : i32
          %64 = arith.cmpi eq, %63, %7 : i32
          %65 = arith.cmpi slt, %arg26, %47 : i32
          %66 = arith.andi %60, %62 : i1
          %67 = arith.andi %66, %65 : i1
          %68 = arith.cmpi slt, %arg27, %47 : i32
          %69 = arith.andi %64, %68 : i1
          %70:16 = scf.if %67 -> (tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, i32, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, i32, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, i32) {
            %72 = tt.splat %arg21 : i32 -> tensor<16xi32>
            %73 = arith.addi %72, %22 : tensor<16xi32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
            %memspacecast = memref.memory_space_cast %alloc_13 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
            %74 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
            %75 = arith.mulf %74, %23 : tensor<16x16xf32>
            %76 = tt.expand_dims %73 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
            %77 = tt.broadcast %76 : tensor<1x16xi32> -> tensor<16x16xi32>
            %78 = arith.addi %43, %77 : tensor<16x16xi32>
            %79 = arith.cmpi slt, %76, %cst_6 : tensor<1x16xi32>
            %80 = tt.broadcast %79 : tensor<1x16xi1> -> tensor<16x16xi1>
            %81 = arith.andi %44, %80 : tensor<16x16xi1>
            %82 = tt.addptr %27, %78 : tensor<16x16x!tt.ptr<i1>>, tensor<16x16xi32>
            %83 = tt.bitcast %82 : tensor<16x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i8>>
            %84 = tt.load %83, %81, %cst_5 : tensor<16x16x!tt.ptr<i8>>
            %85 = arith.cmpi ne, %84, %cst_5 : tensor<16x16xi8>
            %86 = arith.select %85, %75, %cst_4 : tensor<16x16xi1>, tensor<16x16xf32>
            %87 = "tt.reduce"(%86) <{axis = 1 : i32}> ({
            ^bb0(%arg42: f32, %arg43: f32):
              %132 = arith.maxnumf %arg42, %arg43 : f32
              tt.reduce.return %132 : f32
            }) : (tensor<16x16xf32>) -> tensor<16xf32>
            %88 = tt.expand_dims %87 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
            %89 = tt.broadcast %88 : tensor<16x1xf32> -> tensor<16x16xf32>
            %90 = arith.subf %86, %89 : tensor<16x16xf32>
            %91 = math.exp %90 : tensor<16x16xf32>
            %92 = "tt.reduce"(%91) <{axis = 1 : i32}> ({
            ^bb0(%arg42: f32, %arg43: f32):
              %132 = arith.addf %arg42, %arg43 : f32
              tt.reduce.return %132 : f32
            }) : (tensor<16x16xf32>) -> tensor<16xf32>
            %93 = math.log %92 : tensor<16xf32>
            %94 = arith.addf %87, %93 : tensor<16xf32>
            %95 = math.exp %arg23 : tensor<16xf32>
            %96 = arith.addf %94, %cst_3 : tensor<16xf32>
            %97 = math.exp %96 : tensor<16xf32>
            %98 = arith.addf %95, %97 : tensor<16xf32>
            %99 = math.log %98 : tensor<16xf32>
            %100 = arith.cmpf une, %99, %99 : tensor<16xf32>
            %101 = arith.select %100, %arg23, %99 : tensor<16xi1>, tensor<16xf32>
            %102 = arith.subf %arg23, %101 : tensor<16xf32>
            %103 = math.exp %102 : tensor<16xf32>
            %104 = arith.cmpf oeq, %87, %cst_2 : tensor<16xf32>
            %105 = arith.select %104, %cst_1, %103 : tensor<16xi1>, tensor<16xf32>
            %106 = tt.expand_dims %105 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
            %107 = tt.broadcast %106 : tensor<16x1xf32> -> tensor<16x128xf32>
            %108 = arith.mulf %arg22, %107 : tensor<16x128xf32>
            %109 = arith.remsi %arg40, %c6_i32 : i32
            %110 = arith.cmpi eq, %109, %c0_i32 : i32
            %111 = arith.select %110, %108, %arg24 : tensor<16x128xf32>
            %112:5 = scf.if %110 -> (tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>) {
              scf.yield %arg35, %arg36, %arg37, %arg38, %arg39 : tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>
            } else {
              %132 = arith.cmpi eq, %109, %c1_i32 : i32
              %133 = arith.select %132, %108, %arg35 : tensor<16x128xf32>
              %134:4 = scf.if %132 -> (tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>) {
                scf.yield %arg36, %arg37, %arg38, %arg39 : tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>
              } else {
                %135 = arith.cmpi eq, %109, %c2_i32 : i32
                %136 = arith.select %135, %108, %arg36 : tensor<16x128xf32>
                %137:3 = scf.if %135 -> (tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>) {
                  scf.yield %arg37, %arg38, %arg39 : tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>
                } else {
                  %138 = arith.cmpi eq, %109, %c3_i32 : i32
                  %139 = arith.select %138, %108, %arg37 : tensor<16x128xf32>
                  %140:2 = scf.if %138 -> (tensor<16x128xf32>, tensor<16x128xf32>) {
                    scf.yield %arg38, %arg39 : tensor<16x128xf32>, tensor<16x128xf32>
                  } else {
                    %141 = arith.cmpi eq, %109, %c4_i32 : i32
                    %142 = arith.select %141, %108, %arg38 : tensor<16x128xf32>
                    %143 = arith.select %141, %arg39, %108 : tensor<16x128xf32>
                    scf.yield %142, %143 : tensor<16x128xf32>, tensor<16x128xf32>
                  }
                  scf.yield %139, %140#0, %140#1 : tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>
                }
                scf.yield %136, %137#0, %137#1, %137#2 : tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>
              }
              scf.yield %133, %134#0, %134#1, %134#2, %134#3 : tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>
            }
            %113 = arith.addi %arg40, %c1_i32 : i32
            %114 = arith.subf %94, %101 : tensor<16xf32>
            %115 = math.exp %114 : tensor<16xf32>
            %116 = arith.remsi %arg33, %c6_i32 : i32
            %117 = arith.cmpi eq, %116, %c0_i32 : i32
            %118 = arith.select %117, %115, %arg25 : tensor<16xf32>
            %119:5 = scf.if %117 -> (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) {
              scf.yield %arg28, %arg29, %arg30, %arg31, %arg32 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
            } else {
              %132 = arith.cmpi eq, %116, %c1_i32 : i32
              %133 = arith.select %132, %115, %arg28 : tensor<16xf32>
              %134:4 = scf.if %132 -> (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) {
                scf.yield %arg29, %arg30, %arg31, %arg32 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
              } else {
                %135 = arith.cmpi eq, %116, %c2_i32 : i32
                %136 = arith.select %135, %115, %arg29 : tensor<16xf32>
                %137:3 = scf.if %135 -> (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) {
                  scf.yield %arg30, %arg31, %arg32 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
                } else {
                  %138 = arith.cmpi eq, %116, %c3_i32 : i32
                  %139 = arith.select %138, %115, %arg30 : tensor<16xf32>
                  %140:2 = scf.if %138 -> (tensor<16xf32>, tensor<16xf32>) {
                    scf.yield %arg31, %arg32 : tensor<16xf32>, tensor<16xf32>
                  } else {
                    %141 = arith.cmpi eq, %116, %c4_i32 : i32
                    %142 = arith.select %141, %115, %arg31 : tensor<16xf32>
                    %143 = arith.select %141, %arg32, %115 : tensor<16xf32>
                    scf.yield %142, %143 : tensor<16xf32>, tensor<16xf32>
                  }
                  scf.yield %139, %140#0, %140#1 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
                }
                scf.yield %136, %137#0, %137#1, %137#2 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
              }
              scf.yield %133, %134#0, %134#1, %134#2, %134#3 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
            }
            %120 = arith.addi %arg33, %c1_i32 : i32
            %121 = tt.expand_dims %92 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
            %122 = tt.broadcast %121 : tensor<16x1xf32> -> tensor<16x16xf32>
            %123 = arith.divf %91, %122 : tensor<16x16xf32>
            %124 = arith.truncf %123 : tensor<16x16xf32> to tensor<16x16xbf16>
            %125 = tt.reshape %124 : tensor<16x16xbf16> -> tensor<1x16x1x16xbf16>
            %126 = tt.trans %125 {order = array<i32: 2, 0, 1, 3>} : tensor<1x16x1x16xbf16> -> tensor<1x1x16x16xbf16>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
            %127 = bufferization.to_memref %126 : memref<1x1x16x16xbf16, #hivm.address_space<ub>>
            hivm.hir.copy ins(%127 : memref<1x1x16x16xbf16, #hivm.address_space<ub>>) outs(%alloc : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>>)
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
            %128 = llvm.load %57 : !llvm.ptr<11> -> i32
            %129 = arith.andi %128, %8 : i32
            %130 = arith.ori %129, %6 : i32
            llvm.store %130, %57 : i32, !llvm.ptr<11>
            %131 = arith.addi %arg26, %5 : i32
            scf.yield %101, %111, %118, %131, %119#0, %119#1, %119#2, %119#3, %119#4, %120, %112#0, %112#1, %112#2, %112#3, %112#4, %113 : tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, i32, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, i32, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, i32
          } else {
            scf.yield %arg23, %arg24, %arg25, %arg26, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40 : tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, i32, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, i32, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, i32
          }
          %71:4 = scf.if %69 -> (tensor<16x128xf32>, i32, i32, i32) {
            %72 = arith.remsi %arg41, %c6_i32 : i32
            %73 = arith.cmpi eq, %72, %c0_i32 : i32
            %74 = scf.if %73 -> (tensor<16x128xf32>) {
              scf.yield %70#1 : tensor<16x128xf32>
            } else {
              %90 = arith.cmpi eq, %72, %c1_i32 : i32
              %91 = scf.if %90 -> (tensor<16x128xf32>) {
                scf.yield %70#10 : tensor<16x128xf32>
              } else {
                %92 = arith.cmpi eq, %72, %c2_i32 : i32
                %93 = scf.if %92 -> (tensor<16x128xf32>) {
                  scf.yield %70#11 : tensor<16x128xf32>
                } else {
                  %94 = arith.cmpi eq, %72, %c3_i32 : i32
                  %95 = scf.if %94 -> (tensor<16x128xf32>) {
                    scf.yield %70#12 : tensor<16x128xf32>
                  } else {
                    %96 = arith.cmpi eq, %72, %c4_i32 : i32
                    %97 = arith.select %96, %70#13, %70#14 : tensor<16x128xf32>
                    scf.yield %97 : tensor<16x128xf32>
                  }
                  scf.yield %95 : tensor<16x128xf32>
                }
                scf.yield %93 : tensor<16x128xf32>
              }
              scf.yield %91 : tensor<16x128xf32>
            }
            %75 = arith.addi %arg41, %c1_i32 : i32
            %76 = arith.remsi %arg34, %c6_i32 : i32
            %77 = arith.cmpi eq, %76, %c0_i32 : i32
            %78 = scf.if %77 -> (tensor<16xf32>) {
              scf.yield %70#2 : tensor<16xf32>
            } else {
              %90 = arith.cmpi eq, %76, %c1_i32 : i32
              %91 = scf.if %90 -> (tensor<16xf32>) {
                scf.yield %70#4 : tensor<16xf32>
              } else {
                %92 = arith.cmpi eq, %76, %c2_i32 : i32
                %93 = scf.if %92 -> (tensor<16xf32>) {
                  scf.yield %70#5 : tensor<16xf32>
                } else {
                  %94 = arith.cmpi eq, %76, %c3_i32 : i32
                  %95 = scf.if %94 -> (tensor<16xf32>) {
                    scf.yield %70#6 : tensor<16xf32>
                  } else {
                    %96 = arith.cmpi eq, %76, %c4_i32 : i32
                    %97 = arith.select %96, %70#7, %70#8 : tensor<16xf32>
                    scf.yield %97 : tensor<16xf32>
                  }
                  scf.yield %95 : tensor<16xf32>
                }
                scf.yield %93 : tensor<16xf32>
              }
              scf.yield %91 : tensor<16xf32>
            }
            %79 = arith.addi %arg34, %c1_i32 : i32
            %80 = tt.expand_dims %78 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
            %81 = tt.broadcast %80 : tensor<16x1xf32> -> tensor<16x128xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
            %memspacecast = memref.memory_space_cast %alloc_14 : memref<16x128xf32, #hivm.address_space<ub>> to memref<16x128xf32>
            %82 = bufferization.to_tensor %memspacecast restrict writable : memref<16x128xf32>
            %83 = arith.mulf %82, %81 : tensor<16x128xf32>
            %84 = arith.cmpf une, %83, %83 : tensor<16x128xf32>
            %85 = arith.select %84, %cst_10, %83 : tensor<16x128xi1>, tensor<16x128xf32>
            %86 = arith.addf %74, %85 : tensor<16x128xf32>
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
            %87 = llvm.load %57 : !llvm.ptr<11> -> i32
            %88 = arith.andi %87, %9 : i32
            llvm.store %88, %57 : i32, !llvm.ptr<11>
            %89 = arith.addi %arg27, %5 : i32
            scf.yield %86, %89, %79, %75 : tensor<16x128xf32>, i32, i32, i32
          } else {
            scf.yield %arg22, %arg27, %arg34, %arg41 : tensor<16x128xf32>, i32, i32, i32
          }
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
          scf.yield %71#0, %70#0, %70#1, %70#2, %70#3, %71#1, %70#4, %70#5, %70#6, %70#7, %70#8, %70#9, %71#2, %70#10, %70#11, %70#12, %70#13, %70#14, %70#15, %71#3 : tensor<16x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, i32, i32, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, i32, i32, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, tensor<16x128xf32>, i32, i32
        }
        %49 = arith.cmpf une, %48#0, %48#0 : tensor<16x128xf32>
        %50 = arith.select %49, %cst_10, %48#0 : tensor<16x128xi1>, tensor<16x128xf32>
        %51 = arith.muli %38, %31 : tensor<16x1xi32>
        %52 = arith.addi %32, %51 : tensor<16x1xi32>
        %53 = tt.broadcast %52 : tensor<16x1xi32> -> tensor<16x128xi32>
        %54 = arith.addi %53, %34 : tensor<16x128xi32>
        %55 = tt.addptr %35, %54 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
        %56 = arith.truncf %50 : tensor<16x128xf32> to tensor<16x128xbf16>
        tt.store %55, %56 : tensor<16x128x!tt.ptr<bf16>>
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      %16 = arith.divsi %11, %c16_i32 : i32
      %17 = arith.remsi %11, %c16_i32 : i32
      %18 = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32>
      %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %20 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %21 = arith.muli %16, %arg5 : i32
      %22 = arith.muli %17, %arg6 : i32
      %23 = arith.addi %21, %22 : i32
      %24 = tt.splat %arg7 : i32 -> tensor<16x1xi32>
      %25 = tt.splat %23 : i32 -> tensor<16x1xi32>
      %26 = tt.expand_dims %18 {axis = 0 : i32} : tensor<192xi32> -> tensor<1x192xi32>
      %27 = tt.broadcast %26 : tensor<1x192xi32> -> tensor<16x192xi32>
      %28 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x192x!tt.ptr<bf16>>
      %29 = arith.muli %16, %arg8 : i32
      %30 = arith.muli %17, %arg9 : i32
      %31 = arith.addi %29, %30 : i32
      %32 = tt.splat %arg10 : i32 -> tensor<16x1xi32>
      %33 = tt.splat %31 : i32 -> tensor<16x1xi32>
      %34 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x192x!tt.ptr<bf16>>
      %35 = arith.muli %16, %arg11 : i32
      %36 = arith.muli %17, %arg12 : i32
      %37 = arith.addi %35, %36 : i32
      %38 = tt.expand_dims %19 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      scf.for %arg20 = %c0_i32 to %c1024_i32 step %c16_i32  : i32 {
        %39 = tt.splat %arg20 : i32 -> tensor<16xi32>
        %40 = arith.addi %39, %20 : tensor<16xi32>
        %41 = tt.expand_dims %40 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
        %42 = arith.muli %41, %24 : tensor<16x1xi32>
        %43 = arith.addi %25, %42 : tensor<16x1xi32>
        %44 = tt.broadcast %43 : tensor<16x1xi32> -> tensor<16x192xi32>
        %45 = arith.addi %44, %27 : tensor<16x192xi32>
        %46 = arith.cmpi slt, %41, %cst_8 : tensor<16x1xi32>
        %47 = tt.addptr %28, %45 : tensor<16x192x!tt.ptr<bf16>>, tensor<16x192xi32>
        %48 = tt.broadcast %46 : tensor<16x1xi1> -> tensor<16x192xi1>
        %49 = tt.load %47, %48, %cst_0 : tensor<16x192x!tt.ptr<bf16>>
        %50 = arith.addi %arg20, %c1_i32 : i32
        %51 = arith.muli %50, %c2_i32 : i32
        %52 = arith.divsi %51, %c16_i32 : i32
        %53 = arith.divsi %52, %6 : i32
        %54:4 = scf.for %arg21 = %c0_i32 to %51 step %c16_i32 iter_args(%arg22 = %cst_12, %arg23 = %cst_11, %arg24 = %c0_i32, %arg25 = %c0_i32) -> (tensor<16x1xi32>, tensor<16x1xi1>, i32, i32)  : i32 {
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
          %64 = arith.andi %57, %6 : i32
          %65 = arith.andi %58, %6 : i32
          %66 = arith.cmpi eq, %64, %6 : i32
          %67 = arith.cmpi eq, %65, %6 : i32
          %68 = arith.andi %66, %67 : i1
          %69 = arith.andi %57, %7 : i32
          %70 = arith.andi %58, %7 : i32
          %71 = arith.cmpi eq, %69, %c0_i32 : i32
          %72 = arith.cmpi eq, %70, %c0_i32 : i32
          %73 = arith.andi %71, %72 : i1
          %74 = arith.cmpi slt, %arg24, %53 : i32
          %75 = arith.andi %63, %74 : i1
          %76 = arith.cmpi slt, %arg25, %53 : i32
          %77 = arith.andi %68, %73 : i1
          %78 = arith.andi %77, %76 : i1
          %79:3 = scf.if %75 -> (tensor<16x1xi32>, tensor<16x1xi1>, i32) {
            %81 = tt.splat %arg21 : i32 -> tensor<16xi32>
            %82 = arith.addi %81, %20 : tensor<16xi32>
            %83 = tt.expand_dims %82 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
            %84 = arith.muli %83, %32 : tensor<16x1xi32>
            %85 = arith.addi %33, %84 : tensor<16x1xi32>
            %86 = tt.broadcast %85 : tensor<16x1xi32> -> tensor<16x192xi32>
            %87 = arith.addi %86, %27 : tensor<16x192xi32>
            %88 = arith.cmpi slt, %83, %cst_8 : tensor<16x1xi32>
            %89 = tt.addptr %34, %87 : tensor<16x192x!tt.ptr<bf16>>, tensor<16x192xi32>
            %90 = tt.broadcast %88 : tensor<16x1xi1> -> tensor<16x192xi1>
            %91 = tt.load %89, %90, %cst_0 : tensor<16x192x!tt.ptr<bf16>>
            %92 = tt.trans %91 {order = array<i32: 1, 0>} : tensor<16x192xbf16> -> tensor<192x16xbf16>
            %93 = tt.dot %49, %92, %cst_7 : tensor<16x192xbf16> * tensor<192x16xbf16> -> tensor<16x16xf32>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%93 : tensor<16x16xf32>) outs(%alloc_13 : memref<16x16xf32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
            %94 = llvm.load %55 : !llvm.ptr<11> -> i32
            %95 = arith.ori %94, %5 : i32
            %96 = arith.ori %95, %5 : i32
            llvm.store %95, %55 : i32, !llvm.ptr<11>
            llvm.store %96, %56 : i32, !llvm.ptr<11>
            %97 = arith.addi %arg24, %5 : i32
            scf.yield %83, %88, %97 : tensor<16x1xi32>, tensor<16x1xi1>, i32
          } else {
            scf.yield %arg22, %arg23, %arg24 : tensor<16x1xi32>, tensor<16x1xi1>, i32
          }
          %80 = scf.if %78 -> (i32) {
            %81 = tt.broadcast %79#1 : tensor<16x1xi1> -> tensor<16x128xi1>
            %82 = tensor.empty() : tensor<16x128xbf16>
            %83 = scf.for %arg26 = %c0 to %c16 step %c1 iter_args(%arg27 = %82) -> (tensor<16x128xbf16>) {
              %extracted = tensor.extract %79#0[%arg26, %c0] {DiscreteMemAccess} : tensor<16x1xi32>
              %93 = arith.muli %extracted, %arg13 : i32
              %94 = arith.addi %37, %93 : i32
              %95 = tt.splat %94 : i32 -> tensor<1x128xi32>
              %96 = arith.addi %95, %38 : tensor<1x128xi32>
              %97 = arith.extsi %96 : tensor<1x128xi32> to tensor<1x128xi64>
              %98 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<1x128x!tt.ptr<bf16>>
              %99 = tt.addptr %98, %97 : tensor<1x128x!tt.ptr<bf16>>, tensor<1x128xi64>
              %100 = tt.load %99 {DiscreteMemAccess} : tensor<1x128x!tt.ptr<bf16>>
              %inserted_slice = tensor.insert_slice %100 into %arg27[%arg26, 0] [1, 128] [1, 1] : tensor<1x128xbf16> into tensor<16x128xbf16>
              scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x128xbf16>
            } {ExtractedLoadOrStore}
            %84 = arith.select %81, %83, %cst : tensor<16x128xi1>, tensor<16x128xbf16>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
            %85 = hivm.hir.convert_layout %alloc {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<1x1x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<16x16xbf16, #hivm.address_space<cbuf>>
            %memspacecast = memref.memory_space_cast %85 : memref<16x16xbf16, #hivm.address_space<cbuf>> to memref<16x16xbf16>
            %86 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xbf16>
            %87 = tt.dot %86, %84, %cst_10 : tensor<16x16xbf16> * tensor<16x128xbf16> -> tensor<16x128xf32>
            hivm.hir.sync_block_set[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%87 : tensor<16x128xf32>) outs(%alloc_14 : memref<16x128xf32, #hivm.address_space<ub>>)
            hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
            %88 = llvm.load %55 : !llvm.ptr<11> -> i32
            %89 = arith.andi %88, %10 : i32
            %90 = arith.ori %89, %7 : i32
            %91 = arith.ori %90, %7 : i32
            llvm.store %90, %55 : i32, !llvm.ptr<11>
            llvm.store %91, %56 : i32, !llvm.ptr<11>
            %92 = arith.addi %arg25, %5 : i32
            scf.yield %92 : i32
          } else {
            scf.yield %arg25 : i32
          }
          hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
          scf.yield %79#0, %79#1, %79#2, %80 : tensor<16x1xi32>, tensor<16x1xi1>, i32, i32
        }
      }
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    tt.return
  }
}
