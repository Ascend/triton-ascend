#include "TritonToLinalg/Passes.h"
#include "TritonToAnnotation/Passes.h"
#include "bishengir/InitAllDialects.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

int main(int argc, char **argv) {
  // Register all dialects.
  mlir::DialectRegistry registry;
  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
      mlir::bufferization::BufferizationDialect, mlir::gpu::GPUDialect>();
  bishengir::registerAllDialects(registry);

  // Register all passes.
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToAnnotationPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Triton-Adapter test driver\n", registry));
}
