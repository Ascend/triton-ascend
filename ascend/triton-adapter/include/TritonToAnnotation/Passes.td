/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#ifndef TRITON_TO_ANNOTATION_CONVERSION_PASSES
#define TRITON_TO_ANNOTATION_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToAnnotation : Pass<"triton-to-annotation", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Annotation dialect";
  let constructor = "triton::createTritonToAnnotationPass()";
  let dependentDialects = ["annotation::AnnotationDialect"];
}

#endif // TRITON_TO_ANNOTATION_CONVERSION_PASSES
