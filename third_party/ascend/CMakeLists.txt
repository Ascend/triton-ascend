include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR
set(ASCENDNPU_IR_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/AscendNPU-IR")
set(ASCENDNPU_IR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/AscendNPU-IR")

set(BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES ON)
set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)

add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)

# # To enable coverage, you need to pass the argument TRITON_APPEND_CMAKE_ARGS="-DTRITON_ENABLE_COVERAGE=ON" during compilation.
option(TRITON_ENABLE_COVERAGE "Enable code coverage for Ascend plugin" OFF)

add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ir.cc

  LINK_LIBS
  TritonToLinalg
  BiShengIRScopeDialect
  BiShengIRHIVMDialect
)

if(TRITON_ENABLE_COVERAGE)
  message(STATUS "Enabling coverage flags for TritonAscend")
  target_compile_options(TritonAscend PRIVATE
    -fprofile-arcs
    -ftest-coverage
    -O0
    -fprofile-update=atomic
    --coverage
  )
  target_link_options(TritonAscend PRIVATE
    --coverage
    -lgcov
  )
  # branch coverage
  target_compile_definitions(TritonAscend PRIVATE
    COVERAGE_ENABLED=1
  )
endif()

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
