import os

import torch
import torch_npu
import triton
import triton.language as tl

torch.set_printoptions(precision=4,
                       sci_mode=False)


@triton.jit
def load_store_scalar_kernel(
    x_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    x = tl.load(x_ptr)
    tl.store(output_ptr, x)


def load_store_scalar(args):
    x = args
    output = torch.empty_like(x)
    grid = (x.shape[0],)
    block_size = 1024
    load_store_scalar_kernel[grid](x, output, BLOCK_SIZE=block_size)
    return output


def test_load_store_scalar():
    size = (2)
    x = torch.rand(size, device="npu")
    output_triton = load_store_scalar(x)
    assert torch.allclose(output_triton[0], x[0])