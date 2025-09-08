# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest

import triton
import triton.language as tl

import torch
import torch_npu

def mask_store(x, valid):
    result = x.clone()
    flattened = result.flatten()
    flattened[:valid] *= 2
    return flattened.reshape(x.shape)

@triton.jit
def triton_mask_store(x_ptr,
    y_ptr, 
    valid,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr, 
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    row_idx = offsets // n_cols
    col_idx = offsets % n_cols

    mask = offsets < valid

    x = tl.load(x_ptr + col_idx + row_idx * n_cols, mask=mask)
    y = x * 2
    tl.store(y_ptr + col_idx + row_idx * n_cols, y, mask=mask)


@pytest.mark.parametrize('shape', [(16, 32)])
@pytest.mark.parametrize('BLOCK_SIZE', [512, 256, 128])
@pytest.mark.parametrize('valid_num', [256, 128, 127, 15])
def test_cases(shape, BLOCK_SIZE, valid_num):
    x = torch.randn((shape), dtype=torch.float32, device='npu')
    # x = torch.ones((X), dtype=torch.float32, device='npu')
    output1 = mask_store(x, valid_num)
    grid = ((shape[0] * shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    triton_mask_store[grid](x, x, valid_num, shape[0], shape[1], BLOCK_SIZE)

    print(output1)
    print(x)

    torch.testing.assert_close(output1, x, rtol=1e-3, atol=1e-3)