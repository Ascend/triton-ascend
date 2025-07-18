# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest

import triton
import triton.language as tl
import time

import torch
import torch_npu
import test_common

@pytest.mark.skip(reason="to be supported by bishengir-compile")
def test_min_dim0_3d():

    def torch_func(x, dim):
        res = torch.min(x, dim)
        return res

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0,
        N0: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr):
        idx0 = tl.arange(0, N0)
        idx1 = tl.arange(0, N1)
        idx2 = tl.arange(0, N2)
        in_idx = idx2[None, None, :] + idx1[None, :, None] * N2 + idx0[:, None, None] * N2 * N1
        tmp0 = tl.load(in_ptr0 + in_idx)
        tmp1 = tl.min(tmp0, 0)
        out_idx = idx2[None, :] + idx1[:, None] * N2
        tl.store(out_ptr0 + out_idx, tmp1)

    def triton_func(x0, dim):
        N0, N1, N2 = x0.size()
        y0 = test_common.generate_tensor(shape=(N1,N2), dtype = "float32").npu()
        triton_kernel[1, 1, 1](y0, x0, N0, N1, N2)
        return y0

    dim = 0
    N0, N1, N2 = 1, 22, 13
    x0 = test_common.generate_tensor(shape=(N0,N1,N2), dtype = "float32").npu()
    torch_ref = torch_func(x0, dim)
    triton_cal = triton_func(x0, dim)
    test_common.validate_cmp("float32", triton_cal, torch_ref)

def standard_min(x0,dim,dtype):
    res, index = torch.min(x0, dim)
    return res

@triton.jit
def triton_min_dim0(in_ptr0, out_ptr0, M : tl.constexpr, N : tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0,MNUMEL)
    nblk_idx = tl.arange(0,NNUMEL)

    mmask = mblk_idx<M
    nmask = nblk_idx<N

    mask = (mmask[:,None])&(nmask[None,:])

    idx = mblk_idx[:,None]*N+nblk_idx[None,:]
    if in_ptr0.dtype == tl.int8:
        padding = 127
    else :
        padding = float('inf')
    x=tl.load(in_ptr0+idx, mask = mask, other = padding)

    ret = tl.min(x,0)

    tl.store(out_ptr0+nblk_idx, ret, mask = nmask)

types=[
    (torch.float32,'float32'),
    (torch.float16,'float16'),
    # (torch.bfloat16,'bfloat16'),  TODO: waiting for supporting or testing
    (torch.int8,'int8'),
    # (torch.int16,'int16'),  TODO: waiting for supporting or testing
    # (torch.int32,'int32'),  TODO: waiting for supporting or testing
    # (torch.int64,'int64'),  TODO: waiting for supporting or testing
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
#shapes=[
#    (57,3,64,16), (57,-32,64,32), (57,37,64,64), (57,-256,64,256), (57,263,64,512),
#    (64,3,64,16), (64,-32,64,32), (64,37,64,64), (64,-256,64,256), (64,263,64,512),
#    (3,3,8,8), (-32,3,32,8), (37,3,64,8), (-256,3,256,8), (263,3,512,8),
#    (3,1,8,8), (-32,1,32,8), (37,1,64,8), (-256,1,256,8), (263,1,512,8),
#]
shapes=[
    (64,-32,64,32),
]

map_for_64_t = {37:(31,32),263:(107,128)}
map_for_32_t = {263:(137,256)}

# @pytest.mark.parametrize('dtype, sigtype',[(torch.float32,'float32'),])
@pytest.mark.parametrize('M, N, MNUMEL, NNUMEL',[(64,-32,64,32)])
@pytest.mark.parametrize('dtype, sigtype',types)
# @pytest.mark.parametrize('M, N, MNUMEL, NNUMEL',shapes)
def test_min_dim0(dtype, sigtype, M, N, MNUMEL, NNUMEL):

    M = (-M)//torch.tensor(0,dtype=dtype).element_size() if M<0 else M
    N = (-N)//torch.tensor(0,dtype=dtype).element_size() if N<0 else N

    if sigtype == 'int64':
        M = map_for_64_t[M][0] if M in map_for_64_t else M
        MNUMEL = map_for_64_t[M][1] if M in map_for_64_t else MNUMEL
        N = map_for_64_t[N][0] if N in map_for_64_t else N
        NNUMEL = map_for_64_t[N][1] if N in map_for_64_t else NNUMEL

    elif sigtype == 'float32' or sigtype == 'bfloat16' or sigtype == 'int32':
        M = map_for_32_t[M][0] if M in map_for_32_t else M
        MNUMEL = map_for_32_t[M][1] if M in map_for_32_t else MNUMEL
        N = map_for_32_t[N][0] if N in map_for_32_t else N
        NNUMEL = map_for_32_t[N][1] if N in map_for_32_t else NNUMEL

    print(f"min : ({M}, {N}) {dtype} {sigtype}")
    x0 = test_common.generate_tensor(shape = (M,N),dtype = sigtype)

    ans = standard_min(x0, 0, dtype)

    x0=x0.npu()
    print(ans)

    output = torch.zeros((N,), dtype = dtype).npu()
    triton_min_dim0[1,1,1](x0, output, M = M, N = N,MNUMEL = MNUMEL, NNUMEL = NNUMEL, debug = True)
    print(output)

    test_common.validate_cmp(sigtype,output,ans)
