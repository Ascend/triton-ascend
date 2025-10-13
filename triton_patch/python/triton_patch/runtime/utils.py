# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch

from .driver import driver

# npu hardware params
target = driver.active.get_current_target()
device = driver.active.get_current_device()
prop = driver.active.utils.get_device_properties(device)

num_cube_core = prop["num_aicore"]
num_vector_core = prop["num_aicore"]

if "Ascend910B" in target.arch:
    num_vector_core = num_cube_core * 2

# wrapper npu 32 bytes align, get and pass unalign info to triton meta
# then autotune choose tiling param and send them to bishengIR
byte_per_numel = {
    torch.float32: 4,  # torch.float32 or torch.float
    torch.float64: 8,  # torch.float64 or torch.double
    torch.float16: 2,  # torch.float16 or torch.half
    torch.bfloat16: 2,  # torch.bfloat16
    torch.int32: 4,  # torch.int32 or torch.int
    torch.int64: 8,  # torch.int64 or torch.long
    torch.int16: 2,  # torch.int16 or torch.short
    torch.int8: 1,  # torch.int8
    torch.uint8: 1,  # torch.uint8
    torch.bool: 1,  # torch.bool
    torch.complex32: 4,  # torch.complex32 (not yet available in PyTorch as of the latest stable release)
    torch.complex64: 8,  # torch.complex64
    torch.complex128: 16,  # torch.complex128
}

valid_axis_names = {
    "x",
    "y",
    "z",
    "w",
    "v",
    "t",
    "rx",
    "ry",
    "rz",
    "rw",
    "rv",
    "rt",
}


def get_byte_per_numel(dtype: torch.dtype) -> int:
    return 1 if dtype is None else byte_per_numel[dtype]


def is_valid_axis_name(name: str) -> bool:
    return name in valid_axis_names


# move to an appropriate place, currently duplicated with triton.__init__.py
def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n
