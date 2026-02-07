import pytest
import triton
import torch
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice
import test_common


def test_set_element_1d():
    @triton.jit
    def test_kernel(input_ptr, index_ptr, value_ptr, output_ptr, n: tl.constexpr):
        offsets = tl.arange(0, n)
        x = tl.load(input_ptr + offsets)
        idx = tl.load(index_ptr)
        val = tl.load(value_ptr)
        result = libdevice.set_element(x, idx, val)
        tl.store(output_ptr + offsets, result)

    n = 5
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='npu')
    index = torch.tensor(2, dtype=torch.int32, device='npu')
    value = torch.tensor(99.0, device='npu')
    output_tensor = torch.empty_like(input_tensor)

    test_kernel[(1,)](input_tensor, index, value, output_tensor, n)

    expected = input_tensor.clone()
    expected[2] = 99.0

    assert torch.allclose(output_tensor, expected), f"1D test failed. Got {output_tensor}, expected {expected}"
    print("✅ 1D Test passed!")


@pytest.mark.skip() 
def test_set_element_2d():
    @triton.jit
    def test_kernel_2d(input_ptr, indices_ptr, value_ptr, output_ptr,
                       rows: tl.constexpr, cols: tl.constexpr):
        row_idx = tl.arange(0, rows)
        col_idx = tl.arange(0, cols)
        row_grid = tl.reshape(row_idx, (rows, 1))
        col_grid = tl.reshape(col_idx, (1, cols))
        offsets = row_grid * cols + col_grid
        x = tl.load(input_ptr + offsets)
        indices = tl.load(indices_ptr + tl.arange(0, 2))
        val = tl.load(value_ptr)
        result = libdevice.set_element(x, indices, val)
        tl.store(output_ptr + offsets, result)

    rows, cols = 3, 4
    input_tensor = torch.arange(12, dtype=torch.float32, device='npu').reshape(rows, cols)
    indices = torch.tensor([1, 2], dtype=torch.int32, device='npu')
    value = torch.tensor(99.0, device='npu')
    output_tensor = torch.empty_like(input_tensor)

    test_kernel_2d[(1,)](input_tensor, indices, value, output_tensor, rows, cols)

    expected = input_tensor.clone()
    expected[1, 2] = 99.0

    assert torch.allclose(output_tensor, expected), f"2D test failed. Got {output_tensor}, expected {expected}"
    print("✅ 2D Test passed!")


def test_set_element_3d():
    @triton.jit
    def test_kernel_3d(input_ptr, indices_ptr, value_ptr, output_ptr,
                       dim0: tl.constexpr, dim1: tl.constexpr, dim2: tl.constexpr):
        d0_idx = tl.arange(0, dim0)
        d1_idx = tl.arange(0, dim1)
        d2_idx = tl.arange(0, dim2)
        d0_grid = tl.reshape(d0_idx, (dim0, 1, 1))
        d1_grid = tl.reshape(d1_idx, (1, dim1, 1))
        d2_grid = tl.reshape(d2_idx, (1, 1, dim2))
        offsets = d0_grid * dim1 * dim2 + d1_grid * dim2 + d2_grid
        x = tl.load(input_ptr + offsets)
        indices = tl.load(indices_ptr + tl.arange(0, 3))
        val = tl.load(value_ptr)
        result = libdevice.set_element(x, indices, val)
        tl.store(output_ptr + offsets, result)

    dim0, dim1, dim2 = 2, 3, 4
    input_tensor = torch.arange(24, dtype=torch.float32, device='npu').reshape(dim0, dim1, dim2)
    indices = torch.tensor([1, 2, 3], dtype=torch.int32, device='npu')
    value = torch.tensor(99.0, device='npu')
    output_tensor = torch.empty_like(input_tensor)

    test_kernel_3d[(1,)](input_tensor, indices, value, output_tensor, dim0, dim1, dim2)

    expected = input_tensor.clone()
    expected[1, 2, 3] = 99.0

    assert torch.allclose(output_tensor, expected), f"3D test failed. Got {output_tensor}, expected {expected}"
    print("✅ 3D Test passed!")


def test_set_element_4d():
    @triton.jit
    def test_kernel_4d(input_ptr, indices_ptr, value_ptr, output_ptr,
                       d0: tl.constexpr, d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr):
        i0 = tl.reshape(tl.arange(0, d0), (d0, 1, 1, 1))
        i1 = tl.reshape(tl.arange(0, d1), (1, d1, 1, 1))
        i2 = tl.reshape(tl.arange(0, d2), (1, 1, d2, 1))
        i3 = tl.reshape(tl.arange(0, d3), (1, 1, 1, d3))
        offsets = i0 * d1 * d2 * d3 + i1 * d2 * d3 + i2 * d3 + i3
        x = tl.load(input_ptr + offsets)
        indices = tl.load(indices_ptr + tl.arange(0, 4))
        val = tl.load(value_ptr)
        result = libdevice.set_element(x, indices, val)
        tl.store(output_ptr + offsets, result)

    shape = (2, 2, 3, 2)
    input_tensor = torch.arange(24, dtype=torch.float32, device='npu').reshape(shape)
    indices = torch.tensor([1, 1, 2, 1], dtype=torch.int32, device='npu')
    value = torch.tensor(99.0, device='npu')
    output_tensor = torch.empty_like(input_tensor)

    test_kernel_4d[(1,)](input_tensor, indices, value, output_tensor, *shape)

    expected = input_tensor.clone()
    expected[1, 1, 2, 1] = 99.0

    assert torch.allclose(output_tensor, expected), f"4D test failed. Got {output_tensor}, expected {expected}"
    print("✅ 4D Test passed!")


@pytest.mark.skip(reason="Skip this already passed test case to speed up CI.")
def test_set_element_5d():
    @triton.jit
    def test_kernel_5d(input_ptr, indices_ptr, value_ptr, output_ptr,
                       d0: tl.constexpr, d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr, d4: tl.constexpr):
        i0 = tl.reshape(tl.arange(0, d0), (d0, 1, 1, 1, 1))
        i1 = tl.reshape(tl.arange(0, d1), (1, d1, 1, 1, 1))
        i2 = tl.reshape(tl.arange(0, d2), (1, 1, d2, 1, 1))
        i3 = tl.reshape(tl.arange(0, d3), (1, 1, 1, d3, 1))
        i4 = tl.reshape(tl.arange(0, d4), (1, 1, 1, 1, d4))
        offsets = (
            i0 * d1 * d2 * d3 * d4 +
            i1 * d2 * d3 * d4 +
            i2 * d3 * d4 +
            i3 * d4 +
            i4
        )
        x = tl.load(input_ptr + offsets)
        indices = tl.load(indices_ptr + tl.arange(0, 5))
        val = tl.load(value_ptr)
        result = libdevice.set_element(x, indices, val)
        tl.store(output_ptr + offsets, result)

    shape = (2, 1, 2, 2, 3)
    input_tensor = torch.arange(24, dtype=torch.float32, device='npu').reshape(shape)
    indices = torch.tensor([1, 0, 1, 1, 2], dtype=torch.int32, device='npu')
    value = torch.tensor(-88.5, device='npu')
    output_tensor = torch.empty_like(input_tensor)

    test_kernel_5d[(1,)](input_tensor, indices, value, output_tensor, *shape)

    expected = input_tensor.clone()
    expected[1, 0, 1, 1, 2] = -88.5

    assert torch.allclose(output_tensor, expected), f"5D test failed. Got {output_tensor}, expected {expected}"
    print("✅ 5D Test passed!")


@pytest.mark.skip(reason="Skip this already passed test case to speed up CI.")
@pytest.mark.parametrize('dtype_str', [
    'float32',
    'float16',
    'bfloat16',
    'uint8',
    'int8',
    'int16',
    'int32',
    'int64',
    'bool'
])
def test_set_element_dtype(dtype_str):
    """Test set_element with various data types using test_common.generate_tensor."""
    
    @triton.jit
    def kernel(input_ptr, index_ptr, value_ptr, output_ptr, n: tl.constexpr):
        offsets = tl.arange(0, n)
        x = tl.load(input_ptr + offsets)
        idx = tl.load(index_ptr)
        val = tl.load(value_ptr)
        result = libdevice.set_element(x, idx, val)
        tl.store(output_ptr + offsets, result)

    shape = (128,)  # 1D tensor
    index_pos = 50  # modify element at position 50

    input_tensor = test_common.generate_tensor(shape, dtype_str).npu()
    
    if dtype_str in ['float32', 'float16', 'bfloat16']:
        new_val_py = 99.0
    elif dtype_str == 'bool':
        new_val_py = True
    elif dtype_str == 'int8':
        new_val_py = 100  # within [0, 127] as per generate_tensor
    elif dtype_str == 'uint8':
        new_val_py = 200  # within [0, 255]
    elif dtype_str == 'int16':
        new_val_py = 30000
    elif dtype_str == 'int32':
        new_val_py = 1_000_000_000
    elif dtype_str == 'int64':
        new_val_py = 10**12
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    if dtype_str == 'bool':
        value_tensor = torch.tensor(new_val_py, dtype=torch.bool, device='npu')
    else:
        torch_dtype = eval('torch.' + dtype_str)
        value_tensor = torch.tensor(new_val_py, dtype=torch_dtype, device='npu')

    index_tensor = torch.tensor(index_pos, dtype=torch.int32, device='npu')
    output_tensor = torch.empty_like(input_tensor)

    kernel[(1,)](input_tensor, index_tensor, value_tensor, output_tensor, shape[0])

    expected = input_tensor.clone()
    expected[index_pos] = value_tensor.item()

    if dtype_str in ['float16', 'bfloat16', 'float32']:
        assert torch.allclose(output_tensor, expected, atol=1e-3), f"Failed for {dtype_str}"
    else:
        assert torch.equal(output_tensor, expected), f"Failed for {dtype_str}"


if __name__ == "__main__":
    test_set_element_1d()
    test_set_element_2d()
    test_set_element_3d()
    test_set_element_4d()
    test_set_element_5d()
    test_set_element_dtype()