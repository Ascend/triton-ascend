"""Regression repros for backend miscompiles found porting real kernels on Ascend 950DT.

Each test documents one miscompile observed with triton_ascend 3.6.0.post2 / torch_npu
2.10.0.post7 / CANN 9.2.1. Tests are marked xfail while the bug is open; when a fix lands the
xfail flips to XPASS and should be promoted to a strict pass. Reference issues:

- #14  tl.dot + tl.atomic_add in one kernel -> consolidateToTail SIGABRT
- #15  stores of dot-derived tensors miscompiled when a loop wraps tl.dot
- #16  tl.atomic_add(sem="relaxed") aborts the backend compiler
- #17  1D tl.cumsum returns wrong values
- #18  tl.where unreliable with int64 conditions/operands
- #19  while loop carrying a scalar from a nested loop -> MLIR same-parent-block assert
- #20  3D reductions in loop bodies assert; gather-store corrupts later loads
- #22  int64 comparisons on tl.sort-derived values miscompare
"""

import pytest
import torch
import triton
import triton.language as tl


def _npu():
    return torch.device("npu")


@pytest.mark.xfail(reason="issue #14: dot + atomic in one kernel aborts in consolidateToTail")
def test_dot_and_atomic_same_kernel():
    @triton.jit
    def k(a_ptr, b_ptr, out_ptr):
        offs = tl.arange(0, 16)
        a = tl.load(a_ptr + offs[:, None] * 16 + offs[None, :])
        b = tl.load(b_ptr + offs[:, None] * 16 + offs[None, :])
        c = tl.dot(a, b)
        tl.atomic_add(out_ptr, tl.sum(c))

    dev = _npu()
    a = torch.randn(16, 16, device=dev, dtype=torch.bfloat16)
    b = torch.randn(16, 16, device=dev, dtype=torch.bfloat16)
    out = torch.zeros(1, device=dev, dtype=torch.float32)
    k[(1,)](a, b, out)
    torch.npu.synchronize()
    ref = (a.float() @ b.float()).sum()
    assert torch.allclose(out[0], ref, atol=1e-1)


@pytest.mark.xfail(reason="issue #15: dot in a loop, dot-derived store miscompiled")
def test_loop_dot_store():
    @triton.jit
    def k(a_ptr, b_ptr, out_ptr):
        offs = tl.arange(0, 16)
        a = tl.load(a_ptr + offs[:, None] * 16 + offs[None, :])
        b = tl.load(b_ptr + offs[:, None] * 16 + offs[None, :])
        for w in tl.static_range(2):
            aw = a[w * 8:(w + 1) * 8, :]
            bw = b[w * 8:(w + 1) * 8, :]
            c = tl.dot(aw, bw)
            tl.store(out_ptr + w * 8 + tl.arange(0, 8), tl.sum(c, 1))

    dev = _npu()
    a = torch.randn(16, 16, device=dev, dtype=torch.bfloat16)
    b = torch.randn(16, 16, device=dev, dtype=torch.bfloat16)
    out = torch.zeros(16, device=dev, dtype=torch.float32)
    k[(1,)](a, b, out)
    torch.npu.synchronize()
    ref = (a.float() @ b.float()).sum(1)
    assert torch.allclose(out, ref, atol=1e-1)


@pytest.mark.xfail(reason="issue #16: sem=relaxed atomic aborts the backend compiler")
def test_atomic_relaxed():
    @triton.jit
    def k(out_ptr):
        tl.atomic_add(out_ptr, 1.0, sem="relaxed")

    out = torch.zeros(1, device=_npu(), dtype=torch.float32)
    k[(1,)](out)
    torch.npu.synchronize()
    assert out.item() == 1.0


@pytest.mark.xfail(reason="issue #17: 1D cumsum computes wrong values")
def test_cumsum_1d():
    @triton.jit
    def k(x_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        c = tl.cumsum(x, 0)
        tl.store(out_ptr + offs, c)

    n = 64
    dev = _npu()
    x = torch.arange(1, n + 1, device=dev, dtype=torch.int32)
    out = torch.zeros(n, device=dev, dtype=torch.int32)
    k[(1,)](x, out, N=n)
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), torch.cumsum(x.cpu(), 0))


@pytest.mark.xfail(reason="issue #18: tl.where with int64 condition takes the true branch for every lane")
def test_where_int64_condition():
    @triton.jit
    def k(x_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        y = tl.where(x > 2, x, -4611686018427387904)
        tl.store(out_ptr + offs, y)

    n = 8
    dev = _npu()
    x = torch.arange(n, device=dev, dtype=torch.int64)
    out = torch.zeros(n, device=dev, dtype=torch.int64)
    k[(1,)](x, out, N=n)
    torch.npu.synchronize()
    ref = torch.where(x.cpu() > 2, x.cpu(), torch.tensor(-4611686018427387904))
    assert torch.equal(out.cpu(), ref)


@pytest.mark.xfail(reason="issue #19: while + nested loop scalar carries hit a same-parent-block assert")
def test_while_nested_scalar():
    @triton.jit
    def k(x_ptr, out_ptr):
        n = tl.load(x_ptr)
        acc = tl.zeros((), tl.int32)
        while n > 0:
            for i in tl.static_range(4):
                acc = acc + 1
            n = n - 1
        tl.store(out_ptr, acc.to(tl.float32))

    dev = _npu()
    x = torch.tensor([3], device=dev, dtype=torch.int32)
    out = torch.zeros(1, device=dev, dtype=torch.float32)
    k[(1,)](x, out)
    torch.npu.synchronize()
    assert out.item() == 12.0


@pytest.mark.xfail(reason="issue #20: 3D reduce inside a loop hits a same-parent-block assert")
def test_3d_reduce_in_loop():
    @triton.jit
    def k(x_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs[:, None, None] * N * N + offs[None, :, None] * N + offs[None, None, :])
        for i in tl.static_range(2):
            s = tl.sum(x, 2)
            tl.store(out_ptr + offs[:, None] * N + offs[None, :], s)

    n = 8
    dev = _npu()
    x = torch.randn(n, n, n, device=dev)
    out = torch.zeros(n, n, device=dev)
    k[(1,)](x, out, N=n)
    torch.npu.synchronize()
    assert torch.allclose(out.cpu(), x.cpu().sum(2))


def test_gather_store_load_same_program():
    # issue #20 second half: a store to a gathered address must be visible to later loads in
    # the same program; kept unmarked to watch the behaviour, values checked against torch
    @triton.jit
    def k(x_ptr, idx_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        for i in tl.static_range(2):
            idx = tl.load(idx_ptr + i)
            val = tl.load(x_ptr + idx)
            tl.store(x_ptr + idx, val + 100.0)
            after = tl.load(x_ptr + offs)
            tl.store(out_ptr + i * N + offs, after)

    n = 16
    dev = _npu()
    x = torch.arange(n, device=dev, dtype=torch.float32)
    idx = torch.tensor([0, 5], device=dev, dtype=torch.int32)
    out = torch.zeros(2 * n, device=dev, dtype=torch.float32)
    k[(1,)](x, idx, out)
    torch.npu.synchronize()
    assert out.cpu()[n:2 * n][0].item() == 100.0
    assert out.cpu()[n:2 * n][5].item() == 105.0


@pytest.mark.xfail(reason="issue #22: int64 compare on sort-derived values miscompares equal lanes")
def test_sort_threshold_compare():
    @triton.jit
    def k(x_ptr, out_ptr, THR: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        srt = tl.sort(x, descending=True)
        kept = srt > THR
        tl.store(out_ptr + offs, kept.to(tl.int8))

    n = 16
    thr = -(2 ** 31)  # equals the high word of -inf filler keys in real top-k kernels
    dev = _npu()
    keys = torch.full((n,), -(2 ** 63))
    keys[:5] = torch.arange(5) * 2 ** 32
    out = torch.zeros(n, device=dev, dtype=torch.int8)
    k[(1,)](keys.to(dev), out, THR=thr, N=n)
    torch.npu.synchronize()
    got = out.cpu()
    assert got[:5].sum() == 5 and got[5:].sum() == 0
