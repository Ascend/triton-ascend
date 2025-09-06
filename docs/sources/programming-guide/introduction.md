# 编程指南

## 1. Triton算子开发指南

### 1.1 高效开发原则

在昇腾 NPU 上使用 Triton 编写高性能算子时，为了充分发挥硬件性能，建议遵循以下高效开发原则。

#### 1.1.1 充分利用核数

昇腾 NPU 具备多个计算核心，合理分配并充分利用所有可用核心，是提升算子性能的关键因素之一。
在调用 triton_kernel 函数时，通过设置 launch 参数控制使用的核数量。以 GELU 算子为例：

```Python
triton_gelu[48, 1, 1](...)  # 第一个参数表示使用的核数，48表示使用48个核
```

通过对核数的调优，可实现对所有计算资源的充分调度和利用，从而最大化并行度与吞吐量。注意，当前版本核数需小于等于65535。

#### 1.1.2 数据切分Tiling

写 triton_kernel 内核函数时，合理的数据切分策略对性能优化至关重要。通过调整不同的切分粒度参数，可以在不同维度上平衡计算负载与内存访问效率。

常见的切分参数包括：

```text
ncore：使用的核数（跨核切分）
xblock：核间数据块大小（核间切分）
xblock_sub：核内切分粒度（核内细粒度划分）
```

开发者可根据实际场景手动选择最优的切分配置，使得每次计算尽可能充分利用片上内存（On-chip Memory），避免频繁访问全局内存（Global Memory）造成的
性能瓶颈。

以 GELU 算子为例，通过调整切分参数，可以有效适配片上缓存容量限制，从而提升执行效率。

注：Atlas 800T/I A2产品的片上内存容量为 192KB，因此在设计切分策略时需考虑该限制，确保每轮计算的数据量不超过片上内存容量。

#### 1.1.3 存算并行

Triton-Ascend 支持两种数据处理模式：存算串行 和 存算并行。

存算串行：先从全局内存搬运数据到片上内存，完成计算后，再搬运下一批数据。这种方式存在明显的空闲等待时间，效率较低。

存算并行：在搬运第一批数据至片上内存的同时，已开始对其执行计算；随后继续搬运第二批数据，形成“搬运 + 计算”重叠的流水线式操作，显著提升整体吞吐率。

实现存算并行的关键在于合理设计数据切分（Tiling）策略，使得在当前批次数据计算过程中，能够提前准备下一阶段所需的数据，从而实现数据搬运与计算过程的并
行化。

### 1.2 gelu算子示例

gelu算子开发示例，使用3种方式计算结果。

standard_unary      为标准torch计算。

triton_easy_kernel  为简单triton写法表达。

triton_better_kernel为更高效的triton写法表达。

#### 1.2.1 标准torch写法

输入tensor x0，经过torch计算实现 gelu 算子，返回结果值。

```Python
def standard_unary(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res
```

#### 1.2.2 简单triton写法

以下是一个使用 Triton 编写的简单内核示例，用于展示如何定义和调用一个基本的 Triton 内核函数。此示例实现了一个简单的数学运算（GELU 激活函数）。

```Python
# 定义triton_kernel核函数
@triton.jit
def triton_easy_kernel(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr0 + idx_block, ret)
```

注意事项

1. 内存限制：上述写法中，所有输入数据一次性被加载到内存中进行计算。如果输入张量过大，可能会超出单个内核的片上内存容量，导致内存溢出错误。
因此，这种简单的写法更适合于小规模张量的计算或用于理解 Triton 内核的基本写法和调用方式。

2. 适用场景：尽管这种方法有助于快速理解和入门 Triton 编程，但对于大规模数据集或高性能要求的应用场景，建议采用更复杂的数据切分策略（如 Tiling），
以充分利用硬件资源并避免内存溢出问题。通过这种方式，开发者可以快速上手 Triton 编程，同时了解如何定义、调用以及优化 Triton 内核函数。

#### 1.2.3 更高效triton写法

在昇腾 NPU 上使用 Triton 编写高性能算子时，为了充分利用硬件资源、避免内存溢出并提升执行效率，通常需要采用数据切分（Tiling）策略。
下面是一个经过优化的 Triton 内核实现示例，适用于大规模张量计算。

```Python
# 定义triton_kernel核函数
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

# 调用triton_kernel核函数
ncore = 32
xblock = 32768
xblock_sub = 8192
triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
```

关键代码解释

```Python
xoffset = tl.program_id(0) * XBLOCK

    计算当前核处理数据块的起始偏移地址，实现核间切分。每个核仅负责 XBLOCK 大小的数据范围。

for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):

    在单个核内部进一步细分数据块，每次处理 XBLOCK_SUB 大小的数据，实现核内切分。

x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

    构造当前迭代的数据索引数组，用于访问输入和输出张量。

xmask = x_index < xnumel

    设置掩码以防止越界访问，确保只处理合法范围内的数据。

tl.load() 和 tl.store()

    分别用于从全局内存加载数据到片上内存，以及将计算结果写回全局内存。
```

##### 1.2.3.1 核间切分

以输入张量形状为 (32, 32768) 为例：

可设置 ncore = 32，表示使用 32 个核心进行并行计算。
每个核处理的数据量为：

```text
XBLOCK = 总元素数 / 核数 = (32 × 32768) / 32 = 32768
```

通过合理分配核数，可以最大化并行度，提高整体吞吐率。

##### 1.2.3.2 核内切分

由于 Ascend NPU 的片上内存容量有限（Atlas 800T/I A2产品的片上内存容量为 192KB），若一次性加载 32768 个 float32 类型数据（共 32768 × 4 = 128KB），
加上中间变量和缓存需求，可能超出片上内存容量，导致内存溢出。

因此，需对单个核的任务进一步细分为多个批次，每批处理 XBLOCK_SUB 个数据。示例中选择 XBLOCK_SUB = 8192，即：

每次处理 8192 × 4 = 32KB 数据，留出足够空间供其他临时变量使用。
整体任务将在一个核内分 4 次完成。

这种方式有效降低了单次内存占用，同时保留了较高的计算密度。

##### 1.2.3.3 切分参数的硬件依据

Atlas 800T/I A2产品的片上内存大小为 192KB。假设使用 float32 类型数据，则理论可容纳的最大数据量为（每个float32数占用4个字节数）:

```text
最大数据量 = 192 * 1024 / 4 = 49152 个元素
```

实际应用中，需为寄存器、中间变量等预留部分空间，不能完全占满理论上限。因此选择 XBLOCK_SUB = 8192 是一种较为安全且高效的配置。

##### 1.2.3.4 小结

该写法充分体现了 Triton 高效开发的三个原则：

1. 充分利用核数：使用 ncore = 32 启动 32 个核并行计算，充分发挥 Ascend NPU的多核优势。

2. 数据切分 Tiling：通过 XBLOCK（核间切分）与 XBLOCK_SUB（核内切分）组合策略，使得每次处理的数据量适配片上内存容量，既高效利用资源又避免内存
溢出。

3. 存算并行：通过循环中的分段加载与计算，实现了“搬运一批、计算一批”的流水线式操作，提升了数据搬运与计算的并行性。

##### 1.2.3.5 优化建议

为了筛选出最优的切分参数组合，建议：

1. 手动尝试多组 ncore、XBLOCK、XBLOCK_SUB 值；
2. 结合性能分析工具（如 Profiler）评估不同配置下的执行时间；
3. 不断逼近理论性能上限，找到最适合当前硬件和输入规模的参数组合。

##### 1.2.2.5 完整示例

下面是完整示例，可以复制命名为gelu.py，并运行改示例。

```Python
python gelu.py
```

gelu.py 完整代码如下：

```Python
import triton
import triton.language as tl
import torch

def standard_unary(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res

@triton.jit
def triton_easy_kernel(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr0 + idx_block, ret)

@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

def test_gelu_standard_VS_easy_kernel(shape, NUMEL):
    print(f"input : shape = {shape} NUMEL = {NUMEL}")

    x0 = torch.rand(size=shape, dtype=torch.float32).npu()

    ans = standard_unary(x0)
    print(f"standard output: ans = {ans}")

    out0 = torch.zeros(size=shape, dtype=torch.float32).npu()
    triton_easy_kernel[1, 1, 1](x0, out0, NUMEL=NUMEL)
    print(f"triton_easy_kernel output: out0 = {out0}")

    torch.allclose(out0, ans,  rtol=1e-03, atol=1e-03, equal_nan=True)
    print(f"triton_easy_kernel Pass")

def test_gelu_better_kernel(shape, NUMEL):
    print(f"input : shape = {shape} NUMEL = {NUMEL}")

    x0 = torch.rand(size=shape, dtype=torch.float32).npu()

    ans = standard_unary(x0)
    print(f"standard output: ans = {ans}")

    out1 = torch.zeros(size=shape, dtype=torch.float32).npu()
    ncore = 32
    xblock = 32768
    xblock_sub = 8192
    triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
    print(f"triton_better_kernel output: out1 = {out1}")

    torch.allclose(out1, ans,  rtol=1e-03, atol=1e-03, equal_nan=True)
    print(f"triton_better_kernel Pass")

test_gelu_standard_VS_easy_kernel((32, 64), 32*64)
test_gelu_better_kernel((32, 32768), 32*32768)
```

## 2. 临时文件

triton-ascend运行产生临时文件，开发者可通过临时文件进行调试。

```shell
# 临时文件默认路径
cd ~/.triton/cache
cd ~/.triton/dump
```

### 2.1 cache缓存临时文件

~/.triton/cache 目录是cache缓存位置。cache主要用于缓存编译过程中生成的中间产物和最终结果，以提高重复执行或调试过程中的效率。

1. 加速编译过程：当使用 Triton 编写并编译内核时，从高级语言描述到低级硬件指令集的转换过程，包括优化步骤、代码生成等。通过缓存这些编译阶段的结果，如果源代码或者配置没有改变，Triton 可以直接使用缓存版本而无需重新进行整个编译流程，从而显著加快开发和调试速度。
2. 存储编译后的模块：对于每个编译好的 Triton 内核，其对应的机器码或中间表示会被存储在这个缓存目录下。在后续的执行中，如果需要再次加载相同的内核（例如在不同的 Python 脚本中调用），可以直接从缓存中读取，避免了不必要的重复编译工作。
3. 便于调试和分析：缓存目录下的文件可以作为调试信息的重要来源。通过检查这些缓存文件的内容，开发者能够更深入地了解编译器的行为、优化的效果以及生成代码。

在~/.triton/cache路径下，有每一次生成triton_kernel时产生的缓存文件，主要关注 x.ttadapter、x.ttir、x.so 三种类型的文件。

下面分别解释：

```text
fn_npu_.ttadapter # 将 Triton 写的高级代码转换为适合目标硬件（Ascend NPU）的形式。
fn_npu_.ttir      # 从 ttadapter 文件生成 IR 文件，源代码已经被转化为一种更接近机器码的形式。
launcher_cxx11abi1.cpython-311-aarch64-linux-gnu.so  # 最终编译结果的共享库文件，文件名表示了python311版本，CPU架构，打包成可以被python执行的动态链接库。
```

### 2.2 dump临时文件

默认路径~/.triton/dump/，用于存放编译过程中的中间产物或转储（dump）信息。这些文件可以帮助开发者理解编译流程、优化步骤以及最终生成的代码形态，对于调试和性能分析非常有用。

当运行 Triton 编写的程序，并且如果启用了相关的调试或转储选项时，Triton 会自动在 ~/.triton/dump/ 目录下生成相应的转储文件。
设置环境变量TRITON_DEBUG=1，每一次生成新的triton_kernel时，可以打印对应的dump的路径。主要关注3种临时文件： 

```shell
# 设置环境变量DEBUG
export TRITON_DEBUG=1

# 运行某triton用例前，先清理~/.triton/dump和~/.triton/cache下的文件，再执行某算子
python triton_xxx_test.py

# 在窗口打印日志中，找到dump路径。
cd ~/.triton/dump/

# 解释3种临时文件
kernel.ttadapter.mlir   # 将Triton高级语言描述的内核转化为MLIR格式
kernel.ttir.mlir        # 是MLIR格式的文件，但这个文件代表的是更接近目标硬件的中间表示。
launcher_cxx11abi1.cxx  # 用于封装和调用前面步骤中生成的内核代码，使其能够在Python环境中被调用
```

注： 若设置环境变量TRITON_DEBUG=1，执行triton算子后，未打印dump路径，可先手动删除临时文件:

### 2.3 手动删除临时文件

可以手动删除临时文件目录，不会造成不可逆的影响。删除后，执行triton算子将重新编译生成临时文件。

```shell
rm -rf ~/.triton/cache
rm -rf ~/.triton/dump
```
