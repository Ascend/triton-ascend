# Triton Ascend

Triton是近几年来受到开发者青睐的Python化编程语言。开发者仅需关注Tile/Block的切分方式以及基于Tile/Block的运算逻辑，编译器将在Triton代码的编译过程中结合底层硬件特点自动完成内存分配、数据搬运、数据计算、流水并行等，因此，算子的开发难度大幅降低、开发效率显著提升。

Triton Ascend是面向昇腾平台构建的Triton编译框架，旨在让Triton代码能够在昇腾硬件上高效运行。目前，Triton Ascend还在不断完善中，我们将不断提升Triton Python API完备度、数据类型支持度、访存方式灵活性等，并持续优化编译器的自动优化能力，提升Triton Ascend整体的功能与性能泛化性。


- #### RoadMap

| 里程碑 | 重要特性更新情况 | 状态 |
|------|------|------|
| 2025.11.14 | triton-ascend 3.2.0rc4预发布版本上线 | ✅ |
| 2025.09.30 | 完善Scan/Sort类Triton Python API，支持非连续访存，完成vLLM、sglang开源仓中重点Triton算子适配 | ✅ |
| 2025.08.15 | 完善Atomic类Triton Python API支持，完成Flaggems开源仓重点Triton算子适配，提供Matmul等简单算子高性能实现参考用例 | ✅ |
| 2025.06.30 | 支持85% Triton Python API，支持连续访存，覆盖基本使用场景需求 | ✅ |
| 2025.05.20 | Triton Ascend开源，Gitcode代码仓Alive！ | ✅ |

- #### 已支持平台

昇腾设备：Ascend Atlas A2/A3

主机CPU架构：x86/ARM

主机操作系统：Linux Ubuntu

## 帮助文档

欢迎广大开发者试用，但在您开始使用之前，建议您先根据您的开发需求浏览下列文档，希望能够帮助您快速上手！如果您在使用过程中遇到了问题，请您提交Issue反馈相关信息，我们将竭尽全力处理，感谢您的支持！

- #### Triton Ascend 安装
详细安装步骤请参考 [安装指南](./docs/sources/getting-started/installation.md) 。

- #### Triton Ascend Dockerfile开发环境
我们为开发者提供了Dockerfile，帮助开发者快速构建开发环境。Dockerfile文件请参考[Dockerfile](./docker/Dockerfile) 。

- #### Triton Python API支持情况与约束
目前Triton Ascend已经使能了85%以上Triton社区官方提供的Python API，详细的功能支持情况（包括数据类型支持度、使用约束等）请参考 [API 支持情况总览](./docs/sources/python-api/outline.md) 。

- #### Triton算子开发指南（入门级）
在昇腾平台上开发Triton算子的方式与在GPU平台上基本相同。我们提供了下列算子的示例源码与配套说明来解释如何开发Triton算子的设备侧Kernel函数、主机侧调用代码以及算子功能验证代码。
此外，面向不同数据类型，我们提供了用于验证Triton算子精度的示例代码供大家参考：[算子精度验证开发指南](./docs/sources/getting-started/tutorials/07-accuracy-comparison.md) 与 [参考样例Python文件](./ascend/examples/tutorials/14-accuracy-comparison.py) 。

| 算子名称 | 开发指南 | 可执行Python文件 |
|------|------|------|
| VectorAdd |  [VectorAdd开发指南](./docs/sources/getting-started/tutorials/01-vector-add.md) | [VectorAdd Python文件](./ascend/examples/tutorials/01-vector-add.py) |
| Softmax |  [Softmax开发指南](./docs/sources/getting-started/tutorials/02-fused-softmax.md) | [Softmax Python文件](./ascend/examples/tutorials/02-fused-softmax.py) |
| LayerNorm |  [LayerNorm开发指南](./docs/sources/getting-started/tutorials/03-layer-norm.md) | [LayerNorm Python文件](./ascend/examples/tutorials/03-layer-norm.py) |
| FlashAttention |  [FlashAttention开发指南](./docs/sources/getting-started/tutorials/04-fused-attention.md) | [FlashAttention Python文件](./ascend/examples/tutorials/04-fused-attention.py) |
| Matmul |  [Matmul开发指南](./docs/sources/getting-started/tutorials/05-matrix-multiplication.md) | [Matmul Python文件](./ascend/examples/tutorials/05-matrix-multiplication.py) |

- #### Triton算子自动寻优指南（入门级）
Triton Ascend支持Triton原生的Autotune能力。通过对Tile/Block的形状配置进行搜索寻优，开发者可以在不改变Triton算子写法的条件下获得更优的性能。
此外，配合Triton Ascend编译器自有的自动优化算法，我们也额外提供了新的可调优参数，开发者可以按需选用。关于Triton算子自动寻优，详情请参考
[Autotune性能寻优指南](./docs/sources/getting-started/tutorials/06-autotune.md) 。

- #### 高性能Triton算子编程开发指南（进阶级）
为能获得更好的执行性能，除了利用Autotune之外，开发者在编写Triton算子时需要结合昇腾平台的软硬件特点进行开发。我们总结梳理了一些通用优化思路与方法，包括Tile/Block切分方式、高效访存方式以及如何与编译器开展协同优化等，详情请参考[高性能Triton算子编程指南](./docs/HighPerformanceGuide.md) 。

- #### 非昇腾平台Triton算子快速迁移指南
目前，许多开源仓已经提供了面向GPU等平台开发的Triton算子。因为昇腾平台在内存大小、运行时接口功能上与GPU等平台存在差异，将这些算子迁移到昇腾平台运行需要完成少量必要的代码修改，具体修改方法请参考[非昇腾平台Triton算子迁移指南](./docs/sources/programming-guide/migration.md) 。

- #### 开源仓Triton算子适配与支持情况
我们也正逐步将主流开源仓中的GPU Triton算子适配到昇腾平台，当前已适配的算子请参见 [已适配开源仓算子列表](./docs/OPLIST.md) 。

- #### Triton Ascend调试调优工具使用指南
为了简化用户编程难度，助力开发者低成本完成高性能算子开发，MindStudio算子开发工具已适配Triton算子调试调优。  
  1） msProf op: 算子性能调优工具，支持Triton算子上板性能数据采集、内存热力图、仿真流水图生成等，详情参考：[MindStudio 算子性能调优工具使用参考](./docs/sources/mindstudio-guide/01-msProf_op.md)   
  2） msSanitizer: 算子内存检测工具，支持Triton算子异常行为检测，包括内存、竞争、未初始化异常检测，可精准定位异常代码行，详情参考：[MindStudio 算子异常检测工具使用参考](./docs/sources/mindstudio-guide/02-msSanitizer.md)  


- #### Triton Ascend环境变量
Triton Ascend支持Triton原生的环境变量，此外，面向昇腾平台上的新功能特性进行了拓展。Triton Ascend涉及的全量环境变量，请参考 [环境变量总览](./docs/ENVIRONMENT.md) 。

- #### Triton Ascend常见报错与应对方案
针对在开发或迁移Triton算子时经常遇到的报错信息与解决方案，我们正在汇总并梳理相关文档供大家参考。该文档正在开发中，将于近期发布，敬请期待。

## 安全声明
我们重视开发者在使用Triton Ascend时的信息安全，安全防护建议与相关信息请见 [安全声明](./SECURITYNOTE.md) 。

## 许可证
本项目代码与文档均采用 [MIT许可证](./LICENSE) 。
