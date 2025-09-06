# Triton Ascend

Triton是一种编程语言和编译器，用于高效编写定制的深度学习原语。其目标是提供一个开源环境，让开发者能够高效开发代码，同时兼具比其他现有领域专用语言DSL（domain-specific language）更强的灵活性。

Triton-Ascend面向昇腾平台，旨在让Triton代码能够在昇腾硬件上高效运行。

## 1.环境准备
### Python版本要求
当前Triton-Ascend要求的Python版本为:**py3.9-py3.11**。

### 安装Ascend CANN
异构计算架构CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，
向上支持多种AI框架，包括MindSpore、PyTorch、TensorFlow等，向下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平
台。

您可以访问昇腾社区官网，根据其提供的软件安装指引完成 CANN 的安装配置。

在安装过程中，请选择 CANN 版本 **8.2.RC1.alpha003**，并根据实际环境指定CPU架构(AArch64/X86_64)，NPU硬件型号对应的软件包。

建议下载安装:

| 软件类型 | 软件包说明       | 软件包名称                       |
|----------|------------------|----------------------------------|
| Toolkit  | CANN开发套件包   | Ascend-cann-toolkit_version_linux-arch.run  |
| Kernels  | CANN二进制算子包 | Ascend-cann-kernels-chip_type_version_linux-arch.run |

社区下载链接：
```
https://www.hiascend.com/developer/download/community/result?module=cann
```
社区安装指引链接：
```
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
```
该文档提供了完整的安装流程说明与依赖项配置建议，适用于需要全面部署 CANN 环境的用户。

CANN安装完成后，需要配置环境变量才能生效。请用户根据set_env.sh的实际路径执行如下命令。
```
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```
- 注：如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：“/usr/local/Ascend”，非root用户：“${HOME}/Ascend”，${HOME}为当前用户目录。
上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

### 安装python依赖
```
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml
```

### 安装torch_npu

当前配套的torch_npu版本为2.6.0rc1版本。
```
pip install torch_npu==2.6.0rc1
```

## 2.源代码安装 Triton-Ascend
详细安装手册参见[Installation.md](./docs/Installation.md)
### 系统要求
- GCC >= 9.4.0
- GLIBC >= 2.27
## 依赖

### 包版本依赖

Python支持版本为:**py3.9-py3.11**, torch及torch_npu支持版本为:**2.6.0**。

### 安装系统库依赖

安装zlib1g-dev/lld/clang，可选安装ccache包用于加速构建。
- 推荐版本 clang >= 15
- 推荐版本 lld >= 15
```
以ubuntu系统为例：
apt update
apt install zlib1g-dev clang-15 lld-15
apt install ccache # optional
```

### 安装python依赖
```
pip install ninja cmake wheel pybind11 # build-time dependencies
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml torch==2.6.0 torch-npu==2.6.0rc1 # torch dependencies
```

## 基于LLVM构建

Triton 使用 LLVM20 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。

1. `git checkout` 检出指定版本的LLVM.

   ```
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
   ```

2. clang构建安装LLVM
  
- 步骤1：推荐使用clang安装LLVM，环境上请安装clang、lld，并指定版本(推荐版本clang>=15，lld>=15)，
  如未安装，请按下面指令安装clang、lld、ccache：
  ``` 
  apt-get install -y clang-15 lld-15 ccache
  ``` 
  如果环境上有多个版本的clang，请设置clang为当前安装的版本clang-15，如果clang只有15版本，或已指定15版本则跳过该步骤:
  ``` 
  update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20; \
  update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20; \
  update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 20
  ```
- 步骤2：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：
   ```
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```
- 步骤3：执行以下命令进行构建和安装LLVM：
  ```
  cd $HOME/llvm-project  # your clone of LLVM.
  mkdir build
  cd build
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX} \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
  ninja install
  ```
3. GCC构建安装LLVM
- 步骤1：推荐使用clang，如果只能使用GCC安装，请注意[注1] [注2]。设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：
   ```
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```
- 步骤2：执行以下命令进行构建和安装：
   ```
   cd $HOME/llvm-project  # your clone of LLVM.
   mkdir build
   cd build
   cmake -G Ninja  ../llvm  \
      -DLLVM_CCACHE_BUILD=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm"  \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
   ninja install
   ```
- 注1：若在编译时出现错误`ld.lld: error: undefined symbol`，可在步骤2中加入设置`-DLLVM_ENABLE_LLD=ON`。
- 注2：若环境上ccache已安装且正常运行，可设置`-DLLVM_CCACHE_BUILD=ON`加速构建, 否则请勿开启。


### 克隆 Triton-Ascend

```
git clone https://gitee.com/ascend/triton-ascend.git --recurse-submodules --shallow-submodules
```

### 构建 Triton-Ascend

1. 源码安装

- 步骤1：请确认已设置[基于LLVM构建]章节中，LLVM安装的目标路径 ${LLVM_INSTALL_PREFIX}
- 步骤2：请确认已安装clang>=15，lld>=15，ccache
   ```
   cd triton-ascend/
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_PLUGIN_DIRS=./ascend \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton_ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```
- 注3：推荐GCC >= 9.4.0，如果GCC < 9.4，可能报错 “ld.lld: error: unable to find library -lstdc++fs”，说明链接器无法找到 stdc++fs 库。
该库用于支持 GCC 9 之前版本的文件系统特性。此时需要手动把 CMake 文件中相关代码片段的注释打开：
- triton-ascend/CMakeLists.txt
   ```
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```
  打开注释后重新构建项目即可解决该问题。

2. 运行Triton示例
   ```
   # 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 运行tutorials示例：
   python3 ./triton-ascend/ascend/examples/tutorials/01-vector-add.py
   ```


# 3.环境变量

环境变量配置参考下表：

| 环境变量                        | 默认值        | 功能说明                                                                                                                                                                                                                                                                                                                                                                                                                    | 配置说明                                                                                                                      | 变更声明                                                                                                                                                                                           |
|---------------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TRITON_DEBUG                    | 0 或未设置     | 启用 Triton 的调试输出功能，用于在运行时打印详细的调试信息。这对于排查编译或执行阶段的问题非常有用。 当设置为 1 时，Triton 会输出更多关于编译过程、内核生成和执行的信息。 某些实现中可能支持更细粒度的调试级别（如 2, 3 等），具体取决于 Triton 的版本和实现。                                                                                                                                                              | 0：不启用DEBUG 1：启用DEBUG                                                                                                     |                                                                                                                                                                                                    |
| TRITON_ALWAYS_COMPILE           | 0 或未设置     | 控制 Triton 是否每次运行都强制重新编译内核，而不是使用已有的缓存版本。 默认情况下，Triton 会对已经编译过的内核进行缓存（基于参数和配置），以提高性能。 设置为 1 后，Triton 将忽略缓存并每次都重新编译内核，这在调试或测试新编译器特性时非常有用。                                                                                                                                                                           | 0：不启用 1：每次运行都重新编译所有内核                                                                                         |                                                                                                                                                                                                    |
| MLIR_ENABLE_DUMP                | 0 或未设置     | 在每次 MLIR 优化前转储所有内核的 IR。使用 `MLIR_ENABLE_DUMP=kernelName`可以只转储特定内核的IR。                                                                                                                                                                                                                                                                                                                             | 0：不转储 1：转储所有内核IR kernelName：转储特定内核IR                                                                          | Triton 缓存可能干扰转储。如果 `MLIR_ENABLE_DUMP=1`  不生效，可尝试清理 Triton 缓存： `rm -r ~/.triton/cache/*`                                                                                     |
| LLVM_IR_ENABLE_DUMP             | 0 或未设置     | 在每次 LLVM IR 优化前转储 IR。                                                                                                                                                                                                                                                                                                                                                                                              | 0：不转储 1：转储IR                                                                                                             |                                                                                                                                                                                                    |
| TRITON_REPRODUCER_PATH          | 未设置        | 在每个 MLIR 编译阶段前生成 MLIR 复现文件。如果某阶段失败，`<reproducer_path>`  将保存失败前的 MLIR 状态。                                                                                                                                                                                                                                                                                                                   | <reproducer_path>：保存路径                                                                                                    |                                                                                                                                                                                                    |
| TRITON_INTERPRET                | 0 或未设置     |  使用 Triton 解释器而非 GPU 运行，支持在核函数代码中插入 Python 断点                                                                                                                                                                                                                                                                                                                                                        | 0：不支持断点 1：支持断点                                                                                                       |                                                                                                                                                                                                    |
| TRITON_ENABLE_LLVM_DEBUG        | 0 或未设置     | 向LLVM 传递`-debug`参数，输出大量调试信息。若信息过多，可使用`TRITON_LLVM_DEBUG_ONLY`限制输出范围。                                                                                                                                                                                                                                                                                                                         | 0：不传递 1：传递                                                                                                               | 另一种减少输出干扰的方法是：先设置 `LLVM_IR_ENABLE_DUMP=1`运行程序，提取目标LLVM优化通道前的中间表示（IR），然后单独运行LLVM的`opt`工具，此时可通过命令行添加`-debug-only=foo`参数来限定调试范围。 |
| TRITON_LLVM_DEBUG_ONLY          | 未设置        | 功能等同于 LLVM 的`-debug-only`命令行选项。该参数可将 LLVM 调试输出限定到特定的优化通道或组件名称（这些名称通过 LLVM 和 Triton 中的`#define DEBUG_TYPE`宏定义），从而有效减少调试信息的冗余输出。用户可指定一个或多个逗号分隔的值，例如：`TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions`或`TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`。                                           | <comma-separated>：通道或组件名称                                                                                               |                                                                                                                                                                                                    |
| USE_IR_LOC                      | 0 或未设置     | 控制是否在生成的中间表示（IR）中包含位置信息（如文件名、行号等）。这些信息对调试很有帮助，但可能会增加生成的IR的大小。设置为1，会重新解析中间表示(IR)，将位置信息映射为具有特定扩展名的IR文件行号（而非Python源文件行号）。这能建立从IR到LLVM IR/PTX的直接映射关系。配合性能分析工具使用时，可实现对IR指令的细粒度性能剖析。                                                                                                | 0：不包含位置信息 1：包含位置信息                                                                                               |                                                                                                                                                                                                    |
| TRITON_PRINT_AUTOTUNING         | 0 或未设置     | 在自动调优完成后，输出每个内核的最佳配置及总耗时。                                                                                                                                                                                                                                                                                                                                                                          | 0：不输出 1：输出                                                                                                               |                                                                                                                                                                                                    |
| DISABLE_LLVM_OPT                | 0 或未设置     | 当设置为 1 时，可以禁用 LLVM 编译过程中的优化步骤(make_llir和make_ptx的LLVM优化)。当设置为字符串，解析为要禁用的LLVM优化标志列表。例如使用`DISABLE_LLVM_OPT="disable-lsr"`可禁用循环强度优化（该优化在某些存在寄存器压力的内核中可能导致高达10%的性能波动）。                                                                                                                                                               | 0：LLVM 的优化是启用状态 1：禁用 LLVM 编译过程中的优化步骤(make_llir和make_ptx的LLVM优化) <list>:"disable-lsr":禁用循环强度优化 |                                                                                                                                                                                                    |
| MLIR_ENABLE_TIMING              | 0 或未设置     | 启用或禁用 MLIR 编译过程中的时间统计功能。                                                                                                                                                                                                                                                                                                                                                                                  | 0：不启用 1：启用                                                                                                               |                                                                                                                                                                                                    |
| LLVM_ENABLE_TIMING              | 0 或未设置     | 启用或禁用 LLVM 编译过程中的时间统计功能。                                                                                                                                                                                                                                                                                                                                                                                  | 0：不启用 1：启用                                                                                                               |                                                                                                                                                                                                    |
| TRITON_DEFAULT_FP_FUSION        | 1 启用       | 控制是否默认启用浮点运算融合优化，覆盖默认的浮点运算融合行为（如mul+add->fma）。                                                                                                                                                                                                                                                                                                                                            | 0：不启用 1：启用                                                                                                               |                                                                                                                                                                                                    |
| MLIR_ENABLE_REMARK              | 0 或未设置     | 启用MLIR 编译过程中的备注信息输出，包括以备注形式输出的性能警告。                                                                                                                                                                                                                                                                                                                                                           | 0：不启用 1：启用                                                                                                               |                                                                                                                                                                                                    |
| TRITON_KERNEL_DUMP              | 0 或未设置     | 启用或禁用 Triton 内核的转储功能，当启用时，Triton 会将生成的内核代码（各编译阶段IR及最终PTX）保存到指定目录。                                                                                                                                                                                                                                                                                                              | 0：不启用 1：启用                                                                                                               |                                                                                                                                                                                                    |
| TRITON_DUMP_DIR                 | 当前工作目录或未设置 | 指定 Triton 内核转储文件的保存目录。当`TRITON_KERNEL_DUMP=1`时保存IR和PTX的目录。                                                                                                                                                                                                                                                                                                                                           | "path"：保存路径                                                                                                                |                                                                                                                                                                                                    |
| TRITON_KERNEL_OVERRIDE          | 0 或未设置     | 启用或禁用 Triton 内核覆盖功能，允许在每个编译阶段开始时用用户指定的外部文件(IR/PTX等)覆盖默认生成的内核代码。                                                                                                                                                                                                                                                                                                              | 0：不启用 1：启用                                                                                                               |                                                                                                                                                                                                    |
| TRITON_OVERRIDE_DIR             | 当前工作目录或未设置 | 指定 Triton 内核覆盖文件的查找目录。当`TRITON_KERNEL_OVERRIDE=1`时加载IR/PTX文件的目录。                                                                                                                                                                                                                                                                                                                                    | "path"：保存路径                                                                                                               |                                                                                                                                                                                                    |
| TRITON_DEVICE_PRINT             | 0 或未设置     | 启用`tl.device_print`功能，当设置为`1` 或者 `true`时（`TRUE` 将被转换为 `true`）。 重要说明：该功能使用GM缓冲区（其指针被传递给内核）。当前由于未知原因，要求内核标量参数总数（包括隐藏参数`gridx, gridy, gridz`）必须为偶数。例如内核参数为`kernel(ptr0, ptr1, ptr2, BLOCKSIZE: tl.constexpr)`时，前3个指针参数（64位对齐）加上3个隐藏参数（32位对齐）后，需额外添加1个标量参数使总数变为偶数。此限制将在修复该bug后移除。 | 0：不启动 1：启用`tl.device_print`功能                                                                                          | 每个线程的GM缓冲区最大为16KB，超限内容将被丢弃。该值目前固定，后续将通过环境变量调整。                                                                                                             |
| TRITON_BENCH_METHOD             | 未设置        | 使用昇腾NPU时，将`testing.py`中的`do_bench`切换为`do_bench_npu`（需配合`INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE = 1`使用）。设为`default`时即使NPU可用，仍调用原函数。                                                                                                                                                                                                                                                          | "npu"：切换为`do_bench_npu`                                                                                                   |                                                                                                                                                                                                    |
| TRITON_ASCEND_COMPILE_SPEED_OPT | 0 或未设置     | 控制JIT编译器在发现内核编译失败后是否跳过后续编译阶段。设为`1`跳过（默认`0`继续尝试）。                                                                                                                                                                                                                                                                                                                                     | 0：继续尝试 1：跳过                                                                                                               |                                                                                                                                                                                                    |
| TRITON_REGISTER_TENSOR_MSPROF   | 0 或未设置     | 控制使用profiler时是否在运行时向msprof注册tensor信息。因为运行时解析并注册tensor信息会导致运行时开销增大，所以默认关闭。                                                                                                                                                                                                                                                                                                                                       | 0：不注册 1：注册                                                                                                               |                                                                                                                                                                                                    |
| TRITON_NPU_COMPILER_PATH        | 未设置     | bishengir-compile的文件路径，可手动配置                                                                                                                                                                                                                                                                                                                                      |  bishengir-compile的文件路径，可手动配置                                                                                                              |                                                                                                                                                                                                    |
| TRITON_ENABLE_SANITIZER         | 0 或未设置     | 控制使能triton_ascend的地址消毒功能                                                                                                                                                                                                                                                                                                                                       |  0：不启用 1：启用                                                                                                              |                                                                                                                                                                                                    |
| ASCEND_HOME_PATH                |      | triton对CANN包的依赖，由CANN包指定，无需手动配置                                                                                                                                                                                                                                                                                                                                       |  NA                                                                                                              |                                                                                                                                                                                                    |
| FLAGGEMS_CACHE_DIR              |  未设置    | FLAGGEMS算子库文件缓存目录，默认为~/.flaggems                                                                                                                                                                                                                                                                                                                                   | FLAGGEMS算子库文件缓存目录，可手动配置                                                                                                              |                                                                                                                                                                                                    |
| TRITON_VERSION                  |  未设置    | triton_ascend版本号，默认为3.2.0                                                                                                                                                                                                                                                                                                                                   | triton_ascend版本号，可手动配置                                                                                                              |                                                                                                                                                                                                    |
| IS_MANYLINUX                    |  未设置    | 是否打manylinux包，默认为0                                                                                                                                                                                                                                                                                                                                   | 1: 不启用, 1：启用                                                                                                              |                                                                                                                                                                                                    |
| MLIR_ROOT                       |  未设置    | MLIR根目录，当前未启用                                                                                                                                                                                                                                                                                                                                  |                                                                                                              |                                                                                                                                                                                                    |

另有环境变量继承自triton官方，此处未全部列出，详情可参考triton官网。

# 4.示例

环境配置完成后，可通过教程脚本快速上手，教程路径：`triton-ascend/docs/tutorials_src`，解释了每一个示例代码的详细执行步骤。

可执行示例代码路径：`triton-ascend/ascend/examples/tutorials`

```
cd triton-ascend/ascend/examples/tutorials
# take 01-vector-add.py for example
python3 01-vector-add.py
```

# 5.调试Triton-Ascend

参考triton社区提供的调试方法进行调试，官方链接：https://triton-lang.org/main/programming-guide/chapter-3/debugging.html

# 6. Triton算子性能调优

triton autotune性能配置说明参考本仓库的 docs\sources\getting-started\tutorials\06-autotune.md

# 7.当前支持的Ascend设备

  - 已支持：Atlas 800T/I A2产品
  - 开发中：Atlas 800T/I A3产品

# 8.当前支持的triton op列表
## triton op 支持度总览

|                          |        Triton Op       | int8 | int16 | int32 | uint32 | int64 | fp16 | fp32 | bf16 | bool |
|:------------------------:|:----------------------:|------|-------|-------|--------|-------|------|------|------|------|
|       Creation Ops       | arange                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | cat                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | full                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | zeros                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | zeros_like             | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | cast                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|  Shape Manipulation Ops  | broadcast              | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | broadcast_to           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | expand_dims            | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | interleave             | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | join                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | permute                | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓    |
|                          | ravel                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | reshape                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | split                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | trans                  | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓    |
|                          | view                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|    Linear Algebra Ops    | dot                    | ✓    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | dot_scaled             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|    Memory/Pointer Ops    | load                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | store                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | make_block_ptr         | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | advance                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|       Indexing Ops       | flip                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | where                  | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ✓*   |
|                          | swizzle2d              | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|         Math Ops         | add                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | sub                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | mul                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | div                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | floordiv(//)           | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | mod                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | neg                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | invert(!)              | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | and(&)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | or(\|)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | xor(^)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | not(~)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | lshift(<<)             | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | rshift(>>)             | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | gt                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | ge                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | lt                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | le                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | eq                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | ne                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | logical and            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ✓    |
|                          | logical or             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ✓    |
|                          | abs                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | cdiv                   | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | ceil                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | clamp                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | cos                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | div_rn                 | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | erf                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | exp                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | exp2                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | fdiv                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | floor                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | fma                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | log                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | log2                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | maximum                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | minimum                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | rsqrt                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sigmoid                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sin                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | softmax                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sqrt                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sqrt_rn                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | umulhi                 | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|       Reduction Ops      | argmax                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | argmin                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | max                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*   |
|                          | min                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*    |
|                          | reduce                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*    |
|                          | sum                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓*    |
|                          | xor_sum                | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓*    |
|       Scan/Sort Ops      | associative_scan       | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | cumprod                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | cumsum                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | histogram              | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | sort                   | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | gather                 | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|        Atomic Ops        | atomic_add             | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_and             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_cas             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_max             | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_min             | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | atomic_or              | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_xchg            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_xor             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
| Random Number Generation | randint4x              | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | randint                | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | rand                   | ×    | ×     | ×     | ×      | ×     | ×    | ✓    | ×    | ×    |
|                          | randn                  | ×    | ×     | ×     | ×      | ×     | ×    | ✓    | ×    | ×    |
|         Iterators        | range                  | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | static_range           | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|      Inline Assembly     | inline_asm_elementwise | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|     Compiler Hint Ops    | debug_barrier          | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | max_constancy          | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | max_contiguous         | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | multiple_of            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|         Debug Ops        | static_print           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | static_assert          | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | device_print           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ×    | ✓    |
|                          | device_assert          | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |

## 约束说明

- dot: 两个输入A[batch(optional), M, K], B[batch(optional), K, N]，M，N按照16对齐，K按照32B对齐。

- gather: triton.gather(x, index, axis)，假设x的shape为n维度，目前只支持axis=n-1。

- permute: triton.permute(x, dims)，不支持dims=[2, 1, 0]。

- trans: triton.trans(x, dims)，不支持dims=[2, 1 , 0]。

- device_print: 需要增加2个环境变量，TRITON_DEVICE_PRINT=1，TRITON_ENABLE_TASKQUEUE=0。**TRITON_ENABLE_TASKQUEUE=0可能造成程序运行不稳定，建议仅临时使用。**

- atomic_add: 不支持标量（包括长度为1的tensor）访存

- atomic_max: 不支持标量（包括长度为1的tensor）访存

- atomic_min: 不支持标量（包括长度为1的tensor）访存

- permute: 不支持不相邻轴转置，如`(0, 1, 2) -> (2, 1, 0)`

- trans: 不支持不相邻轴转置，如`(0, 1, 2) -> (2, 1, 0)`

- umulhi: 不支持负数输入

- mod: int64仅支持处理 -2^24 ~ 2^24 范围内的数值

- ALL: int8类型由于特殊处理，会占用更大的片上空间，编译时容易造成ub overflow报错，通常调整tilling即可解决
- ALL: triton kernel中同时存在所有tensor总和不能超过96KB，若关闭double buffer，则不能超过192KB
- ALL: 所有tensor不允许某个shape的size小于1
- ALL: ✓*表示triton内部将bool类型转为int8类型进行运算，并能够执行得到结果的OP
- ALL: 不支持使用shape为"[[]]"的标量tensor进行计算

# 9. 当前支持的开源算子仓算子列表

## FlagGems:
- abs
- add
- bitwise_and
- bitwise_not
- bitwise_or
- cos
- div
- eq
- exp
- ge
- gt
- isinf
- rsub
- le
- lt
- mul
- ne
- neg
- reciprocal
- relu
- rsqrt
- sigmoid
- silu
- sin
- sub
- tanh
- triu

其他开源算子仓（如vllm、sglang等）正在逐步支持中，敬请期待。
