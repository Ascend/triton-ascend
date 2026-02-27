# 安装指南

## 环境准备

### Python版本要求

当前Triton-Ascend要求的Python版本为:**py3.9-py3.11**。

### 安装Ascend CANN

异构计算架构CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，
向上支持多种AI框架，包括MindSpore、PyTorch、TensorFlow等，向下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台。

您可以访问昇腾社区官网，根据其提供的软件安装指引完成 CANN 的安装配置。

在安装过程中，CANN 版本“**{version}**”请选择 **8.5.0**，并根据实际环境指定CPU架构 “**{arch}**”(aarch64/x86_64)、NPU硬件型号“**{chip_type}**”（910b等）对应的软件包。

建议下载安装:

| 软件类型 | 软件包说明       | 软件包名称                       |
|----------|------------------|----------------------------------|
| Toolkit  | CANN开发套件包   | Ascend-cann-toolkit_**{version}**_linux-**{arch}**.run  |
| Kernels  | CANN二进制算子包 | Ascend-cann-kernels-**{chip_type}**_**{version}**_linux-**{arch}**.run |

社区下载链接：

```bash
https://www.hiascend.com/developer/download/community/result?module=cann
```

社区安装指引链接：

```bash
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit
```

该文档提供了完整的安装流程说明与依赖项配置建议，适用于需要全面部署 CANN 环境的用户。

CANN安装完成后，需要配置环境变量才能生效。请用户根据set_env.sh的实际路径执行如下命令。

```bash
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```

- 注：如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：`/usr/local/Ascend`，非root用户：`${HOME}/Ascend`，${HOME}为当前用户目录。
上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

### 安装python依赖

```bash
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml
```

### 安装torch_npu

当前配套的torch_npu版本为2.6.0版本。

```bash
pip install torch_npu==2.6.0
```

## 通过pip安装Triton-Ascend

### 最新稳定版本
您可以通过pip安装Triton-Ascend的最新稳定版本

```shell
pip install triton-ascend
```

- 注意：社区 Triton 和 Triton-Ascend 不能同时存在。在安装依赖Triton的其他软件时，会自动安装社区 Triton，将覆盖掉已安装的 Triton-Ascend 目录。
此时也需要先卸载社区 Triton，再安装 Triton-Ascend。
```shell
pip uninstall triton
pip install triton-ascend
```

### nightly build版本
我们为用户提供了每日更新的nightly包，用户可以方便的使用以下命令来安装。
```shell
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```
同时用户也能在 [历史列表](https://test.pypi.org/project/triton-ascend/#history) 中找到所有的nightly build包。

注意，如果您在执行`pip install`时遇到ssl相关报错，可追加`--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org`选项解决。

## 通过源码安装Triton-Ascend

如果您需要对 triton-ascend 进行开发或自定义修改，则应采用源代码编译安装的方法。这种方式允许您根据项目需求调整源代码，并编译安装定制化的triton-ascend 版本。

### 系统要求

- GCC >= 9.4.0
- GLIBC >= 2.27

### 依赖

#### 安装系统库依赖

安装zlib1g-dev/lld/clang，可选安装ccache包用于加速构建。

- 推荐版本 clang >= 15
- 推荐版本 lld >= 15

```bash
以ubuntu系统为例：
apt update
apt install zlib1g-dev clang-15 lld-15
apt install ccache # optional
```

#### 安装python依赖

```bash
pip install ninja cmake wheel pybind11 # build-time dependencies
```

### 基于LLVM构建

Triton 使用 LLVM20 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。

1. `git checkout` 检出指定版本的LLVM.

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
   ```

2. clang构建安装LLVM
  
- 步骤1：推荐使用clang安装LLVM，环境上请安装clang、lld，并指定版本（推荐版本clang>=15，lld>=15），
  如未安装，请按下面指令安装clang、lld、ccache：

  ```bash
  apt-get install -y clang-15 lld-15 ccache
  ```

  如果环境上有多个版本的clang，请设置clang为当前安装的版本clang-15，如果clang只有15版本，或已指定15版本则跳过该步骤:

  ```bash
  update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20; \
  update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20; \
  update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 20
  ```

- 步骤2：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- 步骤3：执行以下命令进行构建和安装LLVM：

  ```bash
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

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- 步骤2：执行以下命令进行构建和安装：

   ```bash
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

#### 克隆 Triton-Ascend

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend && git submodule update --init --depth 1
```

#### 构建 Triton-Ascend

1. 源码安装

- 步骤1：请确认已设置[基于LLVM构建]章节中，LLVM安装的目标路径 ${LLVM_INSTALL_PREFIX}
- 步骤2：请确认已安装clang>=15，lld>=15，ccache

   ```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_PLUGIN_DIRS=./ascend \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```

- 注3：推荐GCC >= 9.4.0，如果GCC < 9.4.0，可能报错 “ld.lld: error: unable to find library -lstdc++fs”，说明链接器无法找到 stdc++fs 库。
该库用于支持 GCC 9 之前版本的文件系统特性。此时需要手动把 CMake 文件中相关代码片段的注释打开：

- triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

  打开注释后重新构建项目即可解决该问题。

## 运行Triton示例

   运行实例: [01-vector-add.py](../../../ascend/examples/tutorials/01-vector-add.py)
   ```bash
   # 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 运行tutorials示例：
   python3 ./triton-ascend/ascend/examples/tutorials/01-vector-add.py
   ```
