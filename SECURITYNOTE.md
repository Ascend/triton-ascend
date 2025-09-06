# Triton Ascend 安全声明

## 系统安全加固

建议用户在系统中配置开启ASLR（级别2 ），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：

    echo 2 > /proc/sys/kernel/randomize_va_space

## 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用triton_ascend。

## 文件权限控制

1. 建议用户对个人的隐私数据、商业资产等敏感文件做好权限控制等安全措施，设定的权限建议参考[文件权限参考](#文件权限参考)进行设置。

2. 用户安装和使用过程需要做好权限控制，建议参考[文件权限参考](#文件权限参考)进行设置。


##### 文件权限参考

|   类型                             |   Linux权限参考最大值   |
|----------------------------------- |-----------------------|
|  用户主目录                         |   750（rwxr-x---）     |
|  程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）     |
|  程序文件目录                       |   550（r-xr-x---）     |
|  配置文件                           |   640（rw-r-----）     |
|  配置文件目录                       |   750（rwxr-x---）     |
|  日志文件(记录完毕或者已经归档)       |   440（r--r-----）     |
|  日志文件(正在记录)                  |   640（rw-r-----）    |
|  日志文件目录                       |   750（rwxr-x---）     |
|  Debug文件                         |   640（rw-r-----）      |
|  Debug文件目录                      |   750（rwxr-x---）     |
|  临时文件目录                       |   750（rwxr-x---）     |
|  维护升级文件目录                   |   770（rwxrwx---）      |
|  业务数据文件                       |   640（rw-r-----）      |
|  业务数据文件目录                   |   750（rwxr-x---）      |
|  密钥组件、私钥、证书、密文文件目录   |   700（rwx------）      |
|  密钥组件、私钥、证书、加密密文       |   600（rw-------）     |
|  加解密接口、加解密脚本              |   500（r-x------）      |


## 构建安全声明

triton_ascend支持源码编译安装，在编译时会下载依赖第三方库并执行构建shell脚本，在编译过程中会产生临时程序文件和编译目录。用户可根据需要自行对源代码目录内的文件进行权限管控降低安全风险。

## 示例代码安全声明

triton_ascend源码仓中展示的样例、UT、tutorials等代码是为了展示某一特性功能，用户需要自行管理因编写的代码导致的安全性或系统不可控行为。请确保编写的代码可意并进循相关安全规范和最佳实践。

## 公网地址声明

在triton_ascend的配置文件和脚本中存在[公网地址](#公网地址)

##### 公网地址
| 类型     | 开源代码地址                                                                                     | 文件名                                      | 公网IP地址/公网URL地址/域名/邮箱地址                                                                 | 用途说明                          |
|----------|------------------------------------------------------------------------------------------------|-------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------|
| 自研     | 不涉及                                                                                         | ascend/examples/generalization_cases/run_daily.sh & scripts/prepare_build.sh | https://gitee.com/shijingchang/triton.git                                                           | 构建依赖代码仓                 |
| 自研     | 不涉及                                                                                         | setup.py                                   | https://gitee.com/ascend/triton-ascend/                                                             | triton_ascend源码仓地址 |
| 开源引入 | https://gitclone.com                                                            | scripts/prepare_build.sh                   | https://gitclone.com/github.com/llvm/llvm-project.git                                               | 依赖的llvm源码仓    |
| 开源引入 | https://repo.huaweicloud.com                                            | scripts/prepare_build.sh                           | https://repo.huaweicloud.com/repository/pypi/simple                                                | 用于配置pybind11下载连接 |
| 开源引入 | https://triton-ascend-artifacts.obs.myhuaweicloud.com | setup.py |https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz | 用于下载预编译的LLVM工具 |
| 开源引入 | https://github.com | .gitmodules |https://github.com/triton-lang/triton.git | triton官方代码仓，用于构建triton_ascend |
| 开源引入 | https://cdn.openai.com | docs/conf.py |https://cdn.openai.com/triton/assets/triton-logo.png | triton log文件地址，用于展示在文档中 |
| 开源引入 | https://repo.huaweicloud.com | scripts/prepare_build.sh |https://repo.huaweicloud.com/repository/pypi/simple/ | pybind11 whl包托管地址 |
