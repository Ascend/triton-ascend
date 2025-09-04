import os
import platform
import re
import contextlib
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
import urllib.request
import glob
import pybind11

from io import BytesIO
from pathlib import Path
from typing import List, NamedTuple, Optional
from dataclasses import dataclass
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.command.clean import clean
from wheel.bdist_wheel import bdist_wheel

root_dir = os.path.dirname(os.path.abspath(__file__))
triton_dir = os.path.join(root_dir, "third_party/triton")


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_env_with_keys(key: list):
    for k in key:
        if k in os.environ:
            return os.environ[k]
    return ""


def remove_directory(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def get_build_type():
    if check_env_flag("DEBUG"):
        return "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        return "RelWithDebInfo"
    elif check_env_flag("TRITON_REL_BUILD_WITH_ASSERTS"):
        return "TritonRelBuildWithAsserts"
    elif check_env_flag("TRITON_BUILD_WITH_O1"):
        return "TritonBuildWithO1"
    else:
        return "TritonRelBuildWithAsserts"


@dataclass
class Backend:
    name: str
    package_data: List[str]
    language_package_data: List[str]
    src_dir: str
    backend_dir: str
    language_dir: Optional[str]
    install_dir: str
    is_external: bool


class BackendInstaller:
    @staticmethod
    def prepare(
        backend_name: str, backend_src_dir: str = None, is_external: bool = False
    ):
        # Initialize submodule if there is one for in-tree backends.
        if not is_external:
            root_dir = os.path.join(os.pardir, "third_party")
            assert backend_name in os.listdir(
                root_dir
            ), f"{backend_name} is requested for install but not present in {root_dir}"

            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", f"{backend_name}"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    cwd=root_dir,
                )
            except subprocess.CalledProcessError:
                pass
            except FileNotFoundError:
                pass

            backend_src_dir = os.path.join(root_dir, backend_name)

        backend_path = os.path.abspath(os.path.join(backend_src_dir, "backend"))
        assert os.path.exists(backend_path), f"{backend_path} does not exist!"

        language_dir = os.path.abspath(os.path.join(backend_src_dir, "language"))
        if not os.path.exists(language_dir):
            language_dir = None

        for file in ["compiler.py", "driver.py"]:
            assert os.path.exists(
                os.path.join(backend_path, file)
            ), f"${file} does not exist in ${backend_path}"

        install_dir = os.path.join(
            triton_dir, "python", "triton", "backends", backend_name
        )
        package_data = [
            f"{os.path.relpath(p, backend_path)}/*"
            for p, _, _, in os.walk(backend_path)
        ]

        language_package_data = []
        if language_dir is not None:
            language_package_data = [
                f"{os.path.relpath(p, language_dir)}/*"
                for p, _, _, in os.walk(language_dir)
            ]

        return Backend(
            name=backend_name,
            package_data=package_data,
            language_package_data=language_package_data,
            src_dir=backend_src_dir,
            backend_dir=backend_path,
            language_dir=language_dir,
            install_dir=install_dir,
            is_external=is_external,
        )

    # Copy all in-tree backends under triton/third_party.
    @staticmethod
    def copy(active):
        return [BackendInstaller.prepare(backend) for backend in active]

    # Copy all external plugins provided by the `TRITON_PLUGIN_DIRS` env var.
    # TRITON_PLUGIN_DIRS is a semicolon-separated list of paths to the plugins.
    # Expect to find the name of the backend under dir/backend/name.conf
    @staticmethod
    def copy_externals():
        backend_dirs = os.getenv("TRITON_PLUGIN_DIRS")
        if backend_dirs is None:
            return []
        backend_dirs = backend_dirs.strip().split(";")
        backend_names = [
            Path(os.path.join(dir, "backend", "name.conf")).read_text().strip()
            for dir in backend_dirs
        ]
        return [
            BackendInstaller.prepare(
                backend_name, backend_src_dir=backend_src_dir, is_external=True
            )
            for backend_name, backend_src_dir in zip(backend_names, backend_dirs)
        ]


class Package(NamedTuple):
    package: str
    name: str
    url: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str


def is_linux_os(id):
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r") as f:
            os_release_content = f.read()
            return f'ID="{id}"' in os_release_content
    return False


# llvm
def get_llvm_package_info():
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[
            platform.machine()
        ]
    except KeyError:
        arch = platform.machine()
    if system == "Darwin":
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == 'arm64' and is_linux_os('almalinux'):
            system_suffix = 'almalinux-arm64'
        elif arch == "arm64":
            system_suffix = "ubuntu-arm64"
        elif arch == "x64":
            vglibc = tuple(map(int, platform.libc_ver()[1].split(".")))
            vglibc = vglibc[0] * 100 + vglibc[1]
            if vglibc > 228:
                # Ubuntu 24 LTS (v2.39)
                # Ubuntu 22 LTS (v2.35)
                # Ubuntu 20 LTS (v2.31)
                system_suffix = "ubuntu-x64"
            elif vglibc > 217:
                # Manylinux_2.28 (v2.28)
                # AlmaLinux 8 (v2.28)
                system_suffix = "almalinux-x64"
            else:
                # Manylinux_2014 (v2.17)
                # CentOS 7 (v2.17)
                system_suffix = "centos-x64"
        else:
            print(
                f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
            )
            return Package(
                "llvm",
                "LLVM-C.lib",
                "",
                "LLVM_INCLUDE_DIRS",
                "LLVM_LIBRARY_DIR",
                "LLVM_SYSPATH",
            )
    else:
        print(
            f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
        )
        return Package(
            "llvm",
            "LLVM-C.lib",
            "",
            "LLVM_INCLUDE_DIRS",
            "LLVM_LIBRARY_DIR",
            "LLVM_SYSPATH",
        )
    # use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    # release_suffix = "assert" if use_assert_enabled_llvm else "release"
    llvm_hash_path = os.path.join(root_dir, "llvm-hash.txt")
    with open(llvm_hash_path, "r") as llvm_hash_file:
        rev = llvm_hash_file.read(8)
    name = f"llvm-{rev}-{system_suffix}"
    url = f"https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz"
    return Package(
        "llvm", name, url, "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH"
    )


def is_offline_build() -> bool:
    """
    Downstream projects and distributions which bootstrap their own dependencies from scratch
    and run builds in offline sandboxes
    may set `TRITON_OFFLINE_BUILD` in the build environment to prevent any attempts at downloading
    pinned dependencies from the internet or at using dependencies vendored in-tree.

    Dependencies must be defined using respective search paths (cf. `syspath_var_name` in `Package`).
    Missing dependencies lead to an early abortion.
    Dependencies' compatibility is not verified.

    Note that this flag isn't tested by the CI and does not provide any guarantees.
    """
    return check_env_flag("TRITON_OFFLINE_BUILD", "")


def open_url(url):
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
    )
    headers = {
        "User-Agent": user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forever.
    return urllib.request.urlopen(request, timeout=300)


def get_triton_cache_path():
    user_home = os.getenv("TRITON_HOME")
    if not user_home:
        user_home = (
            os.getenv("HOME")
            or os.getenv("USERPROFILE")
            or os.getenv("HOMEPATH")
            or None
        )
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


def get_thirdparty_packages(packages: list):
    triton_cache_path = get_triton_cache_path()
    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(triton_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.name)
        if os.environ.get(p.syspath_var_name):
            package_dir = os.environ[p.syspath_var_name]
        version_file_path = os.path.join(package_dir, "version.txt")

        input_defined = p.syspath_var_name in os.environ
        input_exists = os.path.exists(version_file_path)
        input_compatible = input_exists and Path(version_file_path).read_text() == p.url

        if is_offline_build() and not input_defined:
            raise RuntimeError(
                f"Requested an offline build but {p.syspath_var_name} is not set"
            )
        if not is_offline_build() and not input_defined and not input_compatible:
            with contextlib.suppress(Exception):
                remove_directory(package_root_dir)
            os.makedirs(package_root_dir, exist_ok=True)
            print(f"downloading and extracting {p.url} ...")
            with open_url(p.url) as response:
                if p.url.endswith(".zip"):
                    file_bytes = BytesIO(response.read())
                    with zipfile.ZipFile(file_bytes, "r") as file:
                        file.extractall(path=package_root_dir)
                else:
                    with tarfile.open(fileobj=response, mode="r|*") as file:
                        file.extractall(path=package_root_dir)
            # write version url to package_dir
            with open(os.path.join(package_dir, "version.txt"), "w") as f:
                f.write(p.url)
        if p.include_flag:
            thirdparty_cmake_args.append(f"-D{p.include_flag}={package_dir}/include")
        if p.lib_flag:
            thirdparty_cmake_args.append(f"-D{p.lib_flag}={package_dir}/lib")
    return thirdparty_cmake_args


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(root_dir) / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


class CMakeExtension(Extension):
    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class BuildExt(build_ext):
    library_path = [
        "triton/_C/libtriton.so",
        "triton/backends/ascend/triton-adapter-opt",
    ]

    def finalize_options(self):
        super().finalize_options()
        self.inplace = False

    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        match = re.search(
            r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode()
        )
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def get_pybind11_cmake_args(self):
        pybind11_sys_path = get_env_with_keys(["PYBIND11_SYSPATH"])
        if pybind11_sys_path:
            pybind11_include_dir = os.path.join(pybind11_sys_path, "include")
        else:
            pybind11_include_dir = pybind11.get_include()
        return [f"-DPYBIND11_INCLUDE_DIR={pybind11_include_dir}"]

    def get_ext_name(self, index):
        assert index <= len(BuildExt.library_path), "Invalid index"
        return BuildExt.library_path[index].split("/")[-1]

    def get_ext_path(self, index):
        assert index <= len(BuildExt.library_path), "Invalid index"
        loc = BuildExt.library_path[index].rfind("/")
        return os.path.abspath(
            os.path.dirname(
                self.get_ext_fullpath(BuildExt.library_path[index][: loc + 1])
            )
        )

    def install_extension(self):
        for i in range(len(BuildExt.library_path)):
            shutil.copy(
                os.path.join(self.get_ext_path(i), self.get_ext_name(i)),
                os.path.join(root_dir, BuildExt.library_path[i]),
            )

    def build_extension(self, ext):
        cmake_dir = get_cmake_dir()
        lit_dir = shutil.which("lit")
        ninja_dir = shutil.which("ninja")

        thirdparty_cmake_args = get_thirdparty_packages([get_llvm_package_info()])
        thirdparty_cmake_args += self.get_pybind11_cmake_args()

        # python directories
        python_include_dir = sysconfig.get_path("platinclude")

        cmake_args = [
            "-G",
            "Ninja",  # Ninja is much faster than make
            "-DCMAKE_MAKE_PROGRAM=" + ninja_dir,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DLLVM_ENABLE_WERROR=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + self.get_ext_path(0),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + self.get_ext_path(1),
            "-DTRITON_BUILD_TUTORIALS=OFF",
            "-DTRITON_BUILD_PYTHON_MODULE=ON",
            "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
            "-DPYTHON_INCLUDE_DIRS=" + python_include_dir,
            "-DTRITON_CODEGEN_BACKENDS="
            + ";".join([b.name for b in _backends if not b.is_external]),
            "-DTRITON_PLUGIN_DIRS="
            + ";".join([b.src_dir for b in _backends if b.is_external]),
        ]
        if lit_dir is not None:
            cmake_args.append("-DLLVM_EXTERNAL_LIT=" + lit_dir)
        cmake_args.extend(thirdparty_cmake_args)

        cfg = get_build_type()
        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD"):
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld",
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]

        if check_env_flag("TRITON_BUILD_WITH_CCACHE"):
            cmake_args += [
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]

        cmake_args += ["-DTRITON_BUILD_PROTON=OFF"]
        cmake_args_append = os.getenv("TRITON_APPEND_CMAKE_ARGS")
        if cmake_args_append is not None:
            cmake_args += shlex.split(cmake_args_append)
        subprocess.check_call(
            ["cmake", root_dir] + cmake_args, cwd=cmake_dir, env=os.environ.copy()
        )

        # configuration
        build_args = ["--config", cfg]
        max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
        build_args += ["-j" + max_jobs]
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)

        self.install_extension()


class BuildInstall(install):
    def run(self):
        self.single_version_externally_managed = True
        super().run()


class BuildWheel(bdist_wheel):
    def run(self):
        bdist_wheel.run(self)

        if is_manylinux:
            file = glob.glob(os.path.join(self.dist_dir, "*-linux_*.whl"))[0]

            auditwheel_cmd = [
                "auditwheel",
                "-v",
                "repair",
                "--plat",
                f"manylinux_2_27_{platform.machine()}",
                "--plat",
                f"manylinux_2_28_{platform.machine()}",
                "-w",
                self.dist_dir,
                file,
            ]

            try:
                subprocess.run(auditwheel_cmd, check=True, stdout=subprocess.PIPE)
            finally:
                os.remove(file)


class BuildClean(clean):
    def run(self):
        self.clean_egginfo()

        remove_directory(os.path.join(root_dir, "triton"))
        remove_directory(os.path.join(root_dir, "build"))

    def clean_egginfo(self):
        egginfo_dir = os.path.join(root_dir, f"{get_package_name()}" + ".egg-info")

        remove_directory(egginfo_dir)


def get_language_extra_packages(backends):
    packages = []
    for backend in backends:
        if backend.language_dir is None:
            continue

        # Walk the `language` directory of each backend to enumerate
        # any subpackages, which will be added to `triton.language.extra`.
        for dir, _, files in os.walk(backend.language_dir, followlinks=True):
            if (
                not any(f for f in files if f.endswith(".py"))
                or dir == backend.language_dir
            ):
                # Ignore directories with no python files.
                # Also ignore the root directory which corresponds to
                # "triton/language/extra".
                continue
            subpackage = os.path.relpath(dir, backend.language_dir)
            package = os.path.join("triton/language/extra", subpackage)
            packages.append(package)

    return list(packages)


def get_packages(backends):
    packages = [
        "triton",
        "triton/_C",
        "triton/compiler",
        "triton/language",
        "triton/language/extra",
        "triton/runtime",
        "triton/backends",
        "triton/tools",
    ]
    packages += [f"triton/backends/{backend.name}" for backend in backends]
    packages += get_language_extra_packages(backends)
    packages += [
        "triton/triton_patch",
        "triton/triton_patch/language",
        "triton/triton_patch/compiler",
        "triton/triton_patch/runtime",
    ]

    return packages


def get_package_dir(backends):
    triton_prefix_dir = os.path.join(triton_dir, "python/triton")
    triton_patch_prefix_dir = os.path.join(root_dir, "triton_patch/python/triton_patch")

    # upstream triton
    package_dir = {
        "triton": f"{triton_prefix_dir}",
        "triton/_C": f"{triton_prefix_dir}/_C",
        "triton/backends": f"{triton_prefix_dir}/backends",
        "triton/compiler": f"{triton_prefix_dir}/compiler",
        "triton/language": f"{triton_prefix_dir}/language",
        "triton/language/extra": f"{triton_prefix_dir}/language/extra",
        "triton/runtime": f"{triton_prefix_dir}/runtime",
        "triton/tools": f"{triton_prefix_dir}/tools",
    }
    for backend in backends:
        package_dir[f"triton/backends/{backend.name}"] = (
            f"{triton_prefix_dir}/backends/{backend.name}"
        )
    language_extra_list = get_language_extra_packages(backends)
    for extra_full in language_extra_list:
        extra_name = extra_full.replace("triton/language/extra/", "")
        package_dir[extra_full] = f"{triton_prefix_dir}/language/extra/{extra_name}"

    # triton patch
    package_dir["triton/triton_patch"] = f"{triton_patch_prefix_dir}"
    package_dir["triton/triton_patch/language"] = f"{triton_patch_prefix_dir}/language"
    package_dir["triton/triton_patch/compiler"] = f"{triton_patch_prefix_dir}/compiler"
    package_dir["triton/triton_patch/runtime"] = f"{triton_patch_prefix_dir}/runtime"

    patch_paths = {
        "language/_utils.py",
        "compiler/compiler.py",
        "compiler/code_generator.py",
        "compiler/errors.py",
        "runtime/autotuner.py",
        "runtime/autotiling_tuner.py",
        "runtime/jit.py",
        "runtime/tile_generator.py",
        "runtime/utils.py",
        "runtime/libentry.py",
        "runtime/code_cache.py",
        "testing.py",
    }

    for path in patch_paths:
        package_dir[f"triton/{path}"] = f"{triton_patch_prefix_dir}/{path}"

    return package_dir


def get_package_data(backends):
    return {
        "triton/tools": ["compile.h", "compile.c"],
        **{f"triton/backends/{b.name}": b.package_data for b in backends},
        "triton/language/extra": sum((b.language_package_data for b in backends), []),
    }


def get_git_commit_hash(length=8):
    try:
        current_dir = os.getcwd()
        os.chdir(os.environ.get("TRITON_PLUGIN_DIRS", current_dir))

        cmd = ["git", "rev-parse", f"--short={length}", "HEAD"]
        git_commit_hash = subprocess.check_output(cmd).strip().decode("utf-8")
        os.chdir(current_dir)
        return "+git{}".format(git_commit_hash)
    except Exception:
        return ""


# temporary design
# Using version.txt containing version and commitid will be better and
# the version.txt will be converted to versin.py when compilation.
def get_version():
    version = os.environ.get("TRITON_VERSION", "3.2.0") + os.environ.get(
        "TRITON_WHEEL_VERSION_SUFFIX", ""
    )
    if not is_manylinux:
        version += get_git_commit_hash()

    return version


def get_package_name():
    return os.environ.get("TRITON_WHEEL_NAME", "triton_ascend")


def create_symlink_for_backend(backends):
    for backend in backends:
        if os.path.islink(backend.install_dir):
            os.unlink(backend.install_dir)
        remove_directory(backend.install_dir)
        os.symlink(backend.backend_dir, backend.install_dir)

        if backend.language_dir:
            # Link the contents of each backend's `language` directory into
            # `triton.language.extra`.
            extra_dir = os.path.abspath(
                os.path.join(triton_dir, "python", "triton", "language", "extra")
            )
            for x in os.listdir(backend.language_dir):
                src_dir = os.path.join(backend.language_dir, x)
                install_dir = os.path.join(extra_dir, x)
                if os.path.islink(install_dir):
                    os.unlink(install_dir)
                remove_directory(install_dir)
                os.symlink(src_dir, install_dir)


def create_symlink_for_triton(link_map):
    remove_directory(os.path.join(root_dir, "triton"))

    for target, source in link_map.items():
        target_path = Path(os.path.join(root_dir, target))
        source_path = Path(os.path.join(root_dir, source))

        if source_path.is_dir():
            os.makedirs(target_path, exist_ok=True)
            for src_file in source_path.glob("*"):
                if src_file.is_file():
                    dest_file = target_path / src_file.name
                    os.symlink(src_file, dest_file)
        elif source_path.is_file():
            if target_path.exists():
                os.unlink(target_path)
            os.symlink(source_path, target_path)
        else:
            print("[ERROR]: wrong file mapping")


is_manylinux = check_env_flag("IS_MANYLINUX", "FALSE")
readme = os.path.join(root_dir, "README.md")
if not os.path.exists(readme):
    raise FileNotFoundError("Unable to find 'README.md'")
with open(readme, encoding="utf-8") as fdesc:
    long_description = fdesc.read()

_backends = [*BackendInstaller.copy_externals()]
create_symlink_for_backend(_backends)
create_symlink_for_triton(get_package_dir(_backends))


setup(
    name=get_package_name(),
    version=get_version(),
    description="A language and compiler for custom Deep Learning operations on Ascend hardwares",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=get_packages(_backends),
    package_data=get_package_data(_backends),
    include_package_data=True,
    ext_modules=[CMakeExtension("triton", "triton/_C/")],
    cmdclass={
        "build_ext": BuildExt,
        "install": BuildInstall,
        "bdist_wheel": BuildWheel,
        "clean": BuildClean,  # type: ignore[misc]
    },
    zip_safe=False,
    # for PyPI
    keywords=["Compiler", "Deep Learning"],
    url="https://gitee.com/ascend/triton-ascend/",
)
