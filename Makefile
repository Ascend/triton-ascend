# Helper Makefile for development, testing, and packaging.
# Run `make help` before using.

# Configurable variables
PYTHON                      ?= python3
PYTEST                      := $(PYTHON) -m pytest
NUM_PROCS                   ?= $(shell expr $(shell nproc) / 3)
OS_ID                       := $(shell . /etc/os-release && echo $$ID)
ARCH                        := $(shell uname -i)
ARCH_NAME                   := $(subst x86_64,x64,$(subst aarch64,arm64,$(ARCH)))
PLATFORM_NAME               := $(subst x86_64,amd64,$(subst aarch64,arm64,$(ARCH)))
LLVM_REPO                   := https://github.com/llvm/llvm-project.git
LLVM_DIR                    := llvm-project
LLVM_COMMIT                 := $(shell cat llvm-hash.txt)
LLVM_COMMIT_SHORT           := $(shell cut -c1-8 llvm-hash.txt)
LLVM_INSTALL_DIR            := llvm-$(LLVM_COMMIT_SHORT)-$(OS_ID)-$(ARCH_NAME)
LLVM_TARBALL                := $(LLVM_INSTALL_DIR).tar.gz
SUDO                        := $(shell command -v sudo >/dev/null 2>&1 && echo sudo || echo)
TOOLKIT_URL                 := https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com/cann/Ascend-cann-toolkit_8.2.RC1_linux-$(ARCH).run
KERNEL_URL                  := https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com/cann/Ascend-cann-kernels-910b_8.2.RC1_linux-$(ARCH).run
CANN_TOOLKIT                := Ascend-cann-toolkit.run
CANN_KERNELS                := Ascend-cann-kernels.run
DEPS_STAMP                  := .deps_installed
REQ_RT_FILE                 := requirements.txt
REQ_DEV_FILE                := requirements_dev.txt
REQ_RT_STAMP                := .req_rt_installed
REQ_DEV_STAMP               := .req_dev_installed
REQ_RT_HASH                 := $(shell md5sum $(REQ_RT_FILE) 2>/dev/null | cut -d' ' -f1)
REQ_DEV_HASH                := $(shell md5sum $(REQ_DEV_FILE) 2>/dev/null | cut -d' ' -f1)
OBSUTIL_DIR                 := obsutil
OBSUTIL_TAR                 := obsutil.tar.gz
OBSUTIL_URL                 := https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_$(PLATFORM_NAME).tar.gz
OBSUTIL_CONFIG              := ~/.obsutilconfig
TRITON_WHL                  := dist/triton_ascend-*.whl
NINJA_TAR                   := $(shell if [ "`uname -m`" = "aarch64" ]; then echo "ninja-linux-aarch64"; else echo "ninja-linux"; fi)
NINJA_URL                   := https://github.com/ninja-build/ninja/releases/download/v1.13.1/$(NINJA_TAR).zip
IS_MANYLINUX                ?= False
PYPI_URL                    ?= testpypi
TRITON_WHEEL_VERSION_SUFFIX ?= rc3
PYPI_CONFIG                 := ~/.pypirc


.DEFAULT_GOAL := all

# ======================
# Help
# ======================
.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v '^_' | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'


# ======================
# Build: Triton
# ======================
.PHONY: all
all: ## Incremental builds
	@BUILD_DIR=$$($(PYTHON) -c "import sysconfig, sys; plat_name=sysconfig.get_platform(); python_version=sysconfig.get_python_version(); print(f'build/cmake.{plat_name}-{sys.implementation.name}-{python_version}')"); \
	echo "Using build dir: $$BUILD_DIR"; \
	ninja -C $$BUILD_DIR

$(TRITON_WHL): $(DEPS_STAMP) install-dev-reqs
	@echo "Building Triton wheel..."
	@if [ -n "$$HEAD_COMMIT" ]; then \
		echo "Checking out to HEAD_COMMIT: $$HEAD_COMMIT"; \
		git checkout $$HEAD_COMMIT || exit 1; \
	fi && \
	TRITON_PLUGIN_DIRS=./ascend \
	TRITON_BUILD_WITH_CLANG_LLD=true \
	TRITON_BUILD_PROTON=OFF \
	TRITON_WHEEL_NAME="triton-ascend" \
	TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
	MAX_JOBS=$(NUM_PROCS) \
	IS_MANYLINUX=$(IS_MANYLINUX) \
	TRITON_WHEEL_VERSION_SUFFIX=$(TRITON_WHEEL_VERSION_SUFFIX) \
	$(PYTHON) setup.py bdist_wheel

.PHONY: package
package: $(TRITON_WHL) ## Build the Triton wheel package

.PHONY: upload-triton
upload-triton: install-obsutil package
	@WHEEL=$(wildcard dist/*.whl); \
	if [ -z "$$WHEEL" ]; then \
		echo "No wheel found after build"; exit 1; \
	fi; \
	echo "Uploading $$WHEEL to OBS..."; \
	obsutil/obsutil cp -u "$$WHEEL" obs://triton-ascend-artifacts/triton-ascend/

.PHONY: triton
triton: upload-triton ## Build and upload Triton wheel to OBS

.PHONY: upload-pypi
upload-pypi: $(PYPI_CONFIG) install-deps ## Build and upload Triton wheel to PyPI
	for PY in python3.9 python3.10 python3.11; do \
		echo "Building wheel for $$PY..."; \
		rm -rf build dist; \
		make package PYTHON=$$PY IS_MANYLINUX=True; \
		WHEEL=$$(ls dist/*.whl); \
		echo "Uploading $$WHEEL to $(PYPI_URL)..."; \
		$$PY -m twine upload --repository $(PYPI_URL) $$WHEEL; \
		[[ $$? -ne 0 ]] && exit 1; \
		rm -f .req_dev_installed; \
	done

# ======================
# Build: LLVM
# ======================
$(LLVM_DIR): $(DEPS_STAMP)
	@git clone --no-checkout $(LLVM_REPO) $(LLVM_DIR)
	cd $(LLVM_DIR) && git checkout $(LLVM_COMMIT)

.PHONY: clone-llvm
clone-llvm: $(LLVM_DIR) ## Clone LLVM repo at specified commit

$(LLVM_TARBALL): clone-llvm ## Build LLVM and package tarball
	@set -e; \
	echo "Building LLVM to $(LLVM_INSTALL_DIR)..."; \
	\
	if [ "$(OS_ID)" = "ubuntu" ]; then \
		$(PYTHON) -m pip install -r llvm-project/mlir/python/requirements.txt; \
		cmake -GNinja -Bllvm-project/build \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_C_COMPILER=clang \
			-DCMAKE_CXX_COMPILER=clang++ \
			-DCMAKE_ASM_COMPILER=clang \
			-DCMAKE_LINKER=lld \
			-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR) \
			-DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
			-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
			-DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
			-DLLVM_INSTALL_UTILS=ON \
			-DLLVM_ENABLE_ASSERTIONS=ON \
			-DLLVM_ENABLE_TERMINFO=OFF \
			-DLLVM_BUILD_UTILS=ON \
			-DLLVM_BUILD_TOOLS=ON \
			llvm-project/llvm; \
	elif [ "$(OS_ID)" = "almalinux" ]; then \
		/opt/python/cp38-cp38/bin/python3 -m pip install -r llvm-project/mlir/python/requirements.txt; \
		PATH=/opt/python/cp38-cp38/bin:$$PATH; \
		cmake -GNinja -Bllvm-project/build \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_C_COMPILER=clang \
			-DCMAKE_CXX_COMPILER=clang++ \
			-DCMAKE_ASM_COMPILER=clang \
			-DCMAKE_LINKER=lld \
			-DCMAKE_CXX_FLAGS="-Wno-everything" \
			-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR) \
			-DPython3_EXECUTABLE=/opt/python/cp38-cp38/bin/python3 \
			-DLLVM_ENABLE_PROJECTS="mlir;lld" \
			-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
			-DLLVM_INSTALL_UTILS=ON \
			-DLLVM_ENABLE_ASSERTIONS=ON \
			-DLLVM_ENABLE_TERMINFO=OFF \
			-DLLVM_BUILD_UTILS=ON \
			-DLLVM_BUILD_TOOLS=ON \
			-DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
			llvm-project/llvm; \
	else \
		echo "Unsupported OS: $(OS_ID)"; exit 1; \
	fi; \
	ninja -j $(NUM_PROCS) -C llvm-project/build install; \
	tar czf $(LLVM_TARBALL) $(LLVM_INSTALL_DIR)

.PHONY: build-llvm
build-llvm: $(LLVM_TARBALL) ## Clone LLVM repo at specified commit

.PHONY: upload-llvm ## Upload LLVM to OBS
upload-llvm: install-obsutil $(LLVM_TARBALL)
	@echo "Uploading $(LLVM_TARBALL) to OBS..."
	obsutil/obsutil cp -u "$(LLVM_TARBALL)" obs://triton-ascend-artifacts/llvm-builds/

.PHONY: llvm
llvm: ## Conditional build and upload of LLVM
	@if [ -z "$(BASE_COMMIT)" ]; then \
		echo "BASE_COMMIT not set, forcing LLVM upload..."; \
		$(MAKE) upload-llvm; \
	elif [ -n "$$filenames" ] && echo "$$filenames" | grep -q '\bllvm-hash\.txt\b'; then \
		echo "llvm-hash.txt changed. Uploading LLVM..."; \
		$(MAKE) upload-llvm; \
	else \
		echo "No changes to llvm-hash.txt. Skipping upload."; \
	fi


# ======================
# Tests
# ======================
.PHONY: test-unit
test-unit: ## Run unit tests
	cd ascend/examples/pytest_ut && $(PYTEST) -s -v -n $(NUM_PROCS) --dist=loadfile

.PHONY: test-inductor
test-inductor: ## Run inductor tests
	cd ascend/examples/inductor_cases && $(PYTEST) -s -v -n $(NUM_PROCS) --dist=load

.PHONY: test-gen
test-gen: ## Run generalization tests
	cd ascend/examples/generalization_cases && $(PYTEST) -s -v -n $(NUM_PROCS) --dist=load


# ======================
# Image Build
# ======================
.PHONY: image
image: ## Build dev Docker image if relevant files changed
	@set -e; \
	if [ -n "$$HEAD_COMMIT" ]; then \
		GIT_COMMIT_SHORT=$$(git rev-parse --short $$HEAD_COMMIT); \
	else \
		GIT_COMMIT_SHORT=$$(git rev-parse --short HEAD); \
	fi; \
	if [ -z "$(BASE_COMMIT)" ]; then \
		echo "BASE_COMMIT not set. Forcing Docker image build..."; \
		BUILD_IMAGE=1; \
	elif [ -n "$$filenames" ] && echo "$$filenames" | grep -Eq '\b(docker/Dockerfile|Makefile|requirements(_dev)?\.txt)\b'; then \
		echo "Relevant files changed. Building Docker image..."; \
		BUILD_IMAGE=1; \
	else \
		echo "No relevant changes since BASE_COMMIT. Skipping Docker image build."; \
		BUILD_IMAGE=0; \
	fi; \
	if [ $$BUILD_IMAGE -eq 1 ]; then \
		if [ -z "$(QUAY_PASSWD)" ] || [ -z "$(QUAY_USER)" ]; then \
			echo "Please set QUAY_USER and QUAY_PASSWD before building the image."; \
			exit 1; \
		fi; \
		echo "Logging in to Docker..."; \
		echo "$(QUAY_PASSWD)" | docker login -u "$(QUAY_USER)" --password-stdin quay.io; \
		echo "Using commit ID: $$GIT_COMMIT_SHORT"; \
		docker buildx build --platform linux/$(PLATFORM_NAME) \
			-f docker/Dockerfile --push \
			-t quay.io/ascend/triton:dev-$$GIT_COMMIT_SHORT-$(PLATFORM_NAME) .; \
	fi

.PHONY: create-image
create-image: ## Create multi-arch manifest image
	@set -e; \
	if [ -z "$$HEAD_COMMIT" ]; then \
		GIT_COMMIT_SHORT=$$(git rev-parse --short HEAD); \
	else \
		GIT_COMMIT_SHORT=$$(git rev-parse --short $$HEAD_COMMIT); \
	fi; \
	echo "Checking for per-arch images with tag dev-$$GIT_COMMIT_SHORT"; \
	ARCHS="amd64 arm64"; \
	MISSING=0; \
	for arch in $$ARCHS; do \
		TAG="dev-$$GIT_COMMIT_SHORT-$$arch"; \
		if curl -sfI "https://quay.io/v2/ascend/triton/manifests/$$TAG" > /dev/null; then \
			echo "Found tag: $$TAG"; \
		else \
			echo "Missing tag: $$TAG"; \
			MISSING=1; \
		fi; \
	done; \
	if [ "$$MISSING" -eq 0 ]; then \
		echo "Creating multi-arch image: dev-$$GIT_COMMIT_SHORT"; \
		docker buildx imagetools create \
			--tag quay.io/ascend/triton:dev-$$GIT_COMMIT_SHORT \
			quay.io/ascend/triton:dev-$$GIT_COMMIT_SHORT-amd64 \
			quay.io/ascend/triton:dev-$$GIT_COMMIT_SHORT-arm64; \
	else \
		echo "Skipping manifest creation due to missing images."; \
	fi


# ======================
# CANN Installation
# ======================
.PHONY: install-cann
install-cann: $(CANN_TOOLKIT) $(CANN_KERNELS) ## Download and install CANN
	chmod +x $^
	./$(CANN_TOOLKIT) --full --quiet
	./$(CANN_KERNELS) --install --quiet

$(CANN_TOOLKIT):
	@echo "Downloading $(CANN_TOOLKIT)..."
	curl -sSL "$(TOOLKIT_URL)" -o $@

$(CANN_KERNELS):
	@echo "Downloading $(CANN_KERNELS)..."
	curl -sSL "$(KERNEL_URL)" -o $@


# ======================
# Environment Setup
# ======================
.PHONY: install-deps
install-deps: $(DEPS_STAMP) ## Install OS-level dependencies

$(DEPS_STAMP):
ifeq ($(OS_ID),ubuntu)
	@echo "Installing dependencies for Ubuntu..."
	$(SUDO) apt-get update
	$(SUDO) apt-get install --yes --no-install-recommends \
		ca-certificates ccache clang ninja-build libzstd-dev \
	        lld git python3 python3-dev python3-pip zlib1g-dev
	@python3 -m pip install cmake ninja
else ifeq ($(OS_ID),almalinux)
	@echo "Installing dependencies for AlmaLinux..."
	dnf install --assumeyes clang lld cmake ccache git

	# In AlmaLinux, the ninja version provided by dnf is too low.
	curl -Lo $(NINJA_TAR).zip $(NINJA_URL)
	unzip $(NINJA_TAR).zip && install -m 755 ninja /usr/local/bin/
	rm -f $(NINJA_TAR).zip ninja
else
	@echo "Unsupported OS: $(OS_ID)"
	exit 1
endif
	@touch $(DEPS_STAMP)

.PHONY: install-rt-reqs
install-rt-reqs: ## Install triton runtime requirements
	@if [ ! -f $(REQ_RT_STAMP) ] || [ "$(shell cat $(REQ_RT_STAMP))" != "$(REQ_RT_HASH)" ]; then \
		echo "Installing runtime requirements..."; \
		$(PYTHON) -m pip install -r $(REQ_RT_FILE); \
		echo "$(REQ_RT_HASH)" > $(REQ_RT_STAMP); \
	else \
		echo "Runtime requirements already satisfied."; \
	fi

.PHONY: install-dev-reqs
install-dev-reqs: ## Install triton development requirements
	@if [ ! -f $(REQ_DEV_STAMP) ] || [ "$(shell cat $(REQ_DEV_STAMP))" != "$(REQ_DEV_HASH)" ]; then \
		echo "Installing development requirements..."; \
		$(PYTHON) -m pip install -r $(REQ_DEV_FILE); \
		echo "$(REQ_DEV_HASH)" > $(REQ_DEV_STAMP); \
	else \
		echo "Development requirements already satisfied."; \
	fi

.PHONY: install-obsutil ## Install obsutil
install-obsutil: $(OBSUTIL_DIR) $(OBSUTIL_CONFIG) ## Install Huawei OBS utility (obsutil)

# Download obsutil
$(OBSUTIL_DIR):
	@echo "Downloading obsutil..."
	curl -L -o $(OBSUTIL_TAR) $(OBSUTIL_URL)
	tar xzf $(OBSUTIL_TAR)
	rm -f $(OBSUTIL_TAR)
	mv obsutil* $@

# Config obsutil
$(OBSUTIL_CONFIG): $(OBSUTIL_DIR)
	@if [ -z "$$AK" ] || [ -z "$$SK" ]; then \
		echo "Please set AK and SK environment variables before uploading."; \
		exit 1; \
	fi
	$(OBSUTIL_DIR)/obsutil config -i=$(AK) -k=$(SK) -e=https://obs.cn-southwest-2.myhuaweicloud.com

# Config pypi
$(PYPI_CONFIG):
	@if [ -z "$$PASSWORD" ]; then \
		echo "Please set PASSWORD environment variable before uploading to $(PYPI_URL)."; \
		exit 1; \
	fi
	@echo "[$(PYPI_URL)]"         >  $@
	@echo "  username = __token__" >> $@
	@echo "  password = $$PASSWORD" >> $@


# ======================
# Clean
# ======================
.PHONY: clean
clean: ## Remove build and temporary files
	$(PYTHON) setup.py clean
	$(SUDO) rm -rf $(OBSUTIL_DIR) $(OBSUTIL_CONFIG) $(REQ_RT_STAMP) $(REQ_DEV_STAMP) $(DEPS_STAMP) \
	 $(LLVM_INSTALL_DIR) $(LLVM_TARBALL) $(LLVM_DIR) $(CANN_TOOLKIT) $(CANN_KERNELS) $(PYPI_CONFIG)
