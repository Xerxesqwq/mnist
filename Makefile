.PHONY: all clean download_libtorch distclean

LIBTORCH_DIR = libtorch
LIBTORCH_URL_CPU = https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
LIBTORCH_URL_CUDA = https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
BUILD_DIR = build

all: resnet.out

$(LIBTORCH_DIR):
	@echo "Downloading LibTorch..."
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "CUDA detected, downloading CUDA-enabled LibTorch..."; \
		wget -q --show-progress $(LIBTORCH_URL_CUDA) -O libtorch.zip; \
	else \
		echo "CUDA not detected, downloading CPU-only LibTorch..."; \
		wget -q --show-progress $(LIBTORCH_URL_CPU) -O libtorch.zip; \
	fi
	@echo "Extracting LibTorch..."
	@unzip -q libtorch.zip
	@rm libtorch.zip
	@echo "LibTorch downloaded and extracted successfully"

download_libtorch: $(LIBTORCH_DIR)

$(BUILD_DIR)/Makefile: $(LIBTORCH_DIR)
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_PREFIX_PATH=$(shell pwd)/$(LIBTORCH_DIR) ..

$(BUILD_DIR)/resnet.out: $(BUILD_DIR)/Makefile resnet.cpp
	@cd $(BUILD_DIR) && $(MAKE)

resnet.out: $(BUILD_DIR)/resnet.out
	@cp $(BUILD_DIR)/resnet.out .
	@echo "Build complete: resnet.out"

clean:
	@rm -rf $(BUILD_DIR) resnet.out
	@echo "Cleaned build artifacts"

distclean: clean
	@rm -rf $(LIBTORCH_DIR) libtorch.zip
	@echo "Cleaned everything including LibTorch"
